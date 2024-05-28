import copy
import subprocess
import time
import os
import tempfile
import shutil
from jinja2 import Environment, FileSystemLoader

from core.utils.ngrok_ipc_manager import get_server_manager

from core.business.base import BasePluginExecutor

__VER__ = '0.0.0.0'

_CONFIG = {
  'NGROK_PORT' : 8080,
  'NGROK_DOMAIN' : None,
  'ASSETS' : None,
  'JINJA_ARGS' : {},
  'TEMPLATE' : 'basic_server',
  **BasePluginExecutor.CONFIG,
  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES']
  },
}

class NgrokCT:
  NG_TOKEN = 'NGROK_AUTH_TOKEN'
  NG_DOMAIN = 'NGROK_DOMAIN'
  NG_EDGE_LABEL = 'NGROK_EDGE_LABEL'
  HTTP_GET = 'get'
  HTTP_PUT = 'put'
  HTTP_POST = 'post'

class BaseNgrokPlugin(BasePluginExecutor):
  """
  A plugin which exposes all of its methods marked with @endpoint through
  fastapi as http endpoints, and further tunnels traffic to this interface
  via ngrok.

  The @endpoint methods can be triggered via http requests on the web server
  and will be processed as part of the business plugin loop.
  """

  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self.ngrok_process = None
    self.uvicorn_process = None
    super(BaseNgrokPlugin, self).__init__(**kwargs)
    return

  @property
  def ng_token(self):
    return self.os_environ.get(NgrokCT.NG_TOKEN, None)

  @property
  def ng_domain(self):
    return self.os_environ.get(NgrokCT.NG_DOMAIN, None)

  @property
  def ng_edge(self):
    return self.os_environ.get(NgrokCT.NG_EDGE_LABEL, None)

  @staticmethod
  def endpoint(func, method=NgrokCT.HTTP_GET):
    """
    Decorator, marks the method as being exposed as an endpoint.
    """
    func.__endpoint__ = True
    func.__http_method__ = method
    return func

  def _initialize_assets(self, src_dir, dst_dir, jinja_args):
    """
    Initialize and copy fastapi assets, expanding any jinja templates.
    All files from the source directory are copied copied to the
    destination directory with the following exceptions:
      - are symbolic links are ignored
      - files named ending with .jinja are expanded as jinja templates,
        .jinja is removed from the filename and the result copied to
        the destination folder.
    This maintains the directory structure of the source folder.

    Parameters
    ----------
    src_dir: str, path to the source directory
    dst_dir: str, path to the destination directory
    jinja_args: dict, jinja keys to use while expanding the templates

    Returns
    -------
    None
    """
    self.P(f'Copying uvicorn assets from {src_dir} to {dst_dir} with keys {jinja_args}')

    env = Environment(loader=FileSystemLoader('.'))
    # Walk through the source directory.
    for root, _, files in os.walk(src_dir):
      for file in files:
        src_file_path = os.path.join(root, file)
        dst_file_path = os.path.join(
          dst_dir,
          os.path.relpath(src_file_path, src_dir)
        )

        # If we have a symlink don't do anything.
        if os.path.islink(src_file_path):
          continue

        # Make sure the destination directory exists.
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

        # If this file is a jinja template render it to a file with the
        # .jinja suffix removed.
        if src_file_path.endswith(('.jinja')):
          dst_file_path = dst_file_path[:-len('.jinja')]
          template = env.get_template(src_file_path)
          rendered_content = template.render(jinja_args)
          with open(dst_file_path, 'w') as f:
            f.write(rendered_content)
          continue
        #endif jinja template

        # This is not a jinja template, just copy it do the destination
        # folder as is.
        shutil.copy2(src_file_path, dst_file_path)
      #endfor all files
    #endfor os.walk

    if self.cfg_template is not None:
      # Finally render main.py
      template_dir = os.path.join('core', 'business', 'base', 'ngrok_templates')
      app_template = os.path.join(template_dir, f'{self.cfg_template}.jinja')
      app_template = env.get_template(app_template)
      rendered_content = app_template.render(jinja_args)

      with open(os.path.join(dst_dir, 'main.py'), 'w') as f:
        f.write(rendered_content)
    #endif render main.py

    return

  def get_jinja_template_args(self) -> dict:
    """
    Produces the dictionary of arguments that is used for  jinja template
    expansion. These arguments will be used while expanding all the jinja
    templates.

    Users can overload this to pass in custom arguments.

    Parameters
    ----------
    None

    Returns
    -------
    dict - dictionary of arguments that is used for jinja template
           expansion.
    """
    return self.cfg_jinja_args

  def _init_endpoints(self) -> None:
    """
    Populate the set of jinja arguments with values needed to create http
    endpoints for all methods of the plugin marked with @endpoint. Since
    there should be at least one such method, this method is always invoked
    via the on_init hook

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    import inspect
    self._endpoints = {}
    jinja_args = []
    def _filter(obj):
      try:
        return inspect.ismethod(obj)
      except Exception as _:
        pass
      return False

    for name, method in inspect.getmembers(self, predicate=_filter):
      if not hasattr(method, '__endpoint__'):
        continue
      self._endpoints[name] = method
      http_method = method.__http_method__
      signature = inspect.signature(method)
      args = [param.name for param in signature.parameters.values()]
      jinja_args.append({
        'name' : name,
        'method' : http_method,
        'args' : args
      })
    #endfor all methods
    self._node_comms_jinja_args = {
      'node_comm_params' : jinja_args
    }
    return

  def on_init(self):
    port = self.cfg_ngrok_port
    super(BaseNgrokPlugin, self).on_init()

    # Register all endpoint methods.
    self._init_endpoints()

    # Check the config as we're going to use it to start processes.
    if not isinstance(port, int):
      raise ValueError("Ngrok port not an int")
    if port < 0 or port > 65535:
      raise ValueError("Invalid port value {}".format(port))

    # FIXME: move to setup_manager method
    manager_auth = b'abc'
    self._manager = get_server_manager(manager_auth)

    self.P("manager address: {}",format(self._manager.address))
    _, manager_port = self._manager.address

    ngrok_auth_args = ['ngrok', 'authtoken', self.ng_token]
    self.P('Trying to authenticate ngrok')
    auth_process = subprocess.Popen(ngrok_auth_args, stdout=subprocess.DEVNULL)

    # Wait for the process to finish and get the exit code
    if auth_process.wait(timeout=10) != 0:
      raise RuntimeError('Could not authenticate ngrok')
    self.P('Successfully authenticated ngrok.', color='g')

    # Start ngrok in the background.
    self.P('Starting ngrok on port {} with domain {}'.format(port, self.ng_domain))
    if self.ng_edge is not None:
      # A domain was specified in the env, just start on that domain.
      edge_label = self.ng_edge
      ngrok_start_args = [
        'ngrok',
        'tunnel',
        str(port),
        '--label',
        f'edge={edge_label}'
      ]
    elif self.ng_domain is not None:
      ngrok_start_args = [
        'ngrok',
        'http',
        str(port),
        '--domain=' + self.ng_domain
      ]
    else:
      raise RuntimeError("No domain/edge specified")
    #endif

    self.ngrok_process = subprocess.Popen(ngrok_start_args, stdout=subprocess.DEVNULL)
    #FIXME: this returns the pid, but what happens in case of error?
    self.P('ngrok started with PID {}'.format(self.ngrok_process.pid), color='g')

    # Wait until ngrok has started.
    self.P('Waiting for ngrok to start...', color='b')
    time.sleep(10)

    # FIXME: does this assumes linux (even if we run with docker)?
    if not os.path.exists('/proc/' + str(self.ngrok_process.pid)):
      raise RuntimeError('Failed to start ngrok')
    self.P('ngrok started successfully."', color='g')

    # Start the FastAPI app
    self.P('Starting FastAPI app...')
    script_temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(script_temp_dir, 'main.py')
    self.P("Using script at {}".format(script_path))

    src_dir = os.path.join('plugins', 'business', 'ngrok', self.cfg_assets)

    jinja_args = {
      **self.get_jinja_template_args(),
      'manager_port' : manager_port,
      'manager_auth' : manager_auth,
      **self._node_comms_jinja_args
    }
    self._initialize_assets(src_dir, script_temp_dir, jinja_args)

    # Set up the uvicorn environment and process. We want it to have access to
    # our class definitions so we need to set PYTHONPATH. Additionally we set
    # PWD and cwd to the folder containing the fastapi script and assets so we
    uvicorn_args = [
      'uvicorn',
      '--app-dir',
      script_temp_dir,
      'main:app',
      '--host',
      '0.0.0.0',
      '--port',
      str(port)
    ]
    env = copy.deepcopy(os.environ)
    env['PYTHONPATH'] = '.:' + os.getcwd() + ':' + env.get('PYTHONPATH', '')
    env['PWD'] = script_temp_dir
    self.uvicorn_process = subprocess.Popen(
      uvicorn_args,
      env=env,
      cwd=script_temp_dir
    )
    return

  def process(self):
    while not self._manager.get_server_queue().empty():
      request = self._manager.get_server_queue().get()
      id = request['id']
      value = request['value']

      method = value[0]
      args = value[1:]

      try:
        value = self._endpoints.get(method)(*args)
      except Exception as _:
        value = None

      response = {
        'id'    : id,
        'value' : value
      }
      self._manager.get_client_queue().put(response)
    #end while

    return None

  def on_close(self):
    # Teardown uvicorn
    try:
      # FIXME: there must be a clean way to do this.
      self.P("Forcefully killing uvicorn server")
      self.uvicorn_process.kill()
      self.P("Killed uvicorn server")
    except Exception as _:
      self.P('Could not kill uvicorn server')

    # Teardown ngrok
    try:
      self.P('Killing ngrok..')
      self.ngrok_process.kill()
      self.P('Killed ngrok')
    except Exception as _:
      self.P('Could not kill ngrok server')

    # Teardown communicator
    self._manager.shutdown()
    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.
    return
