import os
import shutil
import subprocess
import tempfile

from jinja2 import Environment, FileSystemLoader

from core.business.base.web_app.base_web_app_plugin import BaseWebAppPlugin as BasePlugin
from core.utils.uvicorn_fast_api_ipc_manager import get_server_manager

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePlugin.CONFIG,
  'USE_NGROK' : False,
  'NGROK_DOMAIN' : None,

  'PORT' : 8080,

  'ASSETS' : None,
  'JINJA_ARGS' : {},
  'TEMPLATE' : 'basic_server',

  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES']
  },
}

class FastApiWebAppPlugin(BasePlugin):
  """
  A plugin which exposes all of its methods marked with @endpoint through
  fastapi as http endpoints.

  The @endpoint methods can be triggered via http requests on the web server
  and will be processed as part of the business plugin loop.
  """

  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self.uvicorn_process = None
    super(FastApiWebAppPlugin, self).__init__(**kwargs)
    return

  @staticmethod
  def endpoint(func, method="get"):
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
        src_file_path = self.os_path.join(root, file)
        dst_file_path = self.os_path.join(
          dst_dir,
          self.os_path.relpath(src_file_path, src_dir)
        )

        # If we have a symlink don't do anything.
        if self.os_path.islink(src_file_path):
          continue

        # Make sure the destination directory exists.
        os.makedirs(self.os_path.dirname(dst_file_path), exist_ok=True)

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
      template_dir = self.os_path.join('core', 'business', 'base', 'uvicorn_templates')
      app_template = self.os_path.join(template_dir, f'{self.cfg_template}.jinja')
      app_template = env.get_template(app_template)
      rendered_content = app_template.render(jinja_args)

      with open(self.os_path.join(dst_dir, 'main.py'), 'w') as f:
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
    super(FastApiWebAppPlugin, self).on_init()
    # Register all endpoint methods.
    self._init_endpoints()

    # FIXME: move to setup_manager method
    manager_auth = b'abc'
    self._manager = get_server_manager(manager_auth)

    self.P("manager address: {}",format(self._manager.address))
    _, manager_port = self._manager.address

    # Start the FastAPI app
    self.P('Starting FastAPI app...')
    script_temp_dir = tempfile.mkdtemp()
    script_path = self.os_path.join(script_temp_dir, 'main.py')
    self.P("Using script at {}".format(script_path))

    src_dir = self.os_path.join('plugins', 'business', 'fastapi', self.cfg_assets)

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
      str(self.cfg_port)
    ]
    env = self.deepcopy(os.environ)
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


    # Teardown communicator
    self._manager.shutdown()
    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.

    super(FastApiWebAppPlugin, self).on_close()
    return
