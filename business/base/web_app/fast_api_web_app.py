import os

from jinja2 import Environment, FileSystemLoader

from core.business.base.web_app.base_web_app_plugin import BaseWebAppPlugin as BasePlugin
from core.utils.uvicorn_fast_api_ipc_manager import get_server_manager

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePlugin.CONFIG,
  'NGROK_ENABLED': True,
  'NGROK_DOMAIN': None,
  'NGROK_EDGE_LABEL': None,

  'PORT': None,

  'ASSETS': None,
  'JINJA_ARGS': {},
  'TEMPLATE': 'basic_server',

  'API_TITLE': None,  # default is plugin signature
  'API_SUMMARY': None,  # default is f"FastAPI created by {plugin signature}"
  'API_DESCRIPTION': None,  # default is plugin docstring

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

  @staticmethod
  def endpoint(func=None, *, method="get"):
    """
    Decorator, marks the method as being exposed as an endpoint.
    """
    if func is None:
      def wrapper(func):
        return FastApiWebAppPlugin.endpoint(func, method=method)

      return wrapper

    func.__endpoint__ = True
    func.__http_method__ = method
    return func

  def get_web_server_path(self):
    return self.script_temp_dir

  def initialize_assets(self, src_dir, dst_dir, jinja_args):
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
    super(FastApiWebAppPlugin, self).initialize_assets(src_dir, dst_dir, jinja_args)

    env = Environment(loader=FileSystemLoader('.'))

    # make sure assets folder exists
    os.makedirs(self.os_path.join(dst_dir, 'assets'), exist_ok=True)

    if self.cfg_template is not None:
      # Finally render main.py
      template_dir = self.os_path.join('core', 'business', 'base', 'uvicorn_templates')
      app_template = self.os_path.join(template_dir, f'{self.cfg_template}.jinja')
      # env.get_template expects forward slashes, even on Windows.
      app_template = app_template.replace(os.sep, '/')
      app_template = env.get_template(app_template)
      rendered_content = app_template.render(jinja_args)

      with open(self.os_path.join(dst_dir, 'main.py'), 'w') as f:
        f.write(rendered_content)
    # endif render main.py

    return

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
      params = [param.name for param in signature.parameters.values() if param.name != 'body']
      args = [str(param) for param in signature.parameters.values() if param.name != 'body']
      jinja_args.append({
        'name': name,
        'method': http_method,
        'args': args,
        'params': params
      })
    # endfor all methods
    self._node_comms_jinja_args = {
      'node_comm_params': jinja_args
    }
    return

  def on_init(self):
    # Register all endpoint methods.
    self._init_endpoints()

    # FIXME: move to setup_manager method
    self.manager_auth = b'abc'
    self._manager = get_server_manager(self.manager_auth)

    self.P("manager address: {}", format(self._manager.address))
    _, self.manager_port = self._manager.address

    self.prepared_env['PYTHONPATH'] = '.:' + os.getcwd() + ':' + self.prepared_env.get('PYTHONPATH', '')

    # Start the FastAPI app
    self.P('Starting FastAPI app...')
    super(FastApiWebAppPlugin, self).on_init()
    return

  def _process(self):
    super(FastApiWebAppPlugin, self)._process()
    while not self._manager.get_server_queue().empty():
      request = self._manager.get_server_queue().get()
      id = request['id']
      value = request['value']

      method = value[0]
      args = value[1:]

      try:
        value = self._endpoints.get(method)(*args)
      except Exception as _:
        self.P("Exception occured while processing\n"
               "Request: {}\nArgs: {}\nException:\n{}".format(
                   method, args, self.get_exception()), color='r')
        value = None

      response = {
        'id': id,
        'value': value
      }
      self._manager.get_client_queue().put(response)
    # end while

    return None

  def on_close(self):
    self._manager.shutdown()
    return

  def __get_uvicorn_process_args(self):
    return f"uvicorn --app-dir {self.script_temp_dir} main:app --host 0.0.0.0 --port {self.port}"

  @property
  def jinja_args(self):
    return {
      **self.cfg_jinja_args,
      'manager_port': self.manager_port,
      'manager_auth': self.manager_auth,
      'api_title': repr(self.cfg_api_title or self.get_signature()),
      'api_summary': repr(self.cfg_api_summary or f"FastAPI created by {self.get_signature()} plugin"),
      'api_description': repr(self.cfg_api_description or self.__doc__),
      'api_version': repr(self.__version__),
      **self._node_comms_jinja_args
    }

  def get_start_commands(self):
    super_start_commands = super(FastApiWebAppPlugin, self).get_start_commands()
    return super_start_commands + [self.__get_uvicorn_process_args()]
