from core.business.base import BasePluginExecutor
from core.business.mixins_libs.ngrok_mixin import _NgrokMixinPlugin

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePluginExecutor.CONFIG,
  'ALLOW_EMPTY_INPUTS': True,
  'RUN_WITHOUT_IMAGE': True,
  'PROCESS_DELAY': 0.01,

  'USE_NGROK': False,
  'NGROK_DOMAIN': None,
  'NGROK_EDGE_LABEL': None,

  'PORT': None,

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES']
  },
}


class BaseWebAppPlugin(_NgrokMixinPlugin, BasePluginExecutor):
  """
  A base plugin which will handle the lifecycle of a web application.
  Through this plugin, you can expose your business logic as a web application,
  using some implementation of a web server.

  You can also deploy your web application to the internet using ngrok.
  To do this, set the `USE_NGROK` flag to True in config and set the necessary
  environment variables.

  TODO: add ngrok necessary data in the config (after securing the configs)
  """

  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(BaseWebAppPlugin, self).__init__(**kwargs)
    return

  def __check_port_valid(self):
    # Check the config as we're going to use it to start processes.
    if not isinstance(self.cfg_port, int):
      raise ValueError("Port not an int")
    if self.cfg_port < 0 or self.cfg_port > 65535:
      raise ValueError("Invalid port value {}".format(self.cfg_port))
    return

  @property
  def port(self):
    if 'USED_PORTS' not in self.plugins_shmem:
      return None
    if self.str_unique_identification not in self.plugins_shmem['USED_PORTS']:
      return None
    port = self.plugins_shmem['USED_PORTS'][self.str_unique_identification]
    return port

  def on_init(self):
    self.lock_resource('USED_PORTS')
    if 'USED_PORTS' not in self.plugins_shmem:
      self.plugins_shmem['USED_PORTS'] = {}
    dct_shmem_ports = self.plugins_shmem['USED_PORTS']
    used_ports = dct_shmem_ports.values()

    if self.cfg_port is not None:
      self.__check_port_valid()

      if self.cfg_port in used_ports:
        raise Exception("Port {} is already in use.".format(self.cfg_port))
      else:
        dct_shmem_ports[self.str_unique_identification] = self.cfg_port
    else:
      port = self.np.random.randint(49152, 65535)
      while port in used_ports:
        port = self.np.random.randint(49152, 65535)
      # endwhile
      dct_shmem_ports[self.str_unique_identification] = port
    # endif port
    self.unlock_resource('USED_PORTS')

    super(BaseWebAppPlugin, self).on_init()
    return

  def on_close(self):
    self.lock_resource('USED_PORTS')
    if 'USED_PORTS' in self.plugins_shmem:
      self.plugins_shmem['USED_PORTS'].pop(self.str_unique_identification, None)
    self.unlock_resource('USED_PORTS')

    super(BaseWebAppPlugin, self).on_close()
    return
