from core.business.base import BasePluginExecutor

__VER__ = '1.0.0'

_CONFIG = {
  **BasePluginExecutor.CONFIG,

  'LOCAL_COMMS_ENABLED_ON_STARTUP'            : True,
  'ALLOW_EMPTY_INPUTS'                        : True,

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES'],
  },
}


class LocalComms01Plugin(BasePluginExecutor):
  CONFIG = _CONFIG

  def on_init(self):
    self.__comm_manager = self.global_shmem["comm_manager"]
    # state of the local comms
    self.__enabled = False

    # current request: True to enable, False to disable, None to do nothing
    self.__current_request = self.cfg_local_comms_enabled_on_startup
    return

  def on_command(self, data, enable=False, disable=False, **kwargs):
    if (isinstance(data, str) and data.upper() == 'ENABLE') or enable:
      self.__current_request = True
    elif (isinstance(data, str) and data.upper() == 'DISABLE') or disable:
      self.__current_request = False
    return

  def process(self):
    # if no request, do nothing
    if self.__current_request is None:
      return

    # if not enabled and request is to enable, enable
    if not self.__enabled and self.__current_request:
      self.__comm_manager.enable_local_communication()
      self.__enabled = True
    # if enabled and request is to disable, disable
    elif self.__enabled and not self.__current_request:
      self.__comm_manager.disable_local_communication()
      self.__enabled = False

    # reset request
    self.__current_request = None
    return