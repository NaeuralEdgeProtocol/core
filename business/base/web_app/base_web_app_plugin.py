from core.business.base import BasePluginExecutor
from core.business.mixins_libs.ngrok_mixin import _NgrokMixinPlugin

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePluginExecutor.CONFIG,
  'ALLOW_EMPTY_INPUTS': True,
  'RUN_WITHOUT_IMAGE': True,
  'PROCESS_DELAY': 0.01,

  'USE_NGROK' : False,
  'NGROK_DOMAIN' : None,
  'NGROK_EDGE_LABEL' : None,

  'PORT' : 8080,

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

  def on_init(self):
    super(BaseWebAppPlugin, self).on_init()
    return

  def on_close(self):
    super(BaseWebAppPlugin, self).on_close()
    return
