from core.business.base import BasePluginExecutor as BaseClass

_CONFIG = {
  **BaseClass.CONFIG,

  'PROCESS_DELAY': 5,

  'VALIDATION_RULES': {
    **BaseClass.CONFIG['VALIDATION_RULES'],
  },
}


class Ai4eLabelDataPlugin(BaseClass):
  _CONFIG = _CONFIG

  def on_init(self):
    super(Ai4eLabelDataPlugin, self).on_init()
    return

  def _process(self):
    self.add_payload_by_fields(
      dummy_data='dummy_data'
    )

    return

