from naeural_core.serving.default_inference.th_yf8s import _CONFIG as Y8S_CONFIG
from naeural_core.serving.default_inference.th_yf_lp import ThYfLp as BaseServingProcess

_CONFIG = {
  **BaseServingProcess.CONFIG,
  **Y8S_CONFIG,

  'MAX_BATCH_SECOND_STAGE': 10,

  'COVERED_SERVERS': ['th_yf8s'],

  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
  },
}

DEBUG_COVERED_SERVERS = False


class ThYfsLp(BaseServingProcess):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(ThYfsLp, self).__init__(**kwargs)

    return
