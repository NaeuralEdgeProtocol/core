from core.serving.default_inference.th_yf_base import YfBase as ParentServingProcess

LPD_CONFIG = {
  # "MODEL_WEIGHTS_FILENAME": "20231004_lpd_no_diff_nms.ths",
  # "MODEL_WEIGHTS_FILENAME_DEBUG": "20231004_lpd_no_diff_nms.ths",

  # "MODEL_WEIGHTS_FILENAME": "20231010_lpd_no_diff_896_40_nms.ths",
  # "MODEL_WEIGHTS_FILENAME_DEBUG": "20231010_lpd_no_diff_896_40_nms.ths",

  # "MODEL_WEIGHTS_FILENAME": "20231114_y8n_v1_only_896_10_nms.ths",
  # "MODEL_WEIGHTS_FILENAME_DEBUG": "20231114_y8n_v1_only_896_10_nms.ths",

  # "MODEL_WEIGHTS_FILENAME": "20231122_y8n_lpdr_v5_896_40_nms.ths",
  # "MODEL_WEIGHTS_FILENAME_DEBUG": "20231122_y8n_lpdr_v5_896_40_nms.ths",

  "MODEL_WEIGHTS_FILENAME": "20240221_y8n_lpd_v5_896_40_nms.ths",
  "MODEL_WEIGHTS_FILENAME_DEBUG": "20240221_y8n_lpd_v5_896_40_nms.ths",

  "URL": "minio:LPD/20240221_y8n_lpd_v5_896_40_nms.ths",
  "URL_DEBUG": "minio:LPD/20240221_y8n_lpd_v5_896_40_nms.ths",

  # 'IMAGE_HW': (448, 640),

  'IMAGE_HW': (640, 896)
}

_CONFIG = {
  **ParentServingProcess.CONFIG,

  **LPD_CONFIG,

  'VALIDATION_RULES': {
    **ParentServingProcess.CONFIG['VALIDATION_RULES'],
  },
}

DEBUG_COVERED_SERVERS = False

__VER__ = '0.2.0.0'


class ThLpd(ParentServingProcess):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(ThLpd, self).__init__(**kwargs)
    return
