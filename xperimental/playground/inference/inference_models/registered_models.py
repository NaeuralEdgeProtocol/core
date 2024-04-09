from playground.inference.inference_models.models.alimentare_nepermisa_plugin import AlimentareNepermisaPlugin
from playground.inference.inference_models.models.effdet0_plugin import EffDet0Plugin
from playground.inference.inference_models.models.face_detection_plugin import FaceDetectionPlugin
from playground.inference.inference_models.models.fire_smoke_plugin import FireSmokePlugin
from playground.inference.inference_models.models.employee_detection_plugin import EmployeeDetectionPlugin

_DEFINED_MODELS = {
  'ALIMENTARE_NEPERMISA': AlimentareNepermisaPlugin,
  'EFF_DET0'            : EffDet0Plugin,
  'EMPLOYEE_DETECTION'  : EmployeeDetectionPlugin,
  'FACE_DETECTION'      : FaceDetectionPlugin,
  'FIRE_SMOKE'          : FireSmokePlugin
}