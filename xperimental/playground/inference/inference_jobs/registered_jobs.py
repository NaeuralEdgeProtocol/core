from playground.inference.inference_jobs.jobs.effdet0_plugin import EffDet0JobPlugin
from playground.inference.inference_jobs.jobs.facedetection_plugin import FaceDetectionJobPlugin
from playground.inference.inference_jobs.jobs.employee_detection_plugin import EmployeeDetectionJobPlugin
__VER__ = '1.0.0.0'

_DEFINED_JOBS = {
  'EFF_DET0'          : EffDet0JobPlugin,
  'FACE_DETECTION'    : FaceDetectionJobPlugin,
  'EMPLOYEE_DETECTION': EmployeeDetectionJobPlugin
}