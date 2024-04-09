from playground.inference.inference_jobs.inference_job_exec import InferenceJobExecutor

class FaceDetectionJobPlugin(InferenceJobExecutor):
  def __init__(self, model_engine, **kwargs):
    super().__init__(model_engine, **kwargs)
    return
  
  def _infer(self, data):
    res = self._model_engine.infer(
      signature='FACE_DETECTION',
      data=data
      )
    return res
  
  def create(self, input_data):
    preproc_data = self._preprocess_input_data(input_data)
    batch_data = self._batch_data(preproc_data)
    
    pipeline = [
      {
        'STEP': 0,
        'TYPE': 'INFERENCE',
        'INPUT_DATA': batch_data,
        'EXECUTE': self._infer
      }
      ]
    return pipeline