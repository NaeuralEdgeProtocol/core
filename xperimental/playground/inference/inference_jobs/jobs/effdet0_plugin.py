from playground.inference.inference_jobs.inference_job_exec import InferenceJobExecutor


class EffDet0JobPlugin(InferenceJobExecutor):
  def __init__(self, model_engine, **kwargs):
    super().__init__(model_engine, **kwargs)
    return
  
  def startup(self):
    super().startup()
    self._model_config = {} #self._model_engine.get_config(signature='EFF_DET0')
    return
  
  def _infer(self, data):
    res = self._model_engine.infer(
      signature='EFF_DET0',
      data=data
      )
    return res
  
  def create(self, data, needs):        
    l = []
    dct = {}
    use_full_frame = self._model_config.get('INFERENCE_ON_FULL_IMAGE', True)
    for need in needs:
      if use_full_frame:
        dct['FULL_DATA'] = data
      else:
        top, left, bottom, right = need.location_coords
        patch = data[top:bottom, left:right]
        l.append(patch)
    
    pipeline = [
      {
        'STEP': 0,
        'TYPE': 'INFERENCE',
        'INPUT_DATA': [],
        'EXECUTE': self._infer
      }
      ]
    return pipeline





