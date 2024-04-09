#where will the configuration come? from feature or job configuration

from playground.inference.inference_jobs.inference_job_exec import InferenceJobExecutor

class TimeseriesJobPlugin(InferenceJobExecutor):
  def __init__(self, model_engine, **kwargs):
    super().__init__(model_engine, **kwargs)
    self.config = {'BUFFER_SIZE': 100} #TODO: where this will come from??????????
    return
  
  def startup(self):
    super().startup()
    self._lst = []
    return
  
  def _infer(self, data):
    res = self._model_engine.infer(
      signature='DUMMY_ANALYZING', #DUMMY!!
      data=data
      )
    return res
  
  def create(self, input_data):
    self._lst.append(input_data)
    pipeline = []
    if self.config['BUFFER_SIZE'] == 100:
      pipeline = [
        {
          'STEP': 0,
          'TYPE': 'INFERENCE',
          'INPUT_DATA': self._lst,
          'EXECUTE': self._infer
        }
        ]
      self._lst = []
    return pipeline