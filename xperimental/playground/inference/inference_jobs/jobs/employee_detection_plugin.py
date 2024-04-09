from playground.inference.inference_jobs.inference_job_exec import InferenceJobExecutor

class EmployeeDetectionJobPlugin(InferenceJobExecutor):
  def __init__(self, model_engine, **kwargs):
    super().__init__(model_engine, **kwargs)
    return
  
  #TODO: rethink this approach because this inference step could be incorporated into "EFFDET0" job
  def _infer1(self, data):
    res = self._model_engine.infer(
      signature='EFF_DET0',
      data=data
      )
    return res
  
  def _infer2(self, data):
    res = self._model_engine.infer(
      signature='EMPLOYEE_DETECTION',
      data=data
      )
    return res
  
  def _extract_humans(self, data):
    #here you should extract persons...
    return data
  
  def _prepare_results(self, data):
    #here you should prepare results
    return data
  
  def create(self, input_data):
    preproc_data = self._preprocess_input_data(input_data)
    batch_data = self._batch_data(preproc_data)    
    
    # pipeline = [self._infer]
    
    pipeline = [
      {
        'STEP': 0,
        'TYPE': 'INFERENCE',
        'INPUT_DATA': batch_data,
        'EXECUTE': self._infer1
      },
      {
        'STEP': 1,
        'TYPE': 'CUSTOM',
        'INPUT_DATA': 'STEP_0',
        'EXECUTE': self._extract_humans
      },
      {
        'STEP': 2,
        'TYPE': 'INFERENCE',
        'INPUT_DATA': 'STEP_1',
        'EXECUTE': self._infer2
      },
      {
        'STEP': 3,
        'TYPE': 'CUSTOM',
        'INPUT_DATA': 'STEP_2',
        'EXECUTE': self._prepare_results
      }
      ]
    return pipeline