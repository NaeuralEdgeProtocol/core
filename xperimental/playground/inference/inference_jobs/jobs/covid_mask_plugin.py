from playground.inference.inference_jobs.inference_job_exec import InferenceJobExecutor

class CovidMaskJobPlugin(InferenceJobExecutor):
  def __init__(self, model_engine, **kwargs):
    super().__init__(model_engine, **kwargs)
    return
  
  def _infer1(self, data):
    res = self._model_engine.infer(
      signature='EFF_DET0',
      data=data
      )
    return res
  
  def _infer2(self, data):
    res = self._model_engine.infer(
      signature='COVID_MASK',
      data=data
      )
    return res
  
  def _extract_humans(self, data):
    #here you should extract humans
    return data
  
  def _extract_faces(self, data):
    #here you should extract faces    
    return data
  
  def _prepare_results(self, data):
    #here you should prepare results
    return data
  
  def create(self, input_data):
    preproc_data = self._preprocess_input_data(input_data)
    batch_data = self._batch_data(preproc_data)
    
    pipeline = [
      {
        'STEP': 0,
        'TYPE': 'INFERENCE',
        'INPUT_DATA': batch_data,
        'EXECUTE': self._infer_effdet0
      },
      {
        'STEP': 1,
        'TYPE': 'CUSTOM',
        'INPUT_DATA': 'STEP_0',
        'EXECUTE': self._filter_humans
      },
      {
        'STEP': 2,
        'TYPE': 'CUSTOM',
        'INPUT_DATA': 'STEP_1',
        'EXECUTE': self._filter_humans2
      },
      {
        'STEP': 3,
        'TYPE': 'INFERENCE',
        'INPUT_DATA': 'STEP_2',
        'EXECUTE': self._infer_covid_mask
      },
      {
        'STEP': 4,
        'TYPE': 'CUSTOM',
        'INPUT_DATA': 'STEP_3',
        'EXECUTE': self._prepare_results
      }
      ]
    return pipeline