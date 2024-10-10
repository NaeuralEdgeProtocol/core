from collections import OrderedDict
from naeural_core import DecentrAIObject

__VER__ = '1.0.0.0'

class InferenceJobExecutor(DecentrAIObject):
  def __init__(self, model_engine, **kwargs): 
    self.__version__ = __VER__
    self._model_engine = model_engine
    super().__init__(prefix_log='[INFERENCE_JOB_ENGINE]', **kwargs)
    return
  
  def startup(self):
    super().startup()
    return
  
  def _preprocess_input_data(self, data):
    return data
  
  def _batch_data(self, data):
    lst = [x['DATA'] for x in data]
    return lst
  
  # def _execute(self, input_data, batch_data):
  #   raise NotImplementedError()
  #   return
  
  def _pack_results(self, results):
    lst = ['SYSTEM_TIME', 'VER', 'METADATA', 'INFERENCES']
    if isinstance(results, dict) and not all([x in results for x in lst]):
      results = OrderedDict()
      meta = OrderedDict()    
      meta['SYSTEM_TIME'] = self.log.now_str()
      meta['VER'] = self.__version__
      results['METADATA'] = meta
      results['INFERENCES'] = results
    return results
  
    # def execute(self, input_data):
    #   preproc_data = self._preprocess_input_data(data=input_data)
    #   batch_data = self._batch_data(data=preproc_data)
    #   res = self._execute(input_data=input_data, batch_data=batch_data)
    #   res = self._pack_results(res)
    #   return res
    
