from collections import OrderedDict
from playground.inference import constants as ct
from naeural_core import DecentrAIObject

__VER__ = '1.0.0.0'

class InferenceModelExecutor(DecentrAIObject):
  def __init__(self, config, inference_framework_engine, **kwargs):
    self.config = config
    self._framework_engine = inference_framework_engine
    self._framework = self.config[ct.FRAMEWORK]
    super().__init__(prefix_log='[INFERENCE_MODEL_EXECUTOR]', **kwargs)
    return
  
  def startup(self):
    self.model = None
    super().startup()    
    self._load()
    return
  
  def _load(self):
    self.model = self._framework_engine.load(
      signature=self._framework,
      model_config=self.config
      )
    return
  
  def _preprocess_input_data(self, data):
    return data
  
  def _infer(self, data):
    res = self._framework_engine.infer(
      signature=self._framework,
      model=self.model,
      data=data
      )
    return res
  
  def _postprocess_inference(self, data):
    return data
  
  def _pack_results(self, results):
    results = OrderedDict()
    meta = OrderedDict()    
    meta['SYSTEM_TIME'] = self.log.now_str()
    meta['VER'] = self.__version__
    results['METADATA'] = meta
    results['INFERENCES'] = results
    return results
  
  def infer(self, data):
    preproc_data = self._preprocess_input_data(data)
    infer_res = self._infer(preproc_data)
    postproc_data = self._postprocess_inference(infer_res)
    results = self._pack_results(postproc_data)
    return results