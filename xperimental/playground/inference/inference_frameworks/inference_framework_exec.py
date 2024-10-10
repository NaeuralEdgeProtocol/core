from naeural_core import DecentrAIObject

__version__ = '1.0.0.0'

class InferenceFrameworkExecutor(DecentrAIObject):
  def __init__(self, **kwargs):    
    super().__init__(prefix_log='[INFERENCE_FRAMEWORK_EXECUTOR]', **kwargs)
    return
  
  def load(self, config):
    raise NotImplementedError()
    return
  
  def infer(self, model, data):
    self.log.p('Infering {}'.format(self.__class__.__name__))
    #infer based on model and data
    res = None
    return res