from naeural_core import constants as ct

from naeural_core import DecentrAIObject
from playground.inference.inference_frameworks.registered_frameworks import _DEFINED_FRAMEWORKS

__MODULE__ = ct.INFERENCE
__VER__ = '1.0.0.0'

class InferenceFrameworkEngine(DecentrAIObject):
  def __init__(self, **kwargs):
    self.__module__ = __MODULE__
    self.__version__ = __VER__
    super().__init__(prefix_log='[INFERENCE_FRAMEWORK_ENGINE]', **kwargs)
    return
  
  def startup(self):
    super().startup()
    self._frameworks = {}
    self.P('Creating {} inference frameworks...'.format(len(_DEFINED_FRAMEWORKS)))
    for sign,v in _DEFINED_FRAMEWORKS.items():
      frmk = v(
        log=self.log,
        DEBUG=self.DEBUG
        )
      self._frameworks[sign] = frmk
    return
  
  def load(self, signature, model_config):
    if signature not in self._frameworks:
      raise ValueError("`load` called with unknown signature `{}` that is not in {}".format(
          signature, 
          list(self._frameworks.keys()))
        )
    frmk = self._frameworks[signature]
    model = frmk.load(config=model_config)
    return model
  
  def infer(self, signature, model, data):
    if signature not in self._frameworks:
      raise ValueError("`infer` called with unknown signature `{}` that is not in {}".format(
          signature, 
          list(self._frameworks.keys()))
        )
    frmk = self._frameworks[signature]
    res = frmk.infer(model=model, data=data)
    return res
      
      
    