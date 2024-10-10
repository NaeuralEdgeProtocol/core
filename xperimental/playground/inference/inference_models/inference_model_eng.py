from naeural_core import constants as ct

from naeural_core import DecentrAIObject
from playground.inference.inference_models.registered_models import _DEFINED_MODELS

__MODULE__ = ct.INFERENCE
__VER__ = '1.0.0.0'

class InferenceModelEngine(DecentrAIObject):
  def __init__(self, inference_framework_engine, **kwargs):
    self.__version__ = __VER__
    self.__module__ = __MODULE__
    self._framework_engine = inference_framework_engine
    super().__init__(prefix_log='[INFERENCE_MODEL_ENGINE]', **kwargs)
    return

  def startup(self):
    super().startup()
    self._inference_funcs = {}
    return

  def start_model(self, signature, config):
    if signature not in _DEFINED_MODELS:
      raise ValueError('Inference Plugin `{}` not found in existing defined plugins: {}'.format(signature, list(_DEFINED_MODELS.keys())))
    if signature in self._inference_funcs and self.DEBUG:
      self.log.p('Model `{}` already started')
      return
    
    model = _DEFINED_MODELS[signature](
      config=config,
      inference_framework_engine=self._framework_engine,
      log=self.log,
      DEBUG=self.DEBUG
      )
    self._inference_funcs[signature] = model.infer
    return

  def infer(self, signature, data):
    if signature not in self._inference_funcs:
      raise ValueError("`infer` called with unknown signature `{}` that is not in {}".format(
          signature, 
          list(self._inference_funcs.keys()))
        )
    func = self._inference_funcs[signature]
    res = func(data=data)
    return res
  