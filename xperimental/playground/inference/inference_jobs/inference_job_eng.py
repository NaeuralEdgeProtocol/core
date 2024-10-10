from naeural_core import constants as ct

from naeural_core import DecentrAIObject
from playground.inference.inference_jobs.registered_jobs import _DEFINED_JOBS

__MODULE__ = ct.INFERENCE
__VER__ = '1.0.0.0'

class InferenceJobEngine(DecentrAIObject):
  def __init__(self, model_engine, **kwargs):
    self.__module__ = __MODULE__
    self.__version__ = __VER__
    self._model_engine = model_engine
    super().__init__(prefix_log='[INFERENCE_JOB_ENGINE]', **kwargs)
    return
  
  def startup(self):
    super().startup()
    self._jobs = {}
    self.P('Creating {} inference jobs...'.format(len(_DEFINED_JOBS)))
    for sign,v in _DEFINED_JOBS.items():
      job = v(
        model_engine=self._model_engine,
        log=self.log,
        DEBUG=self.DEBUG
        )
      self._jobs[sign] = job
    return
  
  def create(self, signature, data):
    if signature not in self._jobs:
      raise ValueError("`create` called with unknown signature `{}` that is not in {}".format(
          signature, 
          list(self._jobs.keys()))
        )
    job = self._jobs[signature]
    pipeline = job.create(input_data=data)
    return pipeline
      
      
    