from playground.inference.inference_models.inference_model_exec import InferenceModelExecutor

__VER__ = '1.0.0.0'

class FireSmokePlugin(InferenceModelExecutor):
  def __init__(self, config, **kwargs):
    self.__version__ = __VER__
    super().__init__(config=config, **kwargs)
    return
  
  