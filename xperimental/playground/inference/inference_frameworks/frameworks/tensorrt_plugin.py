from playground.inference.inference_frameworks.inference_framework_exec import InferenceFrameworkExecutor

class TensorrtFrameworkPlugin(InferenceFrameworkExecutor):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
