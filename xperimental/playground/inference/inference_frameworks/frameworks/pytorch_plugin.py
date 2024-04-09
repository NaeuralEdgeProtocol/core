from playground.inference.inference_frameworks.inference_framework_exec import InferenceFrameworkExecutor

class PytorchFrameworkPlugin(InferenceFrameworkExecutor):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def load(self, config):
    self.P('Loading pytorch model')
    #load source code
    
    #instantiate nn.Module class
    
    #load weights
    
    #load to gpu
    return None
  
  def infer(self, model, data):
    self.P('Infering pytorch model')
    return
    
    