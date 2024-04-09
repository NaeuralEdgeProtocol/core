from playground.inference.inference_frameworks.frameworks.keras_plugin import KerasFrameworkPlugin
from playground.inference.inference_frameworks.frameworks.pytorch_plugin import PytorchFrameworkPlugin
from playground.inference.inference_frameworks.frameworks.onnx_plugin import OnnxFrameworkPlugin
from playground.inference.inference_frameworks.frameworks.tensorflow_plugin import TensorflowFrameworkPlugin
from playground.inference.inference_frameworks.frameworks.tensorrt_plugin import TensorrtFrameworkPlugin

__VER__ = '1.0.0.0'

_DEFINED_FRAMEWORKS = {
  'KERAS'       : KerasFrameworkPlugin,
  'ONNX'        : OnnxFrameworkPlugin,
  'PYTORCH'     : PytorchFrameworkPlugin,
  'TENSORFLOW'  : TensorflowFrameworkPlugin,
  'TENSORRT'    : TensorrtFrameworkPlugin
}