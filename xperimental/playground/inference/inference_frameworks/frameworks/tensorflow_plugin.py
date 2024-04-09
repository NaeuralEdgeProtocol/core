import os
import tensorflow.compat.v1 as tf

from playground.inference import constants as ct
from playground.inference.inference_frameworks.inference_framework_exec import InferenceFrameworkExecutor

class TensorflowFrameworkPlugin(InferenceFrameworkExecutor):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def load(self, config):
    path_graph_tensors = self.log.get_model_file(config[ct.GRAPH_TENSORS])
    tensors = self.log.load_json(path_graph_tensors)
    input_names = [v for k,v in tensors.items() if 'INPUT_' in k]
    output_names = [v for k,v in tensors.items() if 'OUTPUT_' in k]
    
    assert input_names is not None and len(input_names) >= 1, 'You should provide the model input names'
    assert output_names is not None and len(output_names) >=1, 'You should provide the model output names'
    
    path_model = config[ct.GRAPH]
    if not os.path.isfile(path_model):
      path_model = self.log.get_model_file(path_model)
      
    graph = self.log.load_tf_graph(path_model)
    assert graph is not None, 'The graph could not be loaded from: {}'.format(path_model)

    sess = tf.Session(graph=graph)
    lst_inputs, lst_outputs = [], []
    for input_name in input_names:
      tf_inp = sess.graph.get_tensor_by_name(input_name)
      lst_inputs.append(tf_inp)
    for output_name in output_names:
      tf_out = sess.graph.get_tensor_by_name(output_name)
      lst_outputs.append(tf_out)
    model = {
      ct.GRAPH: graph,
      ct.SESSION: sess,
      ct.INPUTS: lst_inputs,
      ct.OUTPUTS: lst_outputs
      }
    return model
  
  def infer(self, model, data):
    self.log.p('Inferring')
    sess = model[ct.SESSION]
    inputs = model[ct.INPUTS]
    outputs = model[ct.OUTPUTS]
    if len(inputs) == 1:
      data = [data]
    else:
      assert len(inputs) == len(data), 'Model expects input data for all tensors'
    out = sess.run(
      fetches=outputs,
      feed_dict=dict(zip(inputs, data))
      )
    return out