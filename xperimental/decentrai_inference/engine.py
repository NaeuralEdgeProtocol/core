import numpy as np
from decentrai_inference import constants as ct

from enum import Enum
from naeural_core import DecentrAIObject
from naeural_core import Logger
from decentrai_inference.graphs import InferenceGraphsEnum

__version__ = '1.0.1.0'

class InferenceEngine(DecentrAIObject):
  """
  This class purpose is provide a single entry point to all the inference graphs.
  Please use this class in order to make inference with any available InferenceGraph.
  """
  
  __instance = None
  
  def __init__(self, config, **kwargs):
    if InferenceEngine.__instance != None:
      raise Exception('This class is singleton. Please use get_instance static method')
    else:
      self.__version__ = __version__
      self.config = config
      super().__init__(**kwargs)
      InferenceEngine.__instance = self
    return
  
  def startup(self):
    self.inference_graphs = {}
    self._init_graphs()
    super().startup()
  
  @staticmethod
  def get_instance(config, log):
    if InferenceEngine.__instance is None:
      InferenceEngine(config=config, log=log)
    return InferenceEngine.__instance
  
  def _init_graphs(self):
    self.config_inference = self.config[ct.CONFIG_INFERENCE]
    lst_use_graphs = [k for k,v in self.config_inference['USE_GRAPHS'].items() if v]

    msg = "Initializing following graphs: {}".format(lst_use_graphs)
    self.P(msg, color='y')

    for graph in lst_use_graphs:
      self._start_graph(graph)
    return
  
  def _start_graph(self, graph):
    config_eng = self.config_inference[graph]
    self.log.p('Initializing {}'.format(graph))
    self.inference_graphs[graph] = InferenceGraphsEnum[graph].value(
      config_graph=config_eng, 
      log=self.log
      )
    return
  
  def predict(self, graph, data, predict_batch=False, **kwargs):
    """
    This function calls `predict` or `predict_batch` method of a specific graph

    Parameters
    ----------
    graph : string or InferenceGraphsEnum
      Name of the graph that should be used for inference.
    data : dict, list or np.ndarray
      Images used for inference.
    predict_batch: bool
      Used to specify predict method: standard `predict` or `predict_batch`
    **kwargs : TYPE
      Specific graph .predict arguments.

    Raises
    ------
    ValueError
      If data is not one of (dict, list, np.ndarray) an error will be triggered.

    Returns
    -------
    preds : dict
      Dictionary containing the inference results for provided data.

    """
    assert isinstance(graph, (Enum, str)), 'Graph should be either \
      InferenceGraphsEnum or string. You provided {}'.format(graph)
    if isinstance(graph, Enum):
      graph = graph.name
    assert graph in self.inference_graphs, 'Graph {} not loaded. \
      Available graphs: {}'.format(graph, ', '.join(self.inference_graphs.keys()))
    
    lst_img = []
    if isinstance(data, dict):
      lst_img = list(data.values())
    elif isinstance(data, (list, np.ndarray)):
      lst_img = data
    else:
      raise ValueError('Expected dict, list or np.ndarray for inference data.')
    #endif
    if predict_batch:
      preds = self.inference_graphs[graph].predict_batch(lst_img, **kwargs)
    else:
      preds = self.inference_graphs[graph].predict(lst_img, **kwargs)
    #endif
    
    if isinstance(data, dict):
      preds = preds['INFERENCES']
      preds = dict(zip(data.keys(), preds))
    #endif
    return preds
  
  def deallocate_graphs(self, lst_names):
    self.P('Deallocating graphs: {}'.format(lst_names))
    for name in lst_names:
      del self.inference_graphs[name]
      self.config_inference['USE_GRAPHS'][name] = False
    return
      
  def refresh_graphs(self, lst_names):
    lst_use_graphs = [k for k,v in self.config_inference['USE_GRAPHS'].items() if v]
    lst_new = list(set(lst_names) - set(lst_use_graphs))
    lst_deallocate = list(set(lst_use_graphs) - set(lst_names))
    
    if lst_new:
      self.P('Found {} new graphs to start: {}'.format(len(lst_new), lst_new))
    for name in lst_new:
      self._start_graph(name)
      self.config_inference['USE_GRAPHS'][name] = True
    #endfor
      
    if lst_deallocate:
      self.P('Found {} graphs to deallocate: {}'.format(len(lst_deallocate), lst_deallocate))
      self.deallocate_graphs(lst_deallocate)
    return
  
if __name__ == '__main__':
  from api_engine.config import Config
  log = Logger(lib_name='TST', config_file='main_config.txt')
  config = Config(log=log)
  dct_images = {k:np.random.randint(0, 255, size=(1080, 1920, 3), dtype=np.uint8)
                for k in config['STREAMS_CONFIG'].keys()}
  
  inf_eng = InferenceEngine(log=log, config=config)
  preds = inf_eng.predict(
    graph=InferenceGraphsEnum.COVID_MASK, 
    data=dct_images,
    predict_batch=True
    )
  
