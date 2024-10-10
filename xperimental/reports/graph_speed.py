import numpy as np
import pandas as pd

from naeural_core import Logger
from decentrai_inference.graphs import EffDet2640x1132BS4InferenceGraph, EffDet2768x1358BS4InferenceGraph, EffDet3768x1358BS4InferenceGraph, EffDet5BS4InferenceGraph, EffDet7BS4InferenceGraph,\
  EffDet2640x1132BS1InferenceGraph, EffDet2768x1358BS1InferenceGraph, EffDet3768x1358BS1InferenceGraph, EffDet5BS1InferenceGraph, EffDet7BS1InferenceGraph
    


if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='EE_BMW', 
    config_file=cfg_file, 
    max_lines=1000, 
    TF_KERAS=False
    )
  
  BS = 4
  
  log.p('Start timing graphs for BS={}'.format(BS))
  
  if BS == 1:
    l = [
      EffDet2640x1132BS1InferenceGraph, 
      EffDet2768x1358BS1InferenceGraph,
      EffDet3768x1358BS1InferenceGraph,
      EffDet5BS1InferenceGraph, 
      EffDet7BS1InferenceGraph
    ]
  
  elif BS == 4:
    l = [
      # EffDet2640x1132BS4InferenceGraph, 
      # EffDet2768x1358BS4InferenceGraph,
      # EffDet3768x1358BS4InferenceGraph,
      # EffDet5BS4InferenceGraph, 
      EffDet7BS4InferenceGraph
    ]
    
  dct_time = {'GRAPH': [], 'TIME': []}
  for g_class in l:
    graph = g_class(
      log=log,
      config_path='_local_cache/_data/config/config_inference.txt'
      )
    
    for i in range(10):
      log.p('Inference #{}'.format(i))      
      np_imgs = np.random.randint(low=0, high=256, size=(BS, 1080, 1920, 3), dtype=np.uint8)
      log.start_timer(graph.name)
      graph.predict(np_imgs)
      log.stop_timer(graph.name)
    
    dct_time['GRAPH'].append(graph.name)
    dct_time['TIME'].append(log.get_timer_mean(graph.name))
  #endfor
  
  df_time = pd.DataFrame(dct_time)
  log.p('Time obtained by graphs for BS={}'.format(BS))
  log.p('\n\n{}'.format(df_time))  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  