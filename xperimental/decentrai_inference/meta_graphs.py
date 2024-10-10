#global dependencies
import copy
import itertools
import numpy as np

from enum import Enum
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict

#local dependencies
from naeural_core import constants as ct

from naeural_core import DecentrAIObject
from decentrai_inference.engine import InferenceEngine
from decentrai_inference.graphs import InferenceGraphsEnum

__version__ = '1.1.7.0'

#TODO: how can we bring stream/location custom configs into inference handler
  
class InferenceMetaGraph(DecentrAIObject):
  def __init__(self, config, **kwargs):
    self.config = config
    self.__version__ = __version__
    super().__init__(**kwargs)
    return
  
  def _filter_streams(self, dct_inference, graph_name, feat_name, exact_match=True):
    if exact_match:
      lst_streams = [x for x in dct_inference[graph_name] if feat_name in x[ct.PLUGINS]]
    else:
      lst_streams = [x for x in dct_inference[graph_name] if any(feat_name in plg for plg in x[ct.PLUGINS])]
    lst_inf = [x[ct.INFERENCE] for x in lst_streams]
    np_imgs = [x[ct.IMAGE] for x in lst_streams]
    
    lst_streams_filtered = deepcopy(lst_streams)
    lst_inf_filtered = deepcopy(lst_inf)
    np_imgs_filtered = deepcopy(np_imgs)
    return lst_streams_filtered, lst_inf_filtered, np_imgs_filtered
  
  def startup(self):
    self.inf_engine = InferenceEngine.get_instance(config=self.config, log=self.log)
    super().startup()
    return
  
  def predict_batch(self, np_imgs=None, dct_inference=None):
    raise NotImplementedError()
    return
  
  def used_graphs(self):
    raise NotImplementedError()
    return
  

class CarTrackingInferenceMetaGraph(InferenceMetaGraph):
  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    return
  
  def used_graphs(self):
    l = [
      InferenceGraphsEnum.LP_DETECTION.name,
      InferenceGraphsEnum.LPR.name
      ]
    return l
  
  def predict_batch(self, np_imgs=None, dct_inference=None):
    """
    This method handles batch inference for car tracking plugin
    
    Parameters
    ----------
    np_imgs : List/Array of images
      List containing dictionaries with the following format:
        "STREAM"        : stream name
        "LOCATION_NAME" : location configured on a specific stream
        "IMAGE"         : np array containing location/entire stream image

    Returns
    -------
    dct_inf : PREDS
      Dictionary containing the predictions. Dictionary structure:
        {
        "STREAM": {
          "LOCATION_NAME": [
            {INF1}, {INF2}, {INF3}
          ]
        }
    """
    timer_name = self.timer_name(name='predict_batch')
    self.log.start_timer(timer_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    is_streams_infer = False
    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    #inf lp detection engine
    graph_name = InferenceGraphsEnum.LP_DETECTION.name
    feat_name = ct.CAR_TRACKING_01
    if np_imgs is not None:
      lst_inf = self.inf_engine.predict(
        graph=graph_name, 
        data=np_imgs, 
        predict_batch=True
        )['INFERENCES']
    else:
      is_streams_infer = True
      lst_streams, lst_inf, np_imgs = self._filter_streams(dct_inference, graph_name, feat_name)
    #endif
    
    #extract license plate images from each stream image
    lst_imgs = []
    lst_streams_inf = []
    max_inf = 1
    lst_object_types = self.config[ct.CONFIG_PLUGINS][feat_name]['OBJECT_TYPE']
    for idx, lst in enumerate(lst_inf):
      lst = lst[:max_inf]
      lst = [inf for inf in lst if inf['TYPE'] in lst_object_types]
      lst_img = [np_imgs[idx][x['TLBR_POS'][0]:x['TLBR_POS'][2], 
                              x['TLBR_POS'][1]:x['TLBR_POS'][3], :] for x in lst]
      lst_imgs.append(lst_img)
      lst_streams_inf.append(lst)
    #endfor
    lst_imgs = list(itertools.chain.from_iterable(lst_imgs))
      
    #infer on lpr model
    dct_inf = self.inf_engine.predict(
      graph=InferenceGraphsEnum.LPR.name, 
      data=lst_imgs, 
      predict_batch=True
      )
    np_lpr_proba = dct_inf['INFERENCES']

    #from chained list back to the initial distribution
    crt_idx = 0
    lst_res = []
    for lst in lst_streams_inf:
      ln = len(lst)
      lst_res.append(np_lpr_proba[crt_idx: crt_idx + ln])
      crt_idx+= ln
    #endfor
    
    assert [len(x) for x in lst_streams_inf] == [len(x) for x in lst_res]
    
    for lst_stream_inf, lst_stream_prob in zip(lst_streams_inf, lst_res):
      for j in range(len(lst_stream_inf)):
        lst_stream_inf[j]['LPR_PROBA'] = lst_stream_prob[j]
    #endfor
    
    res = None
    if is_streams_infer:
      for dct_stream, l_inf in zip(lst_streams, lst_streams_inf):
        dct_stream['INFERENCE'] = l_inf
      res = lst_streams
    else:
      #prepare results
      dct_meta['SYSTEM_TIME'] = timestamp
      dct_meta['VER'] = self.__version__
      dct_result['METADATA'] = dct_meta
      dct_result['INFERENCES'] = lst_streams_inf
      res = dct_result
    #endif
    self.log.stop_timer(timer_name)
    return res
  

class BlurLicensePlateInferenceMetaGraph(InferenceMetaGraph):
  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    return
  
  def predict_batch(self, np_imgs=None, dct_inference=None):
    """
    This method handles batch inference for car tracking plugin
    
    Parameters
    ----------
    np_imgs : List/Array of images
      List containing dictionaries with the following format:
        "STREAM"        : stream name
        "LOCATION_NAME" : location configured on a specific stream
        "IMAGE"         : np array containing location/entire stream image

    Returns
    -------
    dct_inf : PREDS
      Dictionary containing the predictions. Dictionary structure:
        {
        "STREAM": {
          "LOCATION_NAME": [
            {INF1}, {INF2}, {INF3}
          ]
        }
    """
    timer_name = self.timer_name(name='predict_batch')
    self.log.start_timer(timer_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    is_streams_infer = False
    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    #inf lp detection engine
    feat_name = ct.BLUR_LICENSE_PLATE_02
    cfg = self.config[ct.CONFIG_PLUGINS][feat_name]
    
    #inf default engine
    graph_name = cfg['INFERENCE_GRAPHS'][0]
    if np_imgs is not None:
      lst_inf = self.inf_engine.predict(
        graph=graph_name, 
        data=np_imgs, 
        predict_batch=True
        )['INFERENCES']
    else:
      is_streams_infer = True
      #avoid modifying the original inference dictionary
      dct_crt_inference = copy.deepcopy(dct_inference)
      lst_streams, lst_inf, np_imgs = self._filter_streams(dct_crt_inference, graph_name, feat_name)
    #endif
    
    #extract car images from each stream image
    lst_imgs = []
    lst_streams_inf = []
    lst_cars_tlbr = []    
    lst_object_types = self.config[ct.CONFIG_PLUGINS][feat_name]['OBJECT_USED_FOR_LP_DETECTION']    
    for idx, lst in enumerate(lst_inf):
      # lst = lst[:max_inf]
      #filter only accepted objects that will be used in LP Detection
      lst = [inf for inf in lst if inf['TYPE'] in lst_object_types]
      lst_tlbr = [x['TLBR_POS'] for x in lst]
      #extract images with objects
      lst_img = [np_imgs[idx][x[0]:x[2], x[1]:x[3], :] for x in lst_tlbr]
      lst_imgs.append(lst_img)
      lst_cars_tlbr.append(lst_tlbr)
      lst_streams_inf.append(lst)
    #endfor
    lst_imgs = list(itertools.chain.from_iterable(lst_imgs))
      
    #infer on lpr model
    dct_inf = self.inf_engine.predict(
      graph=InferenceGraphsEnum.LP_DETECTION.name,
      data=lst_imgs,
      predict_batch=True
      )
    lst_lp = dct_inf['INFERENCES']

    #from chained list back to the initial distribution of inferences per stream
    crt_idx = 0
    lst_res = []
    for lst in lst_streams_inf:
      ln = len(lst)
      lst_res.append(lst_lp[crt_idx: crt_idx + ln])
      crt_idx+= ln
    #endfor
    
    assert [len(x) for x in lst_streams_inf] == [len(x) for x in lst_res]
    
    #postprocess lp tlbrs by adding original object (ex: car) position to detected license plate
    max_lp = 1
    lst_streams_lp = []
    for lst_stream_lp, lst_stream_tlbr in zip(lst_res, lst_cars_tlbr):
      lst = []
      for lst_lp, tlbr_car in zip(lst_stream_lp, lst_stream_tlbr):
        top_car, left_car, bottom_car, right_car = tlbr_car
        lst_lp = lst_lp[:max_lp]
        for dct_lp in lst_lp:
          top, left, bottom, right = dct_lp['TLBR_POS']
          top+= top_car
          left+= left_car
          bottom+= top_car
          right+= left_car
          dct_lp['TLBR_POS'] = [top, left, bottom, right]
          lst.append(dct_lp)
      #endfor
      lst_streams_lp.append(lst)
    #endfor
    
    #send detected license plate to plugin executor
    lst_streams_inf = lst_streams_lp
    
    res = None
    if is_streams_infer:
      for dct_stream, l_inf in zip(lst_streams, lst_streams_inf):
        dct_stream['INFERENCE'] = l_inf
      res = lst_streams
    else:
      #prepare results
      dct_meta['SYSTEM_TIME'] = timestamp
      dct_meta['VER'] = self.__version__
      dct_result['METADATA'] = dct_meta
      dct_result['INFERENCES'] = lst_streams_inf
      res = dct_result
    #endif
    self.log.stop_timer(timer_name)
    return res

  
class OmvEmployeeDetectionInferenceMetaGraph(InferenceMetaGraph):
  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    return
  
  def used_graphs(self):
    feat_name = ct.EMPLOYEE_DETECTION_01
    cfg = self.config[ct.CONFIG_PLUGINS][feat_name]
    l = [
      cfg['INFERENCE_GRAPHS'][0],
      InferenceGraphsEnum.OMV_EMPLOYEE.name
      ]
    return l
  
  def predict_batch(self, np_imgs=None, dct_inference=None):
    """
    This method handles batch inferences for omv employee
    Parameters
    ----------
    np_imgs : List/Array of images

    Returns
    -------
    dct_result : PREDS
      Dictionary containing the predictions. Dictionary structure example:
        {
        "METADATA": {
          "SYSTEM_TIME": '20200618_090324',
          "VER": '1.0.0.0'
        },
        "INFERENCES": [{'TLBR_POS': [295, 542, 510, 622],
                       'PROB_PRC': 1.6298054106300697e-06,
                       'TYPE': 'employee'},
                      {'TLBR_POS': [372, 299, 526, 388],
                       'PROB_PRC': 0.10143492370843887,
                       'TYPE': 'employee'}]
    """
    timer_name = self.timer_name(name='predict_batch')
    self.log.start_timer(timer_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    is_streams_infer = False
    dct_result = OrderedDict()
    dct_meta = OrderedDict()    
    
    feat_name = ct.EMPLOYEE_DETECTION_01
    cfg = self.config[ct.CONFIG_PLUGINS][feat_name]
    
    #inf default engine
    graph_name = cfg['INFERENCE_GRAPHS'][0]
    if dct_inference is None or graph_name not in dct_inference:
      lst_inf = self.inf_engine.predict(
        graph=graph_name, 
        data=np_imgs, 
        predict_batch=True
        )['INFERENCES']
    else:
      is_streams_infer = True
      lst_streams, lst_inf, np_imgs = self._filter_streams(dct_inference, graph_name, feat_name)
    #endif

    #extract license plate images from each stream image
    lst_imgs = []
    lst_streams_inf = []
    lst_object_types = self.config[ct.CONFIG_PLUGINS][feat_name]['OBJECT_TYPE']
    for idx, lst in enumerate(lst_inf):
      lst = [inf for inf in lst if inf['TYPE'] in lst_object_types]
      lst_img = [np_imgs[idx][x['TLBR_POS'][0]:x['TLBR_POS'][2], 
                              x['TLBR_POS'][1]:x['TLBR_POS'][3], :] for x in lst]
      lst_imgs.append(lst_img)
      lst_streams_inf.append(lst)
    #endfor
    lst_imgs = list(itertools.chain.from_iterable(lst_imgs))
    
    #infer on omv employee graph
    dct_inf = self.inf_engine.predict(
      graph=InferenceGraphsEnum.OMV_EMPLOYEE.name, 
      data=lst_imgs,
      predict_batch=True)
    lst_proba = []
    if isinstance(dct_inf['INFERENCES'], np.ndarray):
      lst_proba = dct_inf['INFERENCES'].ravel().tolist()
    
    #from chained list back to the initial distribution
    crt_idx = 0
    lst_res = []
    for lst in lst_streams_inf:
      ln = len(lst)
      lst_res.append(lst_proba[crt_idx: crt_idx + ln])
      crt_idx+= ln
    #endfor
    
    assert [len(x) for x in lst_streams_inf] == [len(x) for x in lst_res]
    
    for lst_stream_inf, lst_stream_prob in zip(lst_streams_inf, lst_res):
      for j in range(len(lst_stream_inf)):
        lst_stream_inf[j]['TYPE'] = 'employee'
        lst_stream_inf[j]['PROB_PRC'] = lst_stream_prob[j]
    #endfor
    
    res = None
    if is_streams_infer:
      for dct_stream, l_inf in zip(lst_streams, lst_streams_inf):
        dct_stream['INFERENCE'] = l_inf
      res = lst_streams
    else:
      #prepare results
      dct_meta['SYSTEM_TIME'] = timestamp
      dct_meta['VER'] = self.__version__
      dct_result['METADATA'] = dct_meta
      dct_result['INFERENCES'] = lst_streams_inf
      res = dct_result
    #endif
    self.log.stop_timer(timer_name)
    return res
  

class CovidInferenceMetaGraph(InferenceMetaGraph):
  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)
    return
  
  def used_graphs(self):
    l = [
      InferenceGraphsEnum.FACE_DETECTION.name,
      InferenceGraphsEnum.COVID_MASK.name
      ]
    return l
  
  def predict_batch(self, np_imgs=None, lst_streams_payload=None, dct_inference=None):
    """
    This method handles batch inference for covid plugin
    
    Parameters
    ----------
    np_imgs : List/Array of images
      List containing dictionaries with the following format:
        "STREAM"        : stream name
        "LOCATION_NAME" : location configured on a specific stream
        "IMAGE"         : np array containing location/entire stream image

    Returns
    -------
    dct_inf : PREDS
      Dictionary containing the predictions. Dictionary structure:
        {
        "STREAM": {
          "LOCATION_NAME": [
            {INF1}, {INF2}, {INF3}
          ]
        }
    """
    plugin_name = ct.COVID_MASK_01
    cfg = self.config[ct.CONFIG_PLUGINS][plugin_name]
    increase_pixels = cfg['INCREASE_FACE_PIXELS']
    thr_mask = cfg['THR_MASK']
    timer_name = self.timer_name(name='predict_batch')
    self.log.start_timer(timer_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    #inf lp detection engine
    graph_name = InferenceGraphsEnum.FACE_DETECTION.name
    if dct_inference is None or graph_name not in dct_inference:
      lst_inf = self.inf_engine.predict(
        graph=graph_name, 
        data=np_imgs,
        predict_batch=True
        )['INFERENCES']
      lst_inf = [[lst[0]] if lst else []  for lst in lst_inf]
    else:
      is_streams_infer = True
      lst_streams, lst_inf, np_imgs = self._filter_streams(
        dct_inference, 
        graph_name, 
        'COVID_MASK',
        exact_match=False
        )
    #endif
    
    #prepare person images for covid model
    for idx, lst in enumerate(lst_inf):
      for dct_face in lst:
        top, left, bottom, right = [int(x) for x in dct_face['TLBR_POS']]
        top    -= increase_pixels
        left   -= increase_pixels
        bottom += increase_pixels
        right  += increase_pixels
        top    = max(0, top)
        left   = max(0, left)
        bottom = min(bottom, np_imgs[idx].shape[0])
        right  = min(right, np_imgs[idx].shape[1])
        dct_face['TLBR_POS'] = [top, left, bottom, right]
    #endfor
        
    lst_imgs = [[np_imgs[idx][x['TLBR_POS'][0]:x['TLBR_POS'][2], 
                              x['TLBR_POS'][1]: x['TLBR_POS'][3], :] for x in lst] 
                for idx, lst in enumerate(lst_inf)]
    lst_imgs = list(itertools.chain.from_iterable(lst_imgs))
      
    #inf on covid model
    dct_inf = self.inf_engine.predict(
      graph=InferenceGraphsEnum.COVID_MASK.name, 
      data=lst_imgs,
      predict_batch=True)
    lst_proba = dct_inf['INFERENCES']
    if isinstance(lst_proba, np.ndarray):
      lst_proba = lst_proba.ravel().tolist()
    
    #from chained list back to the initial distribution
    crt_idx = 0
    lst_res = []
    for lst in lst_inf:
      ln = len(lst)
      lst_res.append(lst_proba[crt_idx: crt_idx + ln])
      crt_idx+= ln
    #endfor
      
    assert [len(x) for x in lst_inf] == [len(x) for x in lst_res], self.log.p(lst_inf, lst_res)
    
    for i in range(len(lst_inf)):
      lst_inf_crt = lst_inf[i]
      lst_prob_crt = lst_res[i]
      for j in range(len(lst_inf_crt)):
        lst_inf_crt[j]['TYPE'] = 'covid'
        lst_inf_crt[j]['PROB_PRC'] = (1 - lst_prob_crt[j])
    #endfor
      
    res = None
    if is_streams_infer:
      for dct_stream, l_inf in zip(lst_streams, lst_inf):
        dct_stream['INFERENCE'] = l_inf
      res = lst_streams
    else:
      #prepare results
      dct_meta['SYSTEM_TIME'] = timestamp
      dct_meta['VER'] = self.__version__
      dct_result['METADATA'] = dct_meta
      dct_result['INFERENCES'] = lst_inf
      res = dct_result
    #endif
    self.log.stop_timer(timer_name)
    return res


class InferenceHandlersEnum(Enum):
  CAR_TRACKING_META_GRAPH            = CarTrackingInferenceMetaGraph
  OMV_EMPLOYEE_DETECTION_META_GRAPH  = OmvEmployeeDetectionInferenceMetaGraph
  COVID_META_GRAPH                   = CovidInferenceMetaGraph
  BLUR_LICENSE_PLATE_META_GRAPH      = BlurLicensePlateInferenceMetaGraph

  
  
  
  
  
  
  