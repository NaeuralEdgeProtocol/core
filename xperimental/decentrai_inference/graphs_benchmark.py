import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from decentrai_inference import constants as ct
from naeural_core import DecentrAIObject
from naeural_core import Logger
from naeural_core.local_libraries.nn.tf import utils as TFUtils

from decentrai_inference.graphs import EffDetInferenceGraph, TFOdapiInferenceGraph, \
  TFOdapi1InferenceGraph, TFOdapi2InferenceGraph

__version__ = '1.0.0.0'
  

class EffDet2InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET2')
    super().__init__(**kwargs)
    return
  
  
#EFF_DET2_640x1132
class EFFDET2_640x1132_BS1(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_640x1132_BS1')
    super().__init__(**kwargs)
    return


class EFFDET2_640x1132_BS4(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_640x1132_BS4')
    super().__init__(**kwargs)
    return
  
    
class EFFDET2_640x1132_BS14(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_640x1132_BS14')
    super().__init__(**kwargs)
    return


#EFF_DET2_768x1358
class EFFDET2_768x1358_BS1(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_768x1358_BS1')
    super().__init__(**kwargs)
    return


class EFFDET2_768x1358_BS4(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_768x1358_BS4')
    super().__init__(**kwargs)
    return


class EFFDET2_768x1358_BS5(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_768x1358_BS5')
    super().__init__(**kwargs)
    return


class EFFDET2_768x1358_BS7(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET2_768x1358_BS7')
    super().__init__(**kwargs)
    return


#EFF_DET3_768x1358
class EFFDET3_768x1358_BS1(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET3_768x1358_BS1')
    super().__init__(**kwargs)
    return


class EFFDET3_768x1358_BS4(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET3_768x1358_BS4')
    super().__init__(**kwargs)
    return
  
  
class EFFDET3_768x1358_BS5(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET3_768x1358_BS5')
    super().__init__(**kwargs)
    return

  
class EFFDET3_768x1358_BS7(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFFDET3_768x1358_BS7')
    super().__init__(**kwargs)
    return


#EFF_DET4
class EffDet4InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET4')
    super().__init__(**kwargs)
    return


class EffDet5BS1InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET5_BS1')
    super().__init__(**kwargs)
    return
  

class EffDet5BS4InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET5_BS4')
    super().__init__(**kwargs)
    return


class EffDet6InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET6')
    super().__init__(**kwargs)
    return


class EffDet7BS1InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET7_BS1')
    super().__init__(**kwargs)
    return


class EffDet7BS4InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET7_BS4')
    super().__init__(**kwargs)
    return


class TFOdapi1SSDMobilenetv2InferenceGraph(TFOdapi1InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI1_SSD_MOBILENETV2')
    super().__init__(**kwargs)
    return


class TFOdapi1FcnNasInferenceGraph(TFOdapi1InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI_FRC_NAS')
    super().__init__(**kwargs)
    return
    

class TFOdapi1OIDv4InferenceGraph(TFOdapi1InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI1_OIDV4')
    super().__init__(**kwargs)
    return
  
  def _load_classes(self):
    cls_file = self.config_graph['CLASSES']
    cls_desc = self.config_graph['CLASSES_DESCRIPTION']
    path_cbbox = self.log.get_model_file(cls_file)
    path_cdesc = self.log.get_model_file(cls_desc)
    
    df_cbbox = pd.read_csv(path_cbbox, header=None, names=['ID'])
    df_cdesc = pd.read_csv(path_cdesc, header=None, names=['ID', 'DESC'])
    df = pd.merge(
      left=df_cbbox,
      right=df_cdesc,
      on='ID', 
      how='inner'
      )
    self.orig_classes = df['DESC'].tolist()
    
    #because the model was with classes starting from 1
    self.orig_classes = ['None'] + self.orig_classes
    
    self.classes = self.orig_classes.copy()
    dct_rename = self.config_graph['RENAME_CLASS']
    for k,v in dct_rename.items():
      idx = self.classes.index(k)
      self.classes[idx] = v
    #endfor
    
    self.log.p('Setting probabilities')
    model_thr = self.config_graph['MODEL_THRESHOLD']
    dct_class_thr = self.config_graph['CLASS_THRESHOLD']
    self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr for c in self.classes}
    self.log.p('Done setting probabilities')
    return


class TFOdapi1TrafficSignsInferenceGraph(TFOdapi1InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI1_TRAFFICSIGNS')
    super().__init__(**kwargs)
    return
  
  def _load_classes(self):
    cls_file = self.config_graph['CLASSES']
    path_cls = self.log.get_model_file(cls_file)
    
    df_cls = pd.read_csv(path_cls, header=None, names=['CLASS'])
    self.orig_classes = df_cls['CLASS'].tolist()
    
    #because the model was with classes starting from 1
    self.orig_classes = ['None'] + self.orig_classes
    
    self.classes = self.orig_classes.copy()
    
    dct_rename = self.config_graph.get('RENAME_CLASS', {})
    if dct_rename:
      for k,v in dct_rename.items():
        idx = self.classes.index(k)
        self.classes[idx] = v
      #endfor
    #endrename
    
    self.log.p('Setting probabilities')
    model_thr = self.config_graph['MODEL_THRESHOLD']
    dct_class_thr = self.config_graph.get('CLASS_THRESHOLD', {})
    self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr for c in self.classes}
    self.log.p('Done setting probabilities')
    return
    


class LPDv2InferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'LPDV2')
    super().__init__(**kwargs)
    return
  
  def _load_classes(self):
    cls_file = self.config_graph['CLASSES']
    path_cls = self.log.get_model_file(cls_file)
    
    df_cls = pd.read_csv(path_cls, header=None, names=['CLASS'])
    self.orig_classes = df_cls['CLASS'].tolist()
    
    #because the model was with classes starting from 1
    self.orig_classes = self.orig_classes + ['None']
    
    self.classes = self.orig_classes.copy()
    
    dct_rename = self.config_graph.get('RENAME_CLASS', {})
    if dct_rename:
      for k,v in dct_rename.items():
        idx = self.classes.index(k)
        self.classes[idx] = v
      #endfor
    #endrename
    
    self.log.p('Setting probabilities')
    model_thr = self.config_graph['MODEL_THRESHOLD']
    dct_class_thr = self.config_graph.get('CLASS_THRESHOLD', {})
    self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr for c in self.classes}
    self.log.p('Done setting probabilities')
    return
    

class TFOdapi2EffD0InferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_EFFD0')
    super().__init__(**kwargs)
    return
  

class TFOdapi2EffD0512x512InferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_EFFD0_512x512')
    super().__init__(**kwargs)
    return
  
  def _preprocess_images(self, np_imgs):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)    
    if isinstance(np_imgs, (np.ndarray)) and len(np_imgs.shape) == 3:
      np_imgs = np.expand_dims(np_imgs, axis=0)
    
    self.resize_shape = (512, 512, 3)
    lst_shape = [x.shape for x in np_imgs]
    self.input_shape = lst_shape
    res_h, res_w, _ = self.resize_shape
    lst_imgs = [self.log.center_image(x, res_h, res_w) for x in np_imgs if x.shape != self.resize_shape]
    np_imgs = np.array(lst_imgs)
    self.log.stop_timer(timer_name)
    return np_imgs
  

class TFOdapi2EffD0574x1020InferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_EFFD0_574x1020')
    super().__init__(**kwargs)
    return

  def _preprocess_images(self, np_imgs):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)    
    if isinstance(np_imgs, (np.ndarray)) and len(np_imgs.shape) == 3:
      np_imgs = np.expand_dims(np_imgs, axis=0)
    
    self.resize_shape = (574, 1020, 3)
    lst_shape = [x.shape for x in np_imgs]
    self.input_shape = lst_shape
    res_h, res_w, _ = self.resize_shape
    lst_imgs = [self.log.center_image(x, res_h, res_w) for x in np_imgs if x.shape != self.resize_shape]
    np_imgs = np.array(lst_imgs)
    self.log.stop_timer(timer_name)
    return np_imgs


class TFOdapi2EffD7InferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_EFFD7')
    super().__init__(**kwargs)
    return
  
  
class TFOdapi2SSDMobilenetv2FPN640x640InferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_SSDMBNV2_FPN_640x640')
    super().__init__(**kwargs)
    return


class TFOdapi2SSDMobilenetv2320x320InferenceGraph(TFOdapi1InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_SSDMBNV2_320x320')
    super().__init__(**kwargs)
    return


class TFOdapi2CenterNetFPN512x512ferenceGraph(TFOdapi2InferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2_CENTERNET_FPN512x512')
    super().__init__(**kwargs)
    return

