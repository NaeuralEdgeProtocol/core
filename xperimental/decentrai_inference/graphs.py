import os
import gc
import cv2
import math
import traceback
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from tqdm import tqdm
from enum import Enum
from abc import abstractmethod
from time import time
from datetime import datetime
from collections import OrderedDict, defaultdict
from decentrai_inference import constants as ct
from naeural_core import DecentrAIObject
from naeural_core import Logger
from naeural_core.local_libraries.nn.tf import utils as TFUtils

__version__ = '1.3.8.0'

INFERENCE_TIMERS = [ct.TIMER_PREPROCESS_IMAGES,
                    ct.TIMER_SESSION_RUN,
                    ct.TIMER_POSTPROCESS_BOXES]

class BaseInferenceGraph(DecentrAIObject):
  def __init__(self, config_graph=None, config_path=None, config_key=None, **kwargs):
    super().__init__(**kwargs)
    self.__version__ = __version__
    self.tf_runoptions = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    self.config_key = config_key
    if config_graph:
      self.config_graph = config_graph
    else:
      if not os.path.exists(config_path):
        self.log.raise_error('Config path provided does not exist: {}'.format(config_path))
        
      config_graph = self.log.load_json(config_path)      
      if isinstance(config_key, InferenceGraphsEnum):
        config_key = config_key.name
      if config_key and config_key in config_graph:
        self.config_graph = config_graph[config_key]
      else:
        self.config_graph = config_graph
      #endif
    #endif
    _name = kwargs.get('name')
    if _name:
      self.name = _name 
    else:
      self.name = self.__class__.__name__.replace('InferenceGraph', '')
    
    self.maybe_download()
    
    #this dictionary will be used in order to calculate benchmarking inference time
    #you should override this dict for every specific graph
    self.dct_operation_per_image = {
        ct.TIMER_PREPROCESS_IMAGES: False,
        ct.TIMER_SESSION_RUN      : False,
        ct.TIMER_POSTPROCESS_BOXES: False
      }
    
    self.graph_name = self.config_graph['NAME']
    self.graph_type = None
    
    graph_file = self.config_graph['GRAPH'] or self.config_graph['PATH_GRAPH']
    if graph_file.endswith('.pb.trt'):
      self.graph_type = ct.GRAPH_TYPE_TENSORFLOW_TRT
    elif graph_file.endswith('.pth.trt'):
      self.graph_type = ct.GRAPH_TYPE_PYTORCH_TRT
    elif graph_file.endswith('.onnx'):
      self.graph_type = ct.GRAPH_TYPE_ONNX
    elif graph_file.endswith('.pb'):
      self.graph_type = ct.GRAPH_TYPE_PB
    elif graph_file.endswith('.h5'):
      self.graph_type = ct.GRAPH_TYPE_KERAS
    elif graph_file.endswith('.pth'):
      self.graph_type = ct.GRAPH_TYPE_PYTORCH
    else:
      raise ValueError('Graph extension not configured: {}'.format(graph_file))
    
    frmk = self.config_graph.get('FRAMEWORK', None)
    if frmk:
      if frmk == ct.FRAMEWORK_TENSORFLOW:
        self.framework = ct.FRAMEWORK_TENSORFLOW
      elif frmk == ct.FRAMEWORK_PYTORCH:
        self.framework = ct.FRAMEWORK_PYTORCH
      else:
        raise ValueError('Framework not configured!')
    else:
      self.log.p('Framework not specified, choosing default value as Tensorflow')
      self.framework = ct.FRAMEWORK_TENSORFLOW
    #endif
    
    task = self.config_graph.get('TASK', None)
    if task:
      if task == ct.GRAPH_TASK_CLASSIFICATION:
        self.graph_task = ct.GRAPH_TASK_CLASSIFICATION
      elif task == ct.GRAPH_TASK_DETECTION:
        self.graph_task = ct.GRAPH_TASK_DETECTION
      else:
        raise ValueError('Graph task not configured!')
    #endif
    
    self.log.p('Framework {}. Starting {} {} from {} with batch size {}'.format(
      self.framework,
      self.__class__.__name__,
      self.graph_type,
      self.config_graph['GRAPH'],
      self.config_graph.get('BATCH_SIZE', None)
      ))
    return
  
  def __del__(self):
    l_inp = self.get_input_tensors()
    l_out = self.get_output_tensors()
    for x in l_inp + l_out:
      del x
    
    try:
      #maybe you have graphs (like style transfer) without session
      self.sess.close()    
      del self.sess
    except:
      pass
    
    del self.graph
    gc.collect()
    return
  
  def get_session_config(self, mem_fraction):
    self.P("{} uses {} of GPU memory".format(self.__class__.__name__, mem_fraction), color='y')
    gpu_options = tf.GPUOptions()
    
    if tf.__version__[:3] in ['2.1', '2.0']:
      gpu_options.allow_growth = True
      
    if mem_fraction is not None:
      gpu_options.per_process_gpu_memory_fraction = mem_fraction
    
    config = tf.ConfigProto(gpu_options=gpu_options)
    return config
  
  def get_session(self, graph, mem_fraction=None):
    config = self.get_session_config(mem_fraction)
    sess = tf.Session(graph=self.graph, config=config)
    return sess
  
  @abstractmethod
  def _load_pb_graph(self):
    raise NotImplementedError()
    
  @abstractmethod
  def _load_onnx_graph(self):
    raise NotImplementedError()
    
  @abstractmethod
  def _load_tensorflow_trt_graph(self):
    raise NotImplementedError()

  @abstractmethod
  def _load_pytorch_trt_graph(self):
    raise NotImplementedError()
  
  @abstractmethod
  def _postprocess_inference(self):
    raise NotImplementedError()
  
  @abstractmethod
  def _preprocess_images(self, images):
    raise NotImplementedError()
  
  @abstractmethod
  def _run_inference(self):
    raise NotImplementedError()
  
  @abstractmethod
  def _sess_run_pb(self, images):
    raise NotImplementedError()
    
  @abstractmethod
  def _sess_run_onnx(self, images):
    raise NotImplementedError()
  
  @abstractmethod
  def _sess_run_tensorflow_trt(self, images):
    raise NotImplementedError()
    
  @abstractmethod
  def _sess_run_pytorch_trt(self, images):
    raise NotImplementedError()
    
  @abstractmethod
  def _predict_keras(self, images):
    raise NotImplementedError()

  @abstractmethod
  def _predict_pytorch(self, images):
    raise NotImplementedError()

  @abstractmethod
  def _filter(self, images):
    raise NotImplementedError()
  
  @abstractmethod
  def predict(self, images):
    raise NotImplementedError()  # import os
  
  def _load_graph(self):
    if self.graph_type == ct.GRAPH_TYPE_TENSORFLOW_TRT:
      self._load_tensorflow_trt_graph()
    elif self.graph_type == ct.GRAPH_TYPE_PYTORCH_TRT:
      self._load_pytorch_trt_graph()
    elif self.graph_type == ct.GRAPH_TYPE_ONNX:
      self._load_onnx_graph()
    elif self.graph_type == ct.GRAPH_TYPE_PB:
      self._load_pb_graph()
    elif self.graph_type == ct.GRAPH_TYPE_KERAS:
      self._load_keras_model()
    elif self.graph_type == ct.GRAPH_TYPE_PYTORCH:
      self._load_pytorch_model()
    else:
      raise ValueError('Graph type not understood.')
    return
    
  def _sess_run(self, images):
    if self.graph_type == ct.GRAPH_TYPE_TENSORFLOW_TRT:
      return self._sess_run_tensorflow_trt(images)
    elif self.graph_type == ct.GRAPH_TYPE_PYTORCH_TRT:
      return self._sess_run_pytorch_trt(images)
    elif self.graph_type == ct.GRAPH_TYPE_ONNX:
      return self._sess_run_onnx(images)
    elif self.graph_type == ct.GRAPH_TYPE_PB:
      return self._sess_run_pb(images)
    elif self.graph_type == ct.GRAPH_TYPE_KERAS:
      return self._predict_keras(images)
    elif self.graph_type == ct.GRAPH_TYPE_PYTORCH:
      return self._predict_pytorch(images)
    else:
      raise ValueError('Graph type not understood.')
    return
  
  def _calculate_benchmark_timing(self, batch_size, lst_timers=INFERENCE_TIMERS):
    assert all(x in INFERENCE_TIMERS for x in lst_timers), 'Please provide allowed inference timers!'
    l = []
    for timer in lst_timers:
      tn = self.timer_name(timer)
      t = self.log.get_timer(tn)
      if self.dct_operation_per_image[timer]:
        t = t * batch_size
      l.append(t)    
    t_per_img = sum(l) / batch_size
    return t_per_img
  
  def get_input_names(self):
    lst_inp = []
    if hasattr(self, 'model'):
      lst_inp = self.model.inputs
    elif hasattr(self, 'graph'):
      graph_file = self.config_graph['GRAPH']
      cfg = self.log.load_models_json(graph_file + '.txt')    
      if cfg: #optional cfg
        lst_inp = [v for k,v in cfg.items() if k.startswith('INPUT_')]
      else:
        assert self.config_graph['INPUT_TENSORS'], 'Please provide input tensor names'
        lst_inp = self.config_graph['INPUT_TENSORS']
    else:
      raise ValueError('You should provide either a graph (pb/trt/onnx) or a keras model')
    lst_inp = [x.replace(':0', '') for x in lst_inp]
    return lst_inp
  
  def get_output_names(self):
    lst_out = []
    if hasattr(self, 'model'):
      lst_out = self.model.outputs
    elif hasattr(self, 'graph'):
      graph_file = self.config_graph['GRAPH']
      cfg = self.log.load_models_json(graph_file + '.txt')
      if cfg: #optional cfg
        lst_out = [v for k,v in cfg.items() if k.startswith('OUTPUT_')]      
      else:
        assert self.config_graph['OUTPUT_TENSORS'], 'Please provide output tensor names'      
        lst_out = self.config_graph['OUTPUT_TENSORS']
    else:
      raise ValueError('You should provide either a graph (pb/trt/onnx) or a keras model')
    lst_out = [x.replace(':0', '') for x in lst_out]
    return lst_out
    
  def get_input_tensors(self):
    lst_inp = self.get_input_names()
    tensors_input = [self.graph.get_tensor_by_name(x+':0') for x in lst_inp]
    return tensors_input
  
  def get_output_tensors(self):
    lst_out = self.get_output_names()
    tensors_output = [self.graph.get_tensor_by_name(x+':0') for x in lst_out]
    return tensors_output
  
  def predict_dummy(self):
    try:
      np_imgs = np.random.uniform(size=(1, 720, 1280, 3))
      self.log.p('Making dummy predictions on graph {} with tensor of shape: {}'.format(self.__class__.__name__, np_imgs.shape))
      self.predict(np_imgs)
      self.log.p('Done making dummy predictions', show_time=True)
    except:
      str_e = traceback.format_exc()
      self.log.p('Exception on dummy inference: {}'.format(str_e), color='red')
    return
  
  def maybe_download(self):
    #return because all download is done when app starts
    return

    for k,v in self.config_graph.items():
      if 'URL_' in k and v != '':
        #expect dropbox url
        _str = os.path.split(v)[1]
        filename = _str.split('?')[0]
        if not self.log.get_model_file(filename):
          self.log.maybe_download(url=v, fn=filename)
    return
    
  def timer_name(self, name=''):
    tn = ''
    if name == '':
      tn = self.__class__.__name__
    else:
      tn = '{}__{}'.format(self.__class__.__name__, name)
    return tn
  
  def benchmark_graph(self, 
                      lst_batch_size, 
                      images, 
                      unique_batches=True,
                      nr_warm_up=10, 
                      nr_inference=100, 
                      augment_images=False,
                      lst_timers=INFERENCE_TIMERS
                      ):
    def _get_batch(batch_size):
      imgs = images
      if isinstance(images, list):
        imgs = np.array(images)
      imgs = imgs[:batch_size]
      imgs = [x for x in imgs]
      return imgs
    #enddef
    
    def _get_unique_batch(batch_size, last_indexes=[]):
      all_idxes = np.arange(len(images))      
      crt_idxes = all_idxes
      if last_indexes:
        crt_idxes = list(set(all_idxes) - set(last_indexes))
      new_sel = np.random.choice(crt_idxes, size=batch_size, replace=False).tolist()
      assert len(set(last_indexes).intersection(set(new_sel))) == 0
      last_indexes = new_sel
      
      imgs = images
      if isinstance(images, list):
        imgs = np.array(images)
      imgs = imgs[last_indexes]
      imgs = [x for x in imgs]
      return imgs, last_indexes
    #enddef
    
    def _repeat_images(batch_size):
      lst = []
      while len(lst) < batch_size:
        for img in images:
          lst.append(img)
      return lst
    #enddef
    
    def _augment_images(batch_size):
      lst = []
      while len(lst) < batch_size:
        for img in images:
          w = img.shape[1]
          h = img.shape[0]
          #rotate matrix
          M = cv2.getRotationMatrix2D((w/2,h/2), 90, 1)
          #rotate
          img = cv2.warpAffine(img,M,(w,h))
          hflip, vflip = np.random.choice(2, size=2, replace=True)
          if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            img = cv2.flip(img, flipCode=c)
          #endif
          lst.append(img)
        #endfor
      #endwhile
      return lst
    #enddef
        
    
    assert (isinstance(images, (list, np.ndarray)) and len(images) >= 1)
    
    can_build_unique_batches = True
    lst_sum = [x.sum() for x in images]
    if len(lst_sum) > len(set(lst_sum)) and unique_batches:
      self.log.p('[Warning] Please note that this results may not be correct \
                 because caching can appear between batch inference')
      self.log.p('[Warning] When benchmarking a graph you should provide \
                 enough UNIQUE images so that you can be sure that no caching \
                is done between inferences. A short analyze\
                of the data shows that you have provided {} unique images'.format(len(set(lst_sum))))
      can_build_unique_batches = False
    #endif
    
    if len(images) < max(lst_batch_size) and unique_batches:
      self.log.p('[Warning] Please note that this results may not be correct \
                 because caching can appear between batch inference.')
      self.log.p('[Warning] You haven`t provided enough images for unique batches.\
                 You`re max batch size is {} and you provided {} images'.format(max(lst_batch_size), len(images)))
      can_build_unique_batches = False
    #endif    
    
    dct_res = {}
    for batch_size in lst_batch_size:
      try:
        lst_times_warmup    = []
        lst_times_inference = []
        #warmup timing
        self.log.reset_timers()
        last_indexes = []
        for _ in tqdm(range(nr_warm_up)):
          if unique_batches:
            if can_build_unique_batches:
              x_batch, last_indexes = _get_unique_batch(batch_size, last_indexes)
            elif augment_images:
              x_batch = _augment_images(batch_size)
            else:
              x_batch = _repeat_images(batch_size)
          else:
            x_batch = _get_batch(batch_size)
          start = time()
          self.predict(x_batch)
          stop = time()
          lst_times_warmup.append(stop - start)
        #endfor
        
        time_warmup = self._calculate_benchmark_timing(batch_size=batch_size)
        
        #actual inference timing
        self.log.reset_timers()
        for _ in tqdm(range(nr_inference)):
          if unique_batches:
            if can_build_unique_batches:
              x_batch, last_indexes = _get_unique_batch(batch_size, last_indexes)
            elif augment_images:
              x_batch = _augment_images(batch_size)
            else:
              x_batch = _repeat_images(batch_size)
          else:
            x_batch = _get_batch(batch_size)
            
          start = time()
          dct_inf = self.predict(x_batch)
          stop = time()
          lst_times_inference.append(stop - start)
        #endfor
        lst_inf = dct_inf['INFERENCES']
        
        time_inference = self._calculate_benchmark_timing(batch_size=batch_size)
          
        dct_res[batch_size] = {
          'TIMES_WARMUP'    : lst_times_warmup,
          'TIMES_INFERENCE' : lst_times_inference,
          'TIME_WARMUP'     : time_warmup,
          'TIME_INFERENCE'  : time_inference,
          'INFERENCES'      : lst_inf,
          'ERROR'           : False
          }
      except Exception as e:
        self.log.p('Exception for bs {}: {}'.format(batch_size, str(e)))
        dct_res[batch_size] = {
          'TIMES_WARMUP'    : [],
          'TIMES_INFERENCE' : [],
          'TIME_WARMUP'     : 0,
          'TIME_INFERENCE'  : 0,
          'INFERENCES'      : [],
          'ERROR'           : True
          }
    #endfor
    return dct_res

  def simple_benchmark(self, np_imgs, batch_size, n_warmup=1, n_iters=1, 
                       auto_fill_batches=True, **kwargs):
    def _get_nr_batches(np_imgs, batch_size):
      nr_batches = int(math.ceil(len(np_imgs) / batch_size))
      return nr_batches
  
    def _data_generator(np_imgs, batch_size):
      nr_batches = _get_nr_batches(np_imgs, batch_size)
      for i in range(nr_batches):
        start = i * batch_size
        stop = (i + 1) * batch_size if i < nr_batches - 1 else len(np_imgs)
        np_batch = np_imgs[start:stop]
        if auto_fill_batches and len(np_batch) < batch_size:
          bs, h, w, c = np_batch.shape
          np_batch = np.vstack([np_batch, np.repeat(np.expand_dims(np_batch[0], axis=0), batch_size-bs, axis=0)])
        yield np_batch
      return
    
    dct_times_warmup    = {'ITER': [], 'NR_BATCH': [], 'TIME': []}
    dct_times_inference = {'ITER': [], 'NR_BATCH': [], 'TIME': []}
    
    if auto_fill_batches:
      nearest_bs_multiple = math.ceil(np_imgs.shape[0] / batch_size) * batch_size
      nr_missing = nearest_bs_multiple - np_imgs.shape[0]
      lst = [x for x in np_imgs]
      lst+= [lst[-1]] * nr_missing
      np_imgs = lst
    
    #warmup
    for i in range(1, n_warmup+1):
      self.log.p(' Warmup {}'.format(i))
      data_gen = _data_generator(np_imgs=np_imgs, batch_size=batch_size)
      
      for nr, np_batch in enumerate(data_gen):
        try:
          start = time()
          self.predict(np_batch, **kwargs)
          stop = time()
          dct_times_warmup['ITER'].append(i)
          dct_times_warmup['NR_BATCH'].append(nr)
          dct_times_warmup['TIME'].append(stop - start)
        except Exception as e:
          dct_times_warmup['ITER'].append(i)
          dct_times_warmup['NR_BATCH'].append(nr)
          dct_times_warmup['TIME'].append(None)
          self.log.p('Exception on warmup {} nr batch{} : {}'.format(i, nr, str(e)), color='r')
      self.log.p(' Done warmup {}'.format(i), show_time=True)
    #end warmup
    
    #iters
    for i in range(1, n_iters+1):
      self.log.p(' Iter {}'.format(i))
      data_gen = _data_generator(np_imgs=np_imgs, batch_size=batch_size)
      
      for nr, np_batch in enumerate(data_gen):
        try:
          start = time()
          self.predict(np_batch, **kwargs)
          stop = time()
          dct_times_inference['ITER'].append(i)
          dct_times_inference['NR_BATCH'].append(nr)
          dct_times_inference['TIME'].append(stop - start)
        except Exception as e:
          dct_times_inference['ITER'].append(i)
          dct_times_inference['NR_BATCH'].append(nr)
          dct_times_inference['TIME'].append(None)
          self.log.p('Exception on warmup {} nr batch{} : {}'.format(i, nr, str(e)), color='r')
      self.log.p(' Done', show_time=True)
    #end iters
    
    return dct_times_warmup, dct_times_inference
  
  def benchmark_batch(self, images, nr_warm_up=10, nr_inference=100):
    self.log.p('[Warning] Obsolete method in case you need to benchmark graph inference time. Plase use `benchmark_graph` method instead')
    assert (isinstance(images, list) and len(images) >= 1) or (isinstance(images, np.ndarray) and len(images.shape) == 4)

    lst_times_warmup    = []
    lst_times_inference = []
    
    #warmup timing
    self.log.reset_timers()
    self.log.p('Warm up on tensor with shape: {} for {} times'.format(images.shape, nr_warm_up))
    for _ in tqdm(range(nr_warm_up)):
      start = time()
      self.predict(images)
      stop = time()
      lst_times_warmup.append(stop - start)
    #endfor
    
    time_warmup = self._calculate_benchmark_timing(batch_size=len(images))
    
    #actual inference timing
    self.log.reset_timers()
    self.log.p('Benchmark on tensor with shape: {} for {} times'.format(images.shape, nr_inference))
    for _ in tqdm(range(nr_inference)):
      start = time()
      dct_inf = self.predict(images)
      stop = time()
      lst_times_inference.append(stop - start)
    #endfor
    lst_inf = dct_inf['INFERENCES']
    
    time_inference = self._calculate_benchmark_timing(batch_size=len(images))
      
    dct_res = {
                'TIMES_WARMUP'    : lst_times_warmup,
                'TIMES_INFERENCE' : lst_times_inference,
                'TIME_WARMUP'     : time_warmup,
                'TIME_INFERENCE'  : time_inference,
                'INFERENCES'      : lst_inf
              }
    return dct_res
  
  def benchmark_image(self, image, batch_size=1, nr_warm_up=10, nr_inference=100):
    self.log.p('[Warning] Obsolete method in case you need to benchmark graph inference time. Plase use `benchmark_graph` method instead')
    #TODO: timers enum
    assert len(image.shape) == 3
    images = np.expand_dims(image, axis=0)
    images = np.repeat(images, batch_size, axis=0)
    dct_res = self.benchmark_batch(images=images, nr_warm_up=nr_warm_up, nr_inference=nr_inference)
    return dct_res
  
  def _fill_batch_most_common_shape(self, images):
    lst_images = [x for x in images]
    nr_crt = len(lst_images)
    nr_completed = 0
    bs = self.config_graph['BATCH_SIZE']
    if nr_crt != bs:
      arr = np.array(range(0, 100, bs))
      idx = np.argwhere((arr - nr_crt) >= 0).ravel()[0]
      nr_total = arr[idx]
      nr_completed = nr_total - nr_crt
      lst_shapes = [x.shape for x in images]
      unique, counts = np.unique(lst_shapes, return_counts=True, axis=0)
      shape = unique[np.argmax(counts)]
      lst_completed = [np.zeros(shape=shape, dtype='uint8') for _ in range(nr_completed)]
      lst_images+= lst_completed
    #endif
    dct_filled = {None: {'IMGS': lst_images, 'COMPLETED': nr_completed}}
    return dct_filled
  
  def _fill_batch_per_shape(self, images):
    dct_images = defaultdict(lambda : [])
    for x in images:
      dct_images[x.shape].append(x)
    dct_filled = {}
    for k, lst in dct_images.items():
      dct = self._fill_batch_most_common_shape(lst)
      dct_filled[k] = dct[None]
    return dct_filled
  
  def _fill_batch_default(self, images):
    return {None: {'IMGS': [x for x in images], 'COMPLETED': 0}}
  
  def _fill_batch(self, np_imgs, batch_strategy_name):
    """
    This method filles the current batch according to batch_strategy_name

    Parameters
    ----------
    np_imgs : List or Array
      Images used for batch inference.
    batch_strategy_name : str
      Batch strategy used.

    Raises
    ------
    ValueError
      DESCRIPTION.

    Returns
    -------
    dct_filled: Dict
      Dictionary with containing filled images per shape type.

    """
    if batch_strategy_name == ct.BATCH_STRATEGY_MOST_COMMON_SHAPE:
      return self._fill_batch_most_common_shape(np_imgs)
    elif batch_strategy_name == ct.BATCH_STRATEGY_PER_SHAPE:
      return self._fill_batch_per_shape(np_imgs)
    elif batch_strategy_name == ct.BATCH_STRATEGY_DEFAULT:
      return self._fill_batch_default(np_imgs)
    else:
      raise ValueError('Not implemented strategy: {}'.format(batch_strategy_name))
    return
  
  def predict_batch(self, np_imgs):
    assert isinstance(np_imgs, (list, np.ndarray))
    timer_name = self.timer_name(ct.TIMER_PREDICT_BATCH)
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      result = None
      batch_strategy = self.config_graph.get('BATCH_STRATEGY', ct.BATCH_STRATEGY_MOST_COMMON_SHAPE)
      # batch_strategy = ct.BATCH_STRATEGY_DEFAULT.name
      dct_imgs = self._fill_batch(np_imgs, batch_strategy)
        
      batch_size = self.config_graph['BATCH_SIZE']
      nr_total_batches = int(sum([math.ceil(len(x['IMGS']) / batch_size) for x in dct_imgs.values()]))
#      print('Using {} strategy, batch size: {}, nr batches: {}'.format(batch_strategy, batch_size, nr_total_batches))
      for shape, dct_filled in dct_imgs.items():
        np_imgs = dct_filled['IMGS']
        nr_batches = int(math.ceil(len(np_imgs) / batch_size))
        nr_completed = 0
        t_start = time()
        for i in range(nr_batches):
          start = i * batch_size
          stop = (i + 1) * batch_size
          batch_images = np_imgs[start: stop]
          if type(batch_images) == np.ndarray and len(batch_images.shape) == 1:
            batch_images = batch_images.tolist()
          
          if i == nr_batches - 1:
            nr_completed = dct_filled['COMPLETED'] #completed images are only in the last batch
#          print(' Predicting on {} images'.format(len(batch_images)))
          res = self.predict(batch_images)['INFERENCES']
          if nr_completed > 0:
            res = res[:-nr_completed]
          if result is None:
            result = res
          else:
            if isinstance(res, np.ndarray):
              result = np.vstack([result, res])
            else:
              result+= res
          #endif
        #endfor
        t_stop = time()
#        print('Total number of batches: {}. Nr. completed: {}'.format(nr_batches, nr_completed))
#        print('Total time: {}'.format(t_stop - t_start))
      #endfor
    #endif
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  
class YoloInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_YOLO')
    
    super().__init__(**kwargs)
    
    self.dct_operation_per_image = {
        ct.TIMER_PREPROCESS_IMAGES: False,
        ct.TIMER_SESSION_RUN:       True,
        ct.TIMER_POSTPROCESS_BOXES: True
      }
    
    self._load_coco_classes()
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_coco_classes(self):
    cls_file = self.config_graph['CLASSES']
    full_cls_file = os.path.join(self.log.get_models_folder(), cls_file)
    self.log.p('Loading {}...'.format(full_cls_file))
    with open(full_cls_file) as f:
      lines = f.read().splitlines()
    self.orig_classes = lines.copy() + [lines[-1]]*150 # pad last class...
    #rename classes
    self.classes = self.orig_classes.copy()
    dct_rename = self.config_graph['RENAME_CLASS']
    for k,v in dct_rename.items():
      idx = self.classes.index(k)
      self.classes[idx] = v
    #endfor
    self.log.p('Loaded {} classes from {}'.format(len(lines), cls_file))
    
    self.log.p('Setting probabilities')
    model_thr = self.config_graph['MODEL_THRESHOLD']
    dct_class_thr = self.config_graph['CLASS_THRESHOLD']
    self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr for c in self.classes}
    self.log.p('Done setting probabilities')
    return
  
  def _load_pb_graph(self):
    self.log.p('Loading TF PB graph', color='g')
    timer_name = self.timer_name(name=ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(timer_name)
    self.yolo_model_shape = (self.config_graph['YOLO_MODEL_SIZE'], 
                             self.config_graph['YOLO_MODEL_SIZE'])

    self.classes_tensor_name = self.config_graph['YOLO_CLASSES_TENSOR_NAME']
    self.scores_tensor_name = self.config_graph['YOLO_SCORES_TENSOR_NAME']
    self.boxes_tensor_name = self.config_graph['YOLO_BOXES_TENSOR_NAME']
    self.input_tensor_name = self.config_graph['YOLO_INPUT_TENSOR_NAME']
    
    path_graph = self.config_graph['PATH_GRAPH']
    if path_graph and os.path.exists(path_graph):
      self.log.p('Loading graph from specific path: {}'.format(path_graph))
      self.graph = self.log.load_tf_graph(path_graph)
    else:
      graph_name = self.config_graph['GRAPH']
      self.log.p('Loading graph from models: {}'.format(graph_name))
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)

    self.tf_classes = self.sess.graph.get_tensor_by_name(self.classes_tensor_name+':0')
    self.tf_scores = self.sess.graph.get_tensor_by_name(self.scores_tensor_name+':0')
    self.tf_boxes = self.sess.graph.get_tensor_by_name(self.boxes_tensor_name+':0')
    self.tf_yolo_input = self.sess.graph.get_tensor_by_name(self.input_tensor_name+':0')
    self.tensor_inputs = [self.tf_yolo_input]
    self.tensor_outputs = [self.tf_scores, self.tf_boxes, self.tf_classes]
    self.log.stop_timer(timer_name)
    return
  
  def _load_onnx_graph(self):
    self.log.p('Loading ONNX graph', color='g')
    timer_name = self.timer_name(name=ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(timer_name)
    self.yolo_model_shape = (self.config_graph['YOLO_MODEL_SIZE'], 
                             self.config_graph['YOLO_MODEL_SIZE'])
    graph_name = self.config_graph['GRAPH']
    model, ort_session, input_names, output_names = self.log.load_onnx_model(
      model_name=graph_name
      )
    self.sess = ort_session
    self.tensor_inputs = input_names
    self.tensor_outputs = output_names
    self.log.stop_timer(timer_name)
    return
  
  def _load_trt_graph(self):
    self.log.p('Loading TRT graph', color='g')
    timer_name = self.timer_name(name=ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(timer_name)
    self.yolo_model_shape = (self.config_graph['YOLO_MODEL_SIZE'], 
                             self.config_graph['YOLO_MODEL_SIZE'])
    graph_name = self.config_graph['GRAPH']    
    
    trt_graph, lst_inputs, lst_outputs = self.log.load_trt_graph(
      graph_file=graph_name
      )
    
    self.sess = tf.Session(graph=trt_graph)
    self.tensor_inputs = [trt_graph.get_tensor_by_name(x) for x in lst_inputs]
    self.tensor_outputs = [trt_graph.get_tensor_by_name(x) for x in lst_outputs]
    self.log.stop_timer(timer_name)
    return
  
  def _sess_run_onnx(self, images):
    self.log.p('Session run ONNX', color='y')
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    preds = self.sess.run(
      input_feed={self.tensor_inputs[0]: images},
      output_names=self.tensor_outputs
      )
    self.log.stop_timer(timer_name)
    return preds  
  
  def _sess_run_pb(self, images):
    self.log.p('Session run PB', color='y')
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    out_scores, out_boxes, out_classes = self.sess.run(
        self.tensor_outputs,
        feed_dict={self.tensor_inputs[0]: images},
        options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return out_scores, out_boxes, out_classes
  
  def _sess_run_trt(self, images):
    self.log.p('Session run TRT', color='y')
    return self._sess_run_pb(images)
  
  def _postprocess_boxes(self, boxes, idx_image=0):
    tn = self.timer_name(name=ct.TIMER_POSTPROCESS_BOXES)
    self.log.start_timer(tn)
    img_shape = self.input_shape[idx_image]
    if self.center_image:
      (top, left, bottom, right), (new_h, new_w) = self.log.center_image_coordinates(src_h=img_shape[0], 
                                                                                     src_w=img_shape[1], 
                                                                                     target_h=self.yolo_model_shape[0], 
                                                                                     target_w=self.yolo_model_shape[1])
      #[0:1] to [0:yolo_model_shape]
      boxes[:,0] = boxes[:,0] * self.yolo_model_shape[0]
      boxes[:,1] = boxes[:,1] * self.yolo_model_shape[1]
      boxes[:,2] = boxes[:,2] * self.yolo_model_shape[0]
      boxes[:,3] = boxes[:,3] * self.yolo_model_shape[1]
      
      #eliminate centering
      boxes[:,0] = boxes[:,0] - top
      boxes[:,1] = boxes[:,1] - left
      boxes[:,2] = boxes[:,2] - top
      boxes[:,3] = boxes[:,3] - left
      
      #translate to original image
      boxes[:,0] = boxes[:,0] / new_h * img_shape[0]
      boxes[:,1] = boxes[:,1] / new_w * img_shape[1]
      boxes[:,2] = boxes[:,2] / new_h * img_shape[0]
      boxes[:,3] = boxes[:,3] / new_w * img_shape[1]
    else:
      boxes[:,0] = boxes[:,0] * img_shape[0]
      boxes[:,1] = boxes[:,1] * img_shape[1]
      boxes[:,2] = boxes[:,2] * img_shape[0]
      boxes[:,3] = boxes[:,3] * img_shape[1]
    #endif
    boxes = boxes.astype(np.int32)
    boxes[:, 0] = np.maximum(0, boxes[:, 0])
    boxes[:, 1] = np.maximum(0, boxes[:, 1])
    boxes[:, 2] = np.minimum(img_shape[0], boxes[:, 2])
    boxes[:, 3] = np.minimum(img_shape[1], boxes[:, 3])
    self.log.stop_timer(tn)
    return boxes
  
  def _postprocess_inference(self, scores, boxes, classes):
    batch_frames = []
    nr_frames = len(scores)
    for _frame in range(nr_frames):
      frame_data = []
      frame_scores = scores[_frame]
      frame_boxes = boxes[_frame]
      frame_classes = classes[_frame].astype(int)
      for _id in range(frame_classes.shape[0]):
        _type = self.classes[frame_classes[_id]]
        if frame_scores[_id] >= self.probas[_type]:
          frame_data.append({
                "TLBR_POS":np.around(frame_boxes[_id]).tolist(), # [TOP, LEFT, BOTTOM, RIGHT]
                "PROB_PRC":np.around(frame_scores[_id]).item(),
                "TYPE": _type
              })
      #end frame iter      
      batch_frames.append(frame_data)
    return batch_frames
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    all_scores = []
    all_boxes = []
    all_classes = []
    for i in range(images.shape[0]):
      imgs = images[i:i+1,:,:,:]
      score, boxes, classes = self._sess_run(imgs)
      boxes = self._postprocess_boxes(boxes, i)
      all_scores.append(score)
      all_boxes.append(boxes)
      all_classes.append(classes)
    #endfor
    batch_frames = self._postprocess_inference(all_scores, all_boxes, all_classes)
    self.log.stop_timer(timer_name)
    return batch_frames
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(name=ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    self.input_shape = [x.shape for x in images]
    if self.center_image:
      images = np.array([self.log.center_image2(x, target_h=self.yolo_model_shape[0], target_w=self.yolo_model_shape[1])
                         for x in images])
    #endif
    
    if images.max() > 1:
      images = images.astype(np.float32) / 255.      
    #endif
      
    shape = images.shape[1:3]
    if shape != self.yolo_model_shape:
      images = np.array([cv2.resize(frame, (self.yolo_model_shape[0], self.yolo_model_shape[1]))
                         for frame in images])
    #endif
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs, preprocess=True, center_image=True):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    
    
    self.center_image = center_image
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      if preprocess:
        _prep_images = self._preprocess_images(np_imgs)
      else:
        _prep_images = np_imgs
      #endif
      result = self._run_inference(_prep_images)
    #endif
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result
  
  def get_input_names(self):
    return [self.input_tensor_name+':0']
    
  def get_output_names(self):
    return [self.scores_tensor_name+':0', 
            self.boxes_tensor_name+':0', 
            self.classes_tensor_name+':0']
  
  def get_input_tensors(self):
    return self.tf_yolo_input

  def get_output_tensors(self):
    return [self.tf_scores, self.tf_boxes, self.tf_classes]
  

class LPDetectionInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'LP_DETECTION')
    
    super().__init__(**kwargs)
    
    self.dct_operation_per_image = {
        ct.TIMER_PREPROCESS_IMAGES: False,
        ct.TIMER_SESSION_RUN:       False,
        ct.TIMER_POSTPROCESS_BOXES: True
      }
    self._load_graph()
    # self.predict_dummy()
    return
    
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    if path_graph and os.path.exists(path_graph):
      self.log.p('Loading graph from specific path: {}'.format(path_graph))
      self.graph = self.log.load_tf_graph(os.path.join(path_graph, graph_name))
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'

    self.sess = tf.Session(graph=self.graph)
    self.tf_image    = self.graph.get_tensor_by_name(self.config_graph['INPUT_0'])
    self.tf_boxes    = self.graph.get_tensor_by_name(self.config_graph['BOXES'])
    self.tf_scores   = self.graph.get_tensor_by_name(self.config_graph['SCORES'])
    self.tf_classes  = self.graph.get_tensor_by_name(self.config_graph['CLASSES'])
    self.tf_num_dets = self.graph.get_tensor_by_name(self.config_graph['NUM_DETECTIONS'])
    self.log.p('Done loading TF ODAPI License Plate Detection graph')
    return
  
  def _postprocess_inference(self, scores, boxes, classes):
    batch_frames = []
    batch_size = scores.shape[0]
    for _frame in range(batch_size):
      frame_scores = scores[_frame]
      frame_boxes = boxes[_frame]
      lst_over_thr = np.where(frame_scores >= self.config_graph['MODEL_THRESHOLD'])[0]
      frame_data = [{
                      "TLBR_POS": np.around(frame_boxes[_id]).tolist(), # [TOP, LEFT, BOTTOM, RIGHT]
                      "PROB_PRC": np.around(frame_scores[_id]).item(),
                      "TYPE"    : 'license_plate'
                    } for _id in lst_over_thr]
      batch_frames.append(frame_data)
    return batch_frames
  
  def _sess_run(self, images):
    if len(images.shape) != 4:
      raise ValueError("Input must be BHWC, received {}".format(images.shape))
    
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    (out_boxes, out_scores, out_classes, out_num_detections) = self.sess.run(
            [self.tf_boxes, self.tf_scores, self.tf_classes, self.tf_num_dets],
            feed_dict={self.tf_image: images},
            options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return out_scores, out_boxes, out_classes
  
  def _postprocess_boxes(self, images, all_boxes):
    timer_name = self.timer_name(name=ct.TIMER_POSTPROCESS_BOXES)
    self.log.start_timer(timer_name)
    max_h, max_w, _= max(self.input_shape)
    lst_boxes = []
    for idx in range(images.shape[0]):
      img_shape = self.input_shape[idx]
      (top, left, bottom, right), (new_h, new_w) = self.log.center_image_coordinates(
          src_h=img_shape[0],
          src_w=img_shape[1],
          target_h=max_h,
          target_w=max_w
          )
      boxes = all_boxes[idx]
      
      #[0:1] to [0:yolo_model_shape]
      boxes[:,0] = boxes[:,0] * max_h
      boxes[:,1] = boxes[:,1] * max_w
      boxes[:,2] = boxes[:,2] * max_h
      boxes[:,3] = boxes[:,3] * max_w
      
      #eliminate centering
      boxes[:,0] = boxes[:,0] - top
      boxes[:,1] = boxes[:,1] - left
      boxes[:,2] = boxes[:,2] - top
      boxes[:,3] = boxes[:,3] - left
      
      #translate to original image
      boxes[:,0] = boxes[:,0] / new_h * img_shape[0]
      boxes[:,1] = boxes[:,1] / new_w * img_shape[1]
      boxes[:,2] = boxes[:,2] / new_h * img_shape[0]
      boxes[:,3] = boxes[:,3] / new_w * img_shape[1]
      
      lst_boxes.append(boxes)
    #endfor
    np_boxes = np.array(lst_boxes)
    np_boxes = np_boxes.astype(np.int32)
    self.log.stop_timer(timer_name)
    return np_boxes
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    
    all_scores, all_boxes, all_classes = self._sess_run(images)
    all_boxes = self._postprocess_boxes(images, all_boxes)
    all_classes = all_classes.astype(np.int32)
    batch_frames = self._postprocess_inference(all_scores, all_boxes, all_classes)
    self.log.stop_timer(timer_name)
    return batch_frames
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(name=ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    self.input_shape = [x.shape for x in images]
    nr_distinct_shapes = len(set(x.shape for x in images))
    if nr_distinct_shapes > 1:
      max_shape = max(self.input_shape)
      max_h = max_shape[0]
      max_w = max_shape[1]
      images = np.array([self.log.center_image2(x, target_h=max_h, target_w=max_w) 
                          for x in images])
    else:
      images = np.array(images)
    #endif
    
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    self.input_shape = None
    self._timestamp = datetime.now()
    self._str_timestamp = self._timestamp.strftime("%Y%m%d_%H%M%S")

    dct_result = OrderedDict()
    dct_meta = OrderedDict()

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
    #endif
    
    dct_meta['SYSTEM_TIME'] = self._str_timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result


class LPRInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'LPR')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return  
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(self.tf_output,
                            feed_dict={self.tf_input: images},
                            options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return probs
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    
    if isinstance(images, np.ndarray):
      images = images / 255
      images = images.astype(np.float32)
    else:
      images = [x/255 for x in images]
      images = [x.astype(np.float32) for x in images]
    
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    images = [cv2.resize(img, (self.config_graph['IMG_H'], self.config_graph['IMG_W'])) for img in images]
    images = [np.expand_dims(img, -1) for img in images]
    images = np.array(images)
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result


class OmvEmployeeInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'OMV_EMPLOYEE')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
    
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return    
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(
            [self.tf_output],
            feed_dict={self.tf_input: images},
            options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return probs
  
  #TODO: replace with log.center_image and TEST
  def _center_image(self, image, resize_h, resize_w, fill_value=127):
    new_h, new_w, _ = image.shape
  
    # determine the new size of the image
    if (float(resize_w)/new_w) < (float(resize_h)/new_h):
        new_h = int((new_h * resize_w)/new_w)
        new_w = int(resize_w)
    else:
        new_w = int((new_w * resize_h)/new_h)
        new_h = int(resize_h)
        
    resized = cv2.resize(image, (new_w, new_h))
  
    # embed the image into the standard letter box
    new_image = np.ones((resize_h, resize_w, 3), dtype=np.uint8 if isinstance(fill_value, int) else np.float32) * fill_value
    top = int((resize_h-new_h)//2)
    bottom = int((resize_h+new_h)//2)
    left = int((resize_w-new_w)//2)
    right = int((resize_w+new_w)//2)
    
    new_image[top:bottom, left:right, :] = resized
    return new_image
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
      
    if isinstance(images, np.ndarray):
      images = images / 255
      images = images.astype(np.float32)
    else:
      images = [x / 255 for x in images]
      images = [x.astype(np.float32) for x in images]
    #endif
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [self._center_image(img, resize_h, resize_w, 0.5) for img in images]
    images = np.array(images)        
    preds = self._sess_run(images)[0]
    self.log.stop_timer(timer_name)
    return preds

  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  
class FoodEmptyInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'GASTRO')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return  
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(self.tf_output, feed_dict={self.tf_input: images}, options=self.tf_runoptions)
    probs = probs.ravel().tolist()
    self.log.stop_timer(timer_name)
    return probs
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)   
    preds = self._sess_run(images)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [cv2.resize(img, (resize_h, resize_w)) for img in images]
    images = np.array(images)  
    images = images / 255.
    images = images.astype(np.float32)
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result


class MerchandiserInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'MERCHANDISER')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input  = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(
            self.tf_output,
            feed_dict={self.tf_input: images},
            options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return probs
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [self.log.center_image(x, target_h=resize_h, target_w=resize_w) for x in images]
    images = np.array(images)        
    images = tf.keras.applications.xception.preprocess_input(images)
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result


class AlimentareNepermisaInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'ALIMENTARE_NEPERMISA')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(self.tf_output, feed_dict={self.tf_input: images}, options=self.tf_runoptions)
    probs = probs.ravel().tolist()
    self.log.stop_timer(timer_name)
    return probs
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [self.log.center_image(x, target_h=resize_h, target_w=resize_w) for x in images]
    images = np.array(images)
    images = (images / 255).astype('float32')
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result
  
  def get_input_tensors(self):
    return [self.tf_input]
  
  def get_output_tensors(self):
    return [self.tf_output]
  

class FireSmokeInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'FIRE_SMOKE')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(self.tf_output, feed_dict={self.tf_input: images}, options=self.tf_runoptions)
    probs = probs.ravel().tolist()
    self.log.stop_timer(timer_name)
    return probs
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [self.log.center_image(x, target_h=resize_h, target_w=resize_w) for x in images]
    images = np.array(images)
    images = images / 255.
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  def get_input_tensors(self):
    return [self.tf_input]
  
  def get_output_tensors(self):
    return [self.tf_output]


class FaceInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'FACE_DETECTION')
    
    super().__init__(**kwargs)
    self.dct_operation_per_image = {
        ct.TIMER_PREPROCESS_IMAGES: False,
        ct.TIMER_SESSION_RUN:       False,
        ct.TIMER_POSTPROCESS_BOXES: True
      }
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):   
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
        
    face_memory_prc = self.config_graph['MEMORY_FRACTION']
    self.sess = self.get_session(
      graph=self.graph,
      mem_fraction=face_memory_prc,
      )
    
    self.tf_classes      = self.sess.graph.get_tensor_by_name(self.config_graph['TENSOR_CLASSES']+':0')
    self.tf_scores       = self.sess.graph.get_tensor_by_name(self.config_graph['TENSOR_SCORES']+':0')
    self.tf_boxes        = self.sess.graph.get_tensor_by_name(self.config_graph['TENSOR_BOXES']+':0')
    self.tf_detections   = self.sess.graph.get_tensor_by_name(self.config_graph['TENSOR_DETECTIONS']+':0')
    self.tf_input        = self.sess.graph.get_tensor_by_name(self.config_graph['TENSOR_INPUT']+':0')
    self.input_model_h   = self.config_graph['IMG_H']
    self.input_model_w   = self.config_graph['IMG_W']
    self.model_threshold = self.config_graph['MODEL_THRESHOLD']
    return
  
  def _sess_run(self, images):
    if len(images.shape) != 4:
      raise ValueError("Input must be BHWC, received {}".format(images.shape))
      
    if images.shape[1:3] != (self.input_model_h, self.input_model_w):
      raise ValueError("Preprocessed images for YOLO must be {}. Received {}".format(
          self.input_model_h, images.shape[1:3]))
    
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    out_scores, out_boxes, out_classes, out_detections = self.sess.run(
        [self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_detections],
        feed_dict={self.tf_input: images},
        options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return out_scores, out_boxes, out_classes, out_detections
  
  def _postprocess_boxes(self, boxes, idx_image=0):
    tn = self.timer_name(name=ct.TIMER_POSTPROCESS_BOXES)
    self.log.start_timer(tn)
    img_shape = self.input_shape[idx_image]

    (top, left, bottom, right), (new_h, new_w) = self.log.center_image_coordinates(src_h=img_shape[0], 
                                                                                   src_w=img_shape[1], 
                                                                                   target_h=self.input_model_h, 
                                                                                   target_w=self.input_model_w)
    
    #[0:1] to [0:yolo_model_shape]
    boxes[:,0]*= self.input_model_h
    boxes[:,1]*= self.input_model_w
    boxes[:,2]*= self.input_model_h
    boxes[:,3]*= self.input_model_w

    #eliminate centering
    boxes[:,0] = boxes[:,0] - top
    boxes[:,1] = boxes[:,1] - left
    boxes[:,2] = boxes[:,2] - top
    boxes[:,3] = boxes[:,3] - left
    
    #translate to original image
    boxes[:,0] = boxes[:,0] / new_h * img_shape[0]
    boxes[:,1] = boxes[:,1] / new_w * img_shape[1]
    boxes[:,2] = boxes[:,2] / new_h * img_shape[0]
    boxes[:,3] = boxes[:,3] / new_w * img_shape[1]

    boxes = boxes.astype(np.int32)
    boxes[:, 0] = np.maximum(0, boxes[:, 0])
    boxes[:, 1] = np.maximum(0, boxes[:, 1])
    boxes[:, 2] = np.minimum(img_shape[0], boxes[:, 2])
    boxes[:, 3] = np.minimum(img_shape[1], boxes[:, 3])
    self.log.stop_timer(tn)
    return boxes
  
  def _postprocess_inference(self, scores, boxes, classes, numdet):
    batch_frames = [] 
    nr_frames = len(scores)
    for _frame in range(nr_frames):
      frame_data = []
      frame_scores = scores[_frame]
      frame_boxes = self._postprocess_boxes(boxes[_frame], _frame)
      frame_classes = classes[_frame].astype(int)
      for _id in range(frame_classes.shape[0]):
        if frame_scores[_id] >= self.model_threshold:
          frame_data.append({
                "TLBR_POS":np.around(frame_boxes[_id]).tolist(), # [TOP, LEFT, BOTTOM, RIGHT]
                "PROB_PRC":np.around(frame_scores[_id]).item(),
                "TYPE": 'face'
              })
      #end frame iter      
      batch_frames.append(frame_data)
    return batch_frames
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    score, boxes, classes, num_det = self._sess_run(images)
    batch_frames = self._postprocess_inference(score, boxes, classes, num_det)
    self.log.stop_timer(timer_name)
    return batch_frames
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(name=ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    images = np.array([self.log.center_image2(x, 
                                              target_h=self.input_model_h, 
                                              target_w=self.input_model_w)
                         for x in images])   
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      self.input_shape = [x.shape for x in np_imgs]
      _prep_images = self._preprocess_images(np_imgs)  
      result = self._run_inference(_prep_images)
    #endif
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result
 
  def get_input_tensors(self):
    return [self.tf_input]    
  
  def get_output_tensors(self):
    return [
      self.tf_classes,
      self.tf_scores,
      self.tf_boxes,
      self.tf_detections
      ]


class CovidMaskInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'COVID_MASK')
    
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.sess = tf.Session(graph=self.graph)
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(self.tf_output, feed_dict={self.tf_input: images}, options=self.tf_runoptions)
    probs = probs.ravel().tolist()
    self.log.stop_timer(timer_name)
    return probs
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [self.log.center_image2(x, target_h=resize_h, target_w=resize_w) for x in images]
    images = np.array(images)
    images = images / 255.
    self.log.stop_timer(timer_name)
    return images
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  def get_input_tensors(self):
    return [self.tf_input]    
  
  def get_output_tensors(self):
    return [self.tf_output]


class CameraQualityCheckInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'CAMERA_QUALITY_CHECK')
    
    super().__init__(**kwargs)
    self._load_classes()
    self._load_graph()
    # self.predict_dummy()
    return
    
  def _load_classes(self):
    path_classes = self.config_graph['CLASSES']
    path_classes = os.path.join(self.log.get_models_folder(), path_classes)
    with open(path_classes) as f:
      self.classes = f.read().splitlines()
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
        
    iqa_memory_prc = self.config_graph['MEMORY_FRACTION']
    self.sess = self.get_session(
      graph=self.graph,
      mem_fraction=iqa_memory_prc,
      )
    
    dct_config = eval(open(path_pb + '.txt', 'r').read())
    self.tf_input = self.graph.get_tensor_by_name(dct_config['INPUT_0'])
    self.tf_output = self.graph.get_tensor_by_name(dct_config['OUTPUT_0'])
    return    
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    probs = self.sess.run(
            [self.tf_output],
            feed_dict={self.tf_input: images},
            options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return probs
  
  def _postprocess_inference(self, preds):
    lst = []
    for pred in preds:
      lst.append([{
        'TYPE': self.classes,
        'PROB_PRC': pred.tolist()
        }])
    return lst
  
  def _run_inference(self, images):
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    if isinstance(images, (np.ndarray)) and len(images.shape) == 3:
      images = np.expand_dims(images, axis=0)
          
    preds = self._sess_run(images)[0]
    preds = self._postprocess_inference(preds)
    self.log.stop_timer(timer_name)
    return preds

  def _preprocess_images(self, images):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    resize_h = self.config_graph['IMG_H']
    resize_w = self.config_graph['IMG_W']
    images = [x[:,:,::-1] for x in images] #model expects BGR format
    images = [cv2.resize(img, (resize_w, resize_h)) for img in images]    
    images = np.array(images)  
    self.log.stop_timer(timer_name)
    return images

  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()    

    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      images = self._preprocess_images(np_imgs)
      result = self._run_inference(images)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  def get_input_tensors(self):
    return [self.tf_input]    
  
  def get_output_tensors(self):
    return [self.tf_output]


class EffDetInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._batch_size = kwargs.get('batch_size', self.config_graph['BATCH_SIZE'])
    self.dct_operation_per_image = {
        ct.TIMER_PREPROCESS_IMAGES: False,
        ct.TIMER_SESSION_RUN:       False,
        ct.TIMER_POSTPROCESS_BOXES: True
      }
    self._load_graph()
    self._load_coco_classes()
    # self.predict_dummy()
    return
  
  def _load_coco_classes(self):
    self.classes = {
        1 : 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6 : 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
        39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
        43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
        54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
        59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
        64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
        73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
        78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
        84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush',
    }
    dct_rename = self.config_graph.get('RENAME_CLASS', {})
    for k,v in dct_rename.items():
      for k_orig,v_orig in self.classes.items():
        if v_orig == k:
          self.classes[k_orig] = v
    #endfor
    
    self.log.p('Setting probabilities')
    model_thr = self.config_graph['MODEL_THRESHOLD']
    dct_class_thr = self.config_graph.get('CLASS_THRESHOLD', {})
    self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr for c in self.classes}
    return
  
  def _load_pb_graph(self):
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    eff_det_memory_prc = self.config_graph.get('MEMORY_FRACTION', None)
    self.sess = self.get_session(
      graph=self.graph,
      mem_fraction=eff_det_memory_prc
      )
    
    self.tf_input_images = self.graph.get_tensor_by_name(self.config_graph['INPUT_0_IMAGES'])
    self.tf_input_min_score = self.graph.get_tensor_by_name(self.config_graph['INPUT_1_MIN_SCORE'])
    self.tf_input_iou = self.graph.get_tensor_by_name(self.config_graph['INPUT_2_IOU'])
    self.tf_output = self.graph.get_tensor_by_name(self.config_graph['OUTPUT_0'])
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    out = self.sess.run(
            self.tf_output,
            feed_dict={self.tf_input_images: images, self.tf_input_min_score: self.config_graph['MODEL_THRESHOLD'], self.tf_input_iou: self.config_graph['IOU_THRESHOLD']},
            options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return out
  
  def _postprocess_boxes(self, boxes, idx_image=0):
    tn = self.timer_name(name=ct.TIMER_POSTPROCESS_BOXES)
    self.log.start_timer(tn)
    img_shape = self.input_shape[idx_image]
    if self.center_image:
      (top, left, bottom, right), (new_h, new_w) = self.log.center_image_coordinates(src_h=img_shape[0], 
                                                                                     src_w=img_shape[1], 
                                                                                     target_h=self.resize_shape[0], 
                                                                                     target_w=self.resize_shape[1])
      #eliminate centering
      boxes[:,0] = boxes[:,0] - top
      boxes[:,1] = boxes[:,1] - left
      boxes[:,2] = boxes[:,2] - top
      boxes[:,3] = boxes[:,3] - left
      
      #translate to original image
      boxes[:,0] = boxes[:,0] / new_h * img_shape[0]
      boxes[:,1] = boxes[:,1] / new_w * img_shape[1]
      boxes[:,2] = boxes[:,2] / new_h * img_shape[0]
      boxes[:,3] = boxes[:,3] / new_w * img_shape[1]
    #endif
    boxes = boxes.astype(np.int32)
    boxes[:, 0] = np.maximum(0, boxes[:, 0])
    boxes[:, 1] = np.maximum(0, boxes[:, 1])
    boxes[:, 2] = np.minimum(img_shape[0], boxes[:, 2])
    boxes[:, 3] = np.minimum(img_shape[1], boxes[:, 3])
    self.log.stop_timer(tn)
    return boxes

  def _postprocess_inference(self, preds):
    lst_all = []
    # arr_idx_img = preds[:, :, 0].astype(np.int32)
    arr_tlbr = preds[:, :, 1:5].astype(np.int32)
    arr_score = preds[:, :, 5]
    arr_class = preds[:, :, 6].astype(np.int32)
    nr_imgs = preds.shape[0]
    for nr in range(nr_imgs):
      lst = []
      sel_idx = np.argwhere((arr_class[nr] >= 1) & (arr_score[nr] >= 0)).ravel()
      boxes = arr_tlbr[nr]
      boxes = self._postprocess_boxes(boxes, idx_image=nr)
      for x in sel_idx:
        idx_class = arr_class[nr][x]
        lbl = self.classes[idx_class]
        score = arr_score[nr][x]
        _boxes = boxes[x].tolist()
        if score >= self.probas[idx_class]:
          lst.append({
            'TLBR_POS': _boxes,
            'PROB_PRC': round(float(score), 3),
            'TYPE'    : lbl
            })
        #endif
      lst_all.append(lst)
    #endfor
    return lst_all

  def _run_inference(self, images):
    assert images is not None and type(images) == np.ndarray and len(images.shape) == 4
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    preds = self._postprocess_inference(preds)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, np_imgs):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(np_imgs, (np.ndarray)) and len(np_imgs.shape) == 3:
      np_imgs = np.expand_dims(np_imgs, axis=0)
    lst_shape = [x.shape for x in np_imgs]
    self.input_shape = lst_shape
    if len(set(lst_shape)) > 1:
      batch_strategy = self.config_graph['BATCH_STRATEGY']
      self.center_image = True
      if batch_strategy == ct.BATCH_STRATEGY_MOST_COMMON_SHAPE:        
        unique, counts = np.unique(self.input_shape, return_counts=True, axis=0)        
        self.resize_shape = tuple(unique[np.argmax(counts)])
        res_h, res_w, _ = self.resize_shape
#        print('Resizing to: {}'.format(self.resize_shape))
        lst_imgs = [self.log.center_image2(x, res_h, res_w) if x.shape != self.resize_shape else x for x in np_imgs]
        np_imgs = np.array(lst_imgs)
      elif batch_strategy == ct.BATCH_STRATEGY_PER_SHAPE:
        np_imgs = np.array(lst_imgs)
      #endif
    else:
      if type(np_imgs) is list:
        np_imgs = np.array(np_imgs)
#    print('Preprocessed tensor shape: {}'.format(np_imgs.shape))
    self.log.stop_timer(timer_name)
    return np_imgs
  
  def predict_dummy(self):
    try:
      np_imgs = np.random.uniform(size=(self._batch_size, 720, 1280, 3))
      self.P('Making dummy predictions on tensor with shape: {}'.format(np_imgs.shape), color='y')
      self.predict(np_imgs)
      self.P('Done making dummy predictions', t=True, color='y')
    except Exception as e:
      self.P('Exception on dummy inference: {}'.format(str(e)), color='r')
    return
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    self.center_image = False
    self.input_shape = None
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
    
    if len(result) != len(np_imgs):      
      self.log.p('* You are using {}'.format(self.config_graph['GRAPH']))
      self.log.p('* You have a batch size of {}'.format(self.config_graph['BATCH_SIZE']))
      self.log.p('* You want to infer {} images'.format(len(np_imgs)))
      self.log.p('* You only got {} results'.format(len(result)))
      raise ValueError('Please make sure that you are using the graph with the proper batch size')
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  def get_input_names(self):
    return [self.config_graph['INPUT_0']]
    
  def get_output_names(self):
    return [self.config_graph['OUTPUT_0']]
  
  def get_input_tensors(self):
    return [self.tf_input_images, self.tf_input_min_score, self.tf_input_iou]
  
  def get_output_tensors(self):
    return [self.tf_output]
  
  
class EffDet0InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET0')
    super().__init__(**kwargs)
    return
  

class EffDet2640x1132BS4InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET2_640x1132_BS4')
    super().__init__(**kwargs)
    return


class EffDet3768x1358BS1InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET3_768x1358_BS1')
    super().__init__(**kwargs)
    return
  

class EffDet3768x1358BS7InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET3_768x1358_BS7')
    super().__init__(**kwargs)
    return


class EffDet4InferenceGraph(EffDetInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'EFF_DET4')
    super().__init__(**kwargs)
    return


class StyleTransferInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'STYLE_TRANSFER')
    super().__init__(**kwargs)
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_pb_graph(self):
    import tensorflow as tf
    graph_name = self.config_graph['GRAPH']
    folder_name = os.path.dirname(graph_name)
    
    full_path = self.log.get_models_subfolder(folder_name)
    self.graph = tf.saved_model.load(full_path)
    
    assert self.graph is not None, 'Graph not found!'
    
    style_image = self.config_graph['STYLE_IMAGE']
    style_image_path = self.log.get_models_file(style_image)
    
    self.style_image = cv2.imread(style_image_path)
    self.style_image = cv2.resize(self.style_image, (256, 256))
    self.style_image = self.style_image[:,:,::-1]
    self.style_image = self._preprocess_image(self.style_image)
    self.style_image = np.expand_dims(self.style_image, axis=0)
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    #TODO: Rares, we should check inference on multiple images and see if style image tensor should be of equal size with images tensor (on batch dim)
    out = self.graph(tf.constant(images), tf.constant(self.style_image))
    self.log.stop_timer(timer_name)
    return out

  def _postprocess_inference(self, preds):
    #extract stylized images
    stylized_images_rgb = preds[0]
    
    l = []
    for img in stylized_images_rgb:
    #postprocess stylized image
      stylized_image_8bit_rgb = (img.numpy() * 255).astype('uint8')
      result = [{'IMG': stylized_image_8bit_rgb}]
      l.append(result)
    return l

  def _run_inference(self, images):
    assert images is not None and type(images) == np.ndarray and len(images.shape) == 4
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    preds = self._sess_run(images)
    preds = self._postprocess_inference(preds)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_image(self, img):
    img = (img / 255).astype(np.float32)
    return img
  
  def _preprocess_images(self, np_imgs):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(np_imgs, (np.ndarray)) and len(np_imgs.shape) == 3:
      np_imgs = np.expand_dims(np_imgs, axis=0)
      
    l = []
    #TODO: what happens if images are of different shape
    #TODO: Rares, we should revisit this one
    for img in np_imgs:
      img = self._preprocess_image(img)
      l.append(img)
    np_imgs = np.array(l)
    self.log.stop_timer(timer_name)
    return np_imgs
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
    
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result

  def get_input_tensors(self):
    return []    
  
  def get_output_tensors(self):
    return []


class TFOdapiInferenceGraph(BaseInferenceGraph):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.dct_operation_per_image = {
        ct.TIMER_PREPROCESS_IMAGES: False,
        ct.TIMER_SESSION_RUN:       False,
        ct.TIMER_POSTPROCESS_BOXES: True
      }
    self._load_classes()
    self._load_graph()
    # self.predict_dummy()
    return
  
  def _load_classes(self):
    cls_file = self.config_graph['CLASSES']
    full_cls_file = os.path.join(self.log.get_models_folder(), cls_file)
    self.log.p('Loading {}...'.format(full_cls_file))
    with open(full_cls_file) as f:
      lines = f.read().splitlines()
    self.orig_classes = lines.copy() + [lines[-1]]*150 # pad last class...
    #rename classes
    self.classes = self.orig_classes.copy()
    dct_rename = self.config_graph.get('RENAME_CLASS', {})
    for k,v in dct_rename.items():
      idx = self.classes.index(k)
      self.classes[idx] = v
    #endfor
    self.log.p('Loaded {} classes from {}'.format(len(lines), cls_file))
    
    self.log.p('Setting probabilities')
    model_thr = self.config_graph['MODEL_THRESHOLD']
    dct_class_thr = self.config_graph.get('CLASS_THRESHOLD', {})
    self.probas = {c: dct_class_thr[c] if c in dct_class_thr else model_thr for c in self.classes}
    self.log.p('Done setting probabilities')
    return

  def _load_pb_graph(self):
    tn = self.timer_name(ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(tn)
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      self.graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      self.graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert self.graph is not None, 'Graph not found!'
    
    self.classes_tensor_name = self.config_graph["CLASSES_TENSOR_NAME"]
    self.scores_tensor_name = self.config_graph["SCORES_TENSOR_NAME"]
    self.boxes_tensor_name = self.config_graph["BOXES_TENSOR_NAME"]
    self.input_tensor_name = self.config_graph["INPUT_TENSOR_NAME"]
    self.numdet_tensor_name = self.config_graph["NUMDET_TENSOR_NAME"]
    
    self.sess = tf.Session(graph=self.graph)
    self.tf_classes = self.sess.graph.get_tensor_by_name(self.classes_tensor_name+":0")
    self.tf_scores = self.sess.graph.get_tensor_by_name(self.scores_tensor_name+":0")
    self.tf_boxes = self.sess.graph.get_tensor_by_name(self.boxes_tensor_name+":0")
    self.tf_numdet = self.sess.graph.get_tensor_by_name(self.numdet_tensor_name+":0")
    self.tf_input = self.sess.graph.get_tensor_by_name(self.input_tensor_name+":0")
    self.log.stop_timer(tn)
    self.log.p('Graph loaded in {}'.format(self.log.get_timer(tn)))
    return
  
  def _sess_run(self, images):
    timer_name = self.timer_name(name=ct.TIMER_SESSION_RUN)
    self.log.start_timer(timer_name)
    np_scores, np_boxes, np_classes, np_numdet = self.sess.run(
        [self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_numdet],
        feed_dict={self.tf_input: images},
        options=self.tf_runoptions)
    self.log.stop_timer(timer_name)
    return np_scores, np_boxes, np_classes, np_numdet
  
  def _postprocess_boxes(self, boxes, idx_image=0):
    tn = self.timer_name(name=ct.TIMER_POSTPROCESS_BOXES)
    self.log.start_timer(tn)
    img_shape = self.input_shape[idx_image]
    if self.center_image:
      (top, left, bottom, right), (new_h, new_w) = self.log.center_image_coordinates(src_h=img_shape[0], 
                                                                                     src_w=img_shape[1], 
                                                                                     target_h=self.resize_shape[0], 
                                                                                     target_w=self.resize_shape[1])
      #[0:1] to [0:yolo_model_shape]
      boxes[:,0] = boxes[:,0] * self.resize_shape[0]
      boxes[:,1] = boxes[:,1] * self.resize_shape[1]
      boxes[:,2] = boxes[:,2] * self.resize_shape[0]
      boxes[:,3] = boxes[:,3] * self.resize_shape[1]
      
      #eliminate centering
      boxes[:,0] = boxes[:,0] - top
      boxes[:,1] = boxes[:,1] - left
      boxes[:,2] = boxes[:,2] - top
      boxes[:,3] = boxes[:,3] - left
      
      #translate to original image
      boxes[:,0] = boxes[:,0] / new_h * img_shape[0]
      boxes[:,1] = boxes[:,1] / new_w * img_shape[1]
      boxes[:,2] = boxes[:,2] / new_h * img_shape[0]
      boxes[:,3] = boxes[:,3] / new_w * img_shape[1]
    else:
      boxes[:,0] = boxes[:,0] * img_shape[0]
      boxes[:,1] = boxes[:,1] * img_shape[1]
      boxes[:,2] = boxes[:,2] * img_shape[0]
      boxes[:,3] = boxes[:,3] * img_shape[1]
    #endif
    boxes = boxes.astype(np.int32)
    boxes[:, 0] = np.maximum(0, boxes[:, 0])
    boxes[:, 1] = np.maximum(0, boxes[:, 1])
    boxes[:, 2] = np.minimum(img_shape[0], boxes[:, 2])
    boxes[:, 3] = np.minimum(img_shape[1], boxes[:, 3])
    self.log.stop_timer(tn)
    return boxes
  
  def _postprocess_inference(self, scores, boxes, classes):
    batch_frames = []
    nr_frames = len(scores)
    for nr_img in range(nr_frames):
      frame_data = []
      frame_scores = scores[nr_img]
      frame_boxes = boxes[nr_img]
      frame_boxes = self._postprocess_boxes(frame_boxes, idx_image=nr_img)
      frame_classes = classes[nr_img].astype(int)
      for _id in range(frame_classes.shape[0]):
        _type = self.classes[frame_classes[_id]]
        lst_exclude = self.config_graph.get('EXCLUDE_CLASS', [])
        if _type in lst_exclude:
          continue
        if frame_scores[_id] >= self.probas[_type]:
          frame_data.append({
                "TLBR_POS":np.around(frame_boxes[_id]).tolist(), # [TOP, LEFT, BOTTOM, RIGHT]
                "PROB_PRC":np.around(frame_scores[_id]).item(),
                "TYPE": _type
              })
      #end frame iter      
      batch_frames.append(frame_data)
    return batch_frames
  
  def _run_inference(self, images):
    assert images is not None and type(images) == np.ndarray and len(images.shape) == 4
    timer_name = self.timer_name(ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(timer_name)
    scores, boxes, classes, _ = self._sess_run(images)
    preds = self._postprocess_inference(scores, boxes, classes)
    self.log.stop_timer(timer_name)
    return preds
  
  def _preprocess_images(self, np_imgs):
    timer_name = self.timer_name(ct.TIMER_PREPROCESS_IMAGES)
    self.log.start_timer(timer_name)
    if isinstance(np_imgs, (np.ndarray)) and len(np_imgs.shape) == 3:
      np_imgs = np.expand_dims(np_imgs, axis=0)
    
    lst_shape = [x.shape for x in np_imgs]
    self.input_shape = lst_shape
    if len(set(lst_shape)) > 1:
      batch_strategy = self.config_graph['BATCH_STRATEGY']
      self.center_image = True
      if batch_strategy == ct.BATCH_STRATEGY_MOST_COMMON_SHAPE:
        unique, counts = np.unique(self.input_shape, return_counts=True, axis=0)
        self.resize_shape = tuple(unique[np.argmax(counts)])
        res_h, res_w, _ = self.resize_shape
        lst_imgs = [self.log.center_image(x, res_h, res_w) if x.shape != self.resize_shape else x for x in np_imgs]
        np_imgs = np.array(lst_imgs)
      elif batch_strategy == ct.BATCH_STRATEGY_PER_SHAPE:
        np_imgs = np.array(lst_imgs)
      #endif
    else:
      if type(np_imgs) is list:
        np_imgs = np.array(np_imgs)
    self.log.stop_timer(timer_name)
    return np_imgs
  
  def _filter(self, preds):
    dct_filter = self.config_graph.get('FILTER', {})
    if dct_filter:
      lst_exclude = dct_filter.get('EXCLUDE', [])
      if lst_exclude:
        filtered_preds = []
        for lst in preds:
          lst = [x for x in lst if x['TYPE'] not in lst_exclude]
          filtered_preds.append(lst)
        preds = filtered_preds
      
      lst_keep = dct_filter.get('KEEP', [])
      if lst_keep:
        filtered_preds = []
        for lst in preds:
          lst = [x for x in lst if x['TYPE'] in lst_keep]
          filtered_preds.append(lst)
        preds = filtered_preds
    return preds
  
  def predict(self, np_imgs):
    timer_name = self.timer_name()
    self.log.start_timer(timer_name)
    self.center_image = False
    self.input_shape = None
    timestamp = self.log.now_str()

    dct_result = OrderedDict()
    dct_meta = OrderedDict()
    
    if np_imgs is None or len(np_imgs) == 0:
      dct_meta['ERROR'] = 'No images received for inference'
      result = []
    else:
      np_imgs = self._preprocess_images(np_imgs)
      result = self._run_inference(np_imgs)
      result = self._filter(result)
      
    dct_meta['SYSTEM_TIME'] = timestamp
    dct_meta['VER'] = self.__version__
    dct_result['METADATA'] = dct_meta
    dct_result['INFERENCES'] = result
    self.log.stop_timer(timer_name)
    return dct_result
  
  def get_input_tensors(self):        
    return [self.tf_input]
  
  def get_output_tensors(self):
    return [
      self.tf_classes,
      self.tf_scores,
      self.tf_boxes,
      self.tf_numdet
      ]
  

class TFOdapi1InferenceGraph(TFOdapiInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI1')
    super().__init__(**kwargs)
    return

  def _load_pb_graph(self):
    tn = self.timer_name(ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(tn)
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert graph is not None, 'Graph not found!'
    
    self.classes_tensor_name = self.config_graph["CLASSES_TENSOR_NAME"]
    self.scores_tensor_name = self.config_graph["SCORES_TENSOR_NAME"]
    self.boxes_tensor_name = self.config_graph["BOXES_TENSOR_NAME"]
    self.input_tensor_name = self.config_graph["INPUT_TENSOR_NAME"]
    self.numdet_tensor_name = self.config_graph["NUMDET_TENSOR_NAME"]
    
    self.sess = tf.Session(graph=graph)
    self.tf_classes = self.sess.graph.get_tensor_by_name(self.classes_tensor_name+":0")
    self.tf_scores = self.sess.graph.get_tensor_by_name(self.scores_tensor_name+":0")
    self.tf_boxes = self.sess.graph.get_tensor_by_name(self.boxes_tensor_name+":0")
    self.tf_numdet = self.sess.graph.get_tensor_by_name(self.numdet_tensor_name+":0")
    self.tf_input = self.sess.graph.get_tensor_by_name(self.input_tensor_name+":0")
    self.log.stop_timer(tn)
    self.log.p('Graph loaded in {}'.format(self.log.get_timer(tn)))
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
    

class TFOdapi2InferenceGraph(TFOdapiInferenceGraph):
  def __init__(self, **kwargs):
    if not kwargs.get('config_graph', None) and not kwargs.get('config_key', None):
      kwargs.setdefault('config_key', 'TF_ODAPI2')
    super().__init__(**kwargs)
    return
  
  def _load_pb_graph(self):
    tn = self.timer_name(ct.TIMER_LOAD_GRAPH)
    self.log.start_timer(tn)
    path_graph = self.config_graph['PATH_GRAPH']
    graph_name = self.config_graph['GRAPH']
    path_pb    = None
    if path_graph and os.path.exists(path_graph):
      path_pb = os.path.join(path_graph, graph_name)
      self.log.p('Loading graph from specific path: {}'.format(path_pb))
      graph = self.log.load_tf_graph(path_pb)
    else:
      self.log.p('Loading graph from models: {}'.format(graph_name))
      path_pb = os.path.join(self.log.get_models_folder(), graph_name)
      graph = TFUtils.load_graph_from_models(
        log=self.log,
        model_name=graph_name
        )
    #endif
    assert graph is not None, 'Graph not found!'
    
    self.classes_tensor_name = self.config_graph["CLASSES_TENSOR_NAME"]
    self.scores_tensor_name = self.config_graph["SCORES_TENSOR_NAME"]
    self.boxes_tensor_name = self.config_graph["BOXES_TENSOR_NAME"]
    self.input_tensor_name = self.config_graph["INPUT_TENSOR_NAME"]
    self.numdet_tensor_name = self.config_graph["NUMDET_TENSOR_NAME"]
    
    self.sess = tf.Session(graph=graph)
    self.tf_classes = self.sess.graph.get_tensor_by_name(self.classes_tensor_name+":0")
    self.tf_scores = self.sess.graph.get_tensor_by_name(self.scores_tensor_name+":0")
    self.tf_boxes = self.sess.graph.get_tensor_by_name(self.boxes_tensor_name+":0")
    self.tf_numdet = self.sess.graph.get_tensor_by_name(self.numdet_tensor_name+":0")
    self.tf_input = self.sess.graph.get_tensor_by_name(self.input_tensor_name+":0")
    self.log.stop_timer(tn)
    self.log.p('Graph loaded in {}'.format(self.log.get_timer(tn)))
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


class InferenceGraphsEnum(Enum):
  ALIMENTARE_NEPERMISA            = AlimentareNepermisaInferenceGraph
  CAMERA_QUALITY_CHECK            = CameraQualityCheckInferenceGraph
  COVID_MASK                      = CovidMaskInferenceGraph
  EFF_DET0                        = EffDet0InferenceGraph
  EFF_DET2_640x1132_BS4           = EffDet2640x1132BS4InferenceGraph
  EFF_DET3_768x1358_BS1           = EffDet3768x1358BS1InferenceGraph
  EFF_DET3_768x1358_BS7           = EffDet3768x1358BS7InferenceGraph
  EFF_DET4                        = EffDet4InferenceGraph
  FACE_DETECTION                  = FaceInferenceGraph
  FIRE_SMOKE                      = FireSmokeInferenceGraph
  GASTRO                          = FoodEmptyInferenceGraph
  LP_DETECTION                    = LPDetectionInferenceGraph
  LPDV2                           = LPDv2InferenceGraph
  LPR                             = LPRInferenceGraph
  MERCHANDISER                    = MerchandiserInferenceGraph
  OMV_EMPLOYEE                    = OmvEmployeeInferenceGraph
  STYLE_TRANSFER                  = StyleTransferInferenceGraph
  TF_YOLO                         = YoloInferenceGraph
  TF_ODAPI1                       = TFOdapi1InferenceGraph
  TF_ODAPI1_SSD_MOBILENETV2       = TFOdapi1SSDMobilenetv2InferenceGraph
  TF_ODAPI_FRC_NAS                = TFOdapi1FcnNasInferenceGraph
  TF_ODAPI1_OIDV4                 = TFOdapi1OIDv4InferenceGraph
  TF_ODAPI1_TRAFFICSIGNS          = TFOdapi1TrafficSignsInferenceGraph
  TF_ODAPI2                       = TFOdapi2InferenceGraph
  TF_ODAPI2_EFFD0                 = TFOdapi2EffD0InferenceGraph
  TF_ODAPI2_EFFD7                 = TFOdapi2EffD7InferenceGraph
  TF_ODAPI2_SSDMBNV2_FPN_640x640  = TFOdapi2SSDMobilenetv2FPN640x640InferenceGraph
  TF_ODAPI2_SSDMBNV2_320x320      = TFOdapi2SSDMobilenetv2320x320InferenceGraph
  TF_ODAPI2_CENTERNET_FPN512x512  = TFOdapi2CenterNetFPN512x512ferenceGraph
  
  
  
class InferenceGraphBuilder(DecentrAIObject):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def build(self, graph, config_graph=None, config_path=None):
    if isinstance(graph, Enum):
      assert graph in InferenceGraphsEnum
    elif isinstance(graph, str):
      for elem in InferenceGraphsEnum:
        if elem.name == graph:
          graph = elem
          break
      assert graph in InferenceGraphsEnum
    #endif
    
    assert config_graph or config_path, 'Either provide a config_graph or a config_path and a config_key'

    if config_graph:
      ig = graph.value(log=self.log, config_graph=config_graph)
    else:
      ig = graph.value(log=self.log, config_path=config_path, config_key=graph.name)
    return ig

if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(lib_name='VP', config_file=cfg_file)
  
  # iqa = CameraQualityCheckInferenceGraph(
  #   log=log,
  #   config_path='decentrai_inference/inference.txt'
  #   )
  
  # iqa.predict(np.random.randint(
  #   low=0,
  #   high=255,
  #   size=(5, 1080, 1920, 3),
  #   dtype=np.uint8
  #   ))
  
  
  
  import matplotlib.pylab as plt
  style = StyleTransferInferenceGraph(
    log=log,
    config_path='decentrai_inference/inference.txt'
    )
  
  path = log.get_output_file('vlcsnap-2021-07-02-13h26m43s659.png')
  img = cv2.imread(path)
  img = img[:,:,::-1]
  dct = style.predict(img)
  lst_inf = dct['INFERENCES']  
  img = lst_inf[0]
    
  plt.imshow(img)
  plt.show()
