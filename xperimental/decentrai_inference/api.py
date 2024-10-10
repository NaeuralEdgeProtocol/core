#global dependencies
import numpy as np

from copy import deepcopy
from collections import OrderedDict

#local dependencies
from naeural_core import DecentrAIObject
from decentrai_inference import constants as ct
from naeural_core.serving.serving_manager import ServingManager

__version__ = '1.0.4.2'

class InferenceApi(DecentrAIObject):
  """
  This class purpose is provide a single entry point to all the inference capabilities (graphs and handlers).
  """
  
  __instance = None
  
  def __init__(self, config, **kwargs):
    self.__version__ = __version__
    self.config = config
    super().__init__(**kwargs)
    return
  
  def startup(self):
    self.inf_engine = ServingManager(
      log=self.log,
      config_fn='inference/model_testing/simple_tf_config.txt',
      server_names=[],    
      prefix_log='[TFMGR]',
      no_parallel=False
      ) 
    self.inf_handlers = {}
    super().startup()
    return
  
  def _prepare_inference_payload(self, captures_data):
    """
    This method iterates over the current active streams, extracts images (full or zone) and groups them by inference executors

    Parameters
    ----------
    captures_data : Dictionary
      This method receives the active streams (as dict) and their data (as dict).

    Returns
    -------
    dct_inf_payload : Dictionary
      This method outputs a dictionary containing inference executors as keys and list of dictionaries with images (together with image description - Stream, Location).
    
    dct_inf_payload = {
      'MA': [
          {
          'STREAM'        : STREAM1, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s1
          },
          {
          'STREAM'        : STREAM3, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s3
          }
        ],
      'MB': [
          {
          'STREAM'        : STREAM1, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s1
          },
          {
          'STREAM'        : STREAM2, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s2
          },
          {
          'STREAM'        : STREAM3, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s3
          },
          {
          'STREAM'        : STREAM4, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s4
          },
          {
          'STREAM'        : STREAM5, 
          'LOCATION_NAME' : None, 
          'PLUGINS'       : ['PersonQueue'],
          'IMAGE'         : img_s5
          }
        ]
      }
    """
    
    self.log.start_timer(ct.TIMER_PREPARE_PAYLOAD)
    dct_app_streams = deepcopy(self.config[ct.CONFIG_STREAMS])
    dct_app_plugins = deepcopy(self.config[ct.CONFIG_PLUGINS])
    
    dct_inf_payload = {}
    for stream_name, dct_stream in dct_app_streams.items():
      if stream_name not in captures_data or captures_data[stream_name] is None:
        continue #this means that the stream has stopped
      
      dct_inf = {}
      for plugin in dct_stream[ct.PLUGINS]:
        plugin_name = plugin[ct.SIGNATURE]
        if plugin_name not in dct_app_plugins:
          continue # we may have a disabled plugin due to lack of config data        
        app_feat = dct_app_plugins[plugin_name]                
        
        for plugin_loc in plugin[ct.LOCATIONS]:
          lst_inf_graphs = plugin_loc.get(ct.INFERENCE_GRAPHS, app_feat.get(ct.INFERENCE_GRAPHS))
          inf_metagraph = plugin_loc.get(ct.INFERENCE_META_GRAPHS, app_feat.get(ct.INFERENCE_META_GRAPHS))
          lst_inf_on_full_img = plugin_loc.get(ct.INFERENCE_ON_FULL_IMAGE, app_feat.get(ct.INFERENCE_ON_FULL_IMAGE))
          
          #check if the plugin requires running inference graphs or it doesn't need inference (ex: Zone extractor / Blur Zone doesn't need inference)
          if lst_inf_graphs is None and inf_metagraph is None:
            continue
          #endif
          
          #add all executors to the current stream dct_inf
          lst_inf_execs = lst_inf_graphs + [inf_metagraph] if inf_metagraph else lst_inf_graphs
          for x in lst_inf_execs:
            if x not in dct_inf:
              dct_inf[x] = []
          
          #send the full image to the custom plugin inference handlers
          lst_inf_on_full_img+= [True] * (len(lst_inf_execs) - len(lst_inf_graphs))
          
          for inf_exec, inf_on_full_img in zip(lst_inf_execs, lst_inf_on_full_img):
            if inf_on_full_img:
              #if plugin needs inference on full image, ensure that you will do only one inference on full image per stream
              #ex: multiple feats per stream (like PeopleCounting + PersonQueue) need only one inference per image
              if not dct_inf[inf_exec]:
                dct_inf[inf_exec].append({
                    'STREAM'        : stream_name, 
                    'LOCATION_NAME' : None, 
                    'PLUGINS'       : [plugin['SIGNATURE']],
                    'IMAGE'         : captures_data[stream_name]
                    })
              else:
                #if stream already has another plugin that needs inference on full image, \
                #then add this new plugin to the list of plugins that will need inference (like PeopleCounting + PersonQueue)
                dct = [x for x in dct_inf[inf_exec] if \
                         x['STREAM'] == stream_name and\
                         x['LOCATION_NAME'] is None
                       ][0]
                dct['PLUGINS'] = list(set(dct['PLUGINS'] + [plugin['SIGNATURE']]))
              #endif
            else:
              #the plugin is design to use only portion of image (like Gastro or AlimentareNepermisa)
              #add each location to dct_inf in order to be used in inference
              loc_name = plugin_loc['LOCATION_NAME']
              top      = plugin_loc['TOP']
              left     = plugin_loc['LEFT']
              bottom   = plugin_loc['BOTTOM']
              right    = plugin_loc['RIGHT']
              dct_inf[inf_exec].append({
                  'STREAM'        : stream_name, 
                  'LOCATION_NAME' : loc_name, 
                  'PLUGINS'       : [plugin['SIGNATURE']],
                  'IMAGE'         : captures_data[stream_name][top:bottom, left:right, :]
                  })
              #endfor
            #endif
          #endfor
        #endfor
      
      #append current stream inference needs to inference dict
      for inf_exec, lst in dct_inf.items():
        if inf_exec not in dct_inf_payload:
          dct_inf_payload[inf_exec] = []
        for x in lst:
          dct_inf_payload[inf_exec].append(x)        
      #endfor
    #endfor
    
    # self.log.p('Inference description:')
    # self.log.p(' Inference executors ({}): {}'.format(len(dct_inf_payload.keys()), list(dct_inf_payload.keys())))
    # for k,lst in dct_inf_payload.items():
    #   self.log.p(' {}: {} streams'.format(k, len(lst)))
    
    self.log.stop_timer(ct.TIMER_PREPARE_PAYLOAD)
    # self.log.p('Total prepare_payload: {}'.format(self.log.get_timer(tn)))    
    return dct_inf_payload
  
  def _prioritize_inference(self, dct_inf_payload):
    #set execution order/priority. inference graphs first, handlers after
    lst_graphs = list(filter(lambda x: not x.endswith('HANDLER'), dct_inf_payload.keys()))
    lst_handlers = list(filter(lambda x: x.endswith('HANDLER'), dct_inf_payload.keys()))
    lst_ordered = lst_graphs + lst_handlers
    dct_inf_payload = OrderedDict({k: dct_inf_payload[k] for k in lst_ordered})
    return dct_inf_payload
  
  def _infer(self, dct_inference_payload):
    """
    This method receives the inference needs, makes inference for each graph/handler and adds inference results to the incoming structure
    The output of this method will contain all the data required for the plugins to execute their job
    """
    tn = self.timer_name(ct.TIMER_INFER)
    self.log.start_timer(tn)
    dct_inference = {}
    for inference_server, lst_streams_payload in dct_inference_payload.items():      
      np_imgs = [x['IMAGE'] for x in lst_streams_payload]
      dct_inf = self.predict(
        inference_server=inference_server, 
        data=np_imgs, 
        predict_batch=True
        )
      for idx, dct_pay in enumerate(lst_streams_payload):
        dct_pay['INFERENCE'] = dct_inf['INFERENCES'][idx]
      dct_inference[inference_server] = lst_streams_payload
    self.log.stop_timer(tn)
    return dct_inference
  
  def _prepare_inference_results(self, dct_inf):
    """
    This method transfers the "model" oriented payload resulted from the inference step into "stream" oriented template
    Basically, the resulting payload will contain a dictionary where each key is a stream and the value is a dictionary that contain Model, Location and the result of inference
    
    dct_res = {
      'STREAM1': {
          'MA': {
              None: inference_results,
              'LOCATION1': inference_results_loc1,
              'LOCATION2': inference_results_loc2
            },
          'MB': {
              None: inference_results
            }
        },
      'STREAM2': {
          'MB': {
              None: inference_results
            }         
        },
      'STREAM3': {
          'MA': {
              None: inference_results
            },
          'MB': {
              None: inference_results
            }
        },
      'STREAM4': {
          'MB': {
              None: inference_results
            }         
        },
      'STREAM5': {
          'MB': {
              None: inference_results
            }         
        }
      }
    """
    
    tn = self.timer_name(name=ct.TIMER_PREPARE_RESULTS)
    self.log.start_timer(tn)
    dct_res = {}
    for inf_executor, lst_inf_res in dct_inf.items():
      for crt_dct_inf in lst_inf_res:
        stream_name = crt_dct_inf['STREAM']
        location_name = crt_dct_inf['LOCATION_NAME']
        inf = crt_dct_inf['INFERENCE']
        if stream_name not in dct_res:
          dct_res[stream_name] = {}
        if inf_executor not in dct_res[stream_name]:
          dct_res[stream_name][inf_executor] = {}
        dct_res[stream_name][inf_executor][location_name] = inf
    self.log.stop_timer(tn)
    return dct_res
  
  def infer(self, captures_data):
    """
    This method makes inference on the images received from the active streams

    Parameters
    ----------
    captures_data : Dictionary
      Dictionary containing stream name and current data & metadata from that specific stream.

    Returns
    -------
    dct_res : Dictionary
      DESCRIPTION.
    """
    tn = self.timer_name(name=ct.TIMER_RUN_INFERENCE)
    self.log.start_timer(tn)
    dct_inf_payload = self._prepare_inference_payload(captures_data)
    dct_inf_payload = self._prioritize_inference(dct_inf_payload)
    dct_inf = self._infer(dct_inf_payload)
    dct_res = self._prepare_inference_results(dct_inf)
    self.log.stop_timer(tn)
    return dct_res
  
  def predict(self, inference_server, data, predict_batch=False, **kwargs):
    """
    This function calls `predict` or `predict_batch` method of a specific graph

    Parameters
    ----------
    inference_server : string
      Name of the ifnerence server that should be used for inference.
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
    lst_img = []
    if isinstance(data, dict):
      lst_img = list(data.values())
    elif isinstance(data, (list, np.ndarray)):
      lst_img = data
    else:
      raise ValueError('Expected dict, list or np.ndarray for inference data.')
    #endif
    if predict_batch:
      preds = self.inf_engine.predict(
        server_name=inference_server,
        inputs=lst_img
        )
    else:
      preds = self.inf_engine.predict(
        server_name=inference_server,
        inputs=lst_img
        )
    #endif
    
    if isinstance(data, dict):
      preds = preds['INFERENCES']
      preds = dict(zip(data.keys(), preds))
    #endif
    return preds
  
  def keep_active_servers(self, server_names):
    self.inf_engine.keep_active_servers(
      active_server_names=server_names
      )
    return
  
