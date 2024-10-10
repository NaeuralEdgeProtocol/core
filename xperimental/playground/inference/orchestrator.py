  # INFERENCE NECESITIES:
  # 1. We know about each stream the following information:
  #   1.1 What type of stream it is: "VideoFile", "VideoStream", "StructuredData"
  #   1.2 What functionalities needs for each "location"
  #   1.3 Each functionality has a specific behaviour: Stateless/Statefull
  #   1.4  
  

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from naeural_core import DecentrAIObject
from naeural_core import Logger
from playground.inference.inference_frameworks.inference_framework_eng import InferenceFrameworkEngine
from playground.inference.inference_models.inference_model_eng import InferenceModelEngine
from playground.inference.inference_jobs.inference_job_eng import InferenceJobEngine

class InferenceOrchestrator(DecentrAIObject):
  def __init__(self, log, **kwargs):
    super().__init__(log=log, **kwargs)
    return
  
  def startup(self):
    super().startup()
    
    self._framework_engine = InferenceFrameworkEngine(log=self.log)
    self._model_engine = InferenceModelEngine(
      inference_framework_engine=self._framework_engine,
      log=self.log
      )    
    self._job_engine = InferenceJobEngine(
      model_engine=self._model_engine, 
      log=self.log
      )
    return
  
  def start_models(self, models):
    for signature, config in models.items():
      self._model_engine.start_model(
        signature=signature,
        config=config
        )
    return
  
  def _read_config(self, dct_app_streams, dct_app_plugins):
    #1. create job needs
    dct_inf_needs = {}
    for stream_name, dct_stream in dct_app_streams.items():
      if stream_name not in captures_data or captures_data[stream_name] is None:
        continue #this means that the stream has stopped
      
      dct_inf = {}
      for feat in dct_stream['PLUGINS']:
        app_feat = dct_app_plugins[feat['SIGNATURE']]
        lst_inf_jobs = app_feat.get('INFERENCE_JOBS')
        lst_inf_on_full_img = app_feat.get('INFERENCE_ON_FULL_IMAGE', None)
        if not lst_inf_on_full_img:
          lst_inf_on_full_img = [True] * lst_inf_jobs
        
        #check if the plugin requires running inference graphs or it doesn't need inference (ex: Zone extractor / Blur Zone doesn't need inference)
        if lst_inf_jobs is None:
          continue
        
        #add all executors to the current stream dct_inf
        for x in lst_inf_jobs:
          if x not in dct_inf:
            dct_inf[x] = []
        
        for inf_job, inf_on_full_img in zip(lst_inf_jobs, lst_inf_on_full_img):
          if inf_on_full_img:
            #if plugin needs inference on full image, ensure that you will do only one inference on full image per stream
            #ex: multiple feats per stream (like PeopleCounting + PersonQueue) need only one inference per image
            if not dct_inf[inf_job]:
              dct_inf[inf_job].append({
                  'STREAM'        : stream_name, 
                  'LOCATION_NAME' : None, 
                  'PLUGINS'       : [feat['SIGNATURE']],
                  'DATA'          : captures_data[stream_name]
                  })
            else:
              #if stream already has another plugin that needs inference on full image, \
              #then add this new plugin to the list of plugins that will need inference (like PeopleCounting + PersonQueue)
              for dct in dct_inf[inf_job]:
                if dct['STREAM'] == stream_name and dct['LOCATION_NAME'] is None:
                  dct['PLUGINS'] = list(set(dct['PLUGINS'] + [feat['SIGNATURE']]))
                  break
          else:
            #the plugin is design to use only portion of image (like Gastro or AlimentareNepermisa)
            #add each location to dct_inf in order to be used in inference
            for loc in feat['LOCATIONS']:
              loc_name = loc['LOCATION_NAME']
              top      = loc['TOP']
              left     = loc['LEFT']
              bottom   = loc['BOTTOM']
              right    = loc['RIGHT']
              img      = captures_data[stream_name][top:bottom, left:right, :]
              dct_inf[inf_job].append({
                  'STREAM'        : stream_name, 
                  'LOCATION_NAME' : loc_name, 
                  'PLUGINS'       : [feat['SIGNATURE']],
                  'DATA'          : img
                  })
      #endfor
      
      #append current stream inference needs to inference dict
      for inf_job, lst in dct_inf.items():
        if inf_job not in dct_inf_needs:
          dct_inf_needs[inf_job] = []
        for x in lst:
          dct_inf_needs[inf_job].append(x)
    #endfor
    return dct_inf_needs
  
  def create_jobs_pipeline(self, dct_app_streams, dct_app_plugins):        
    #1. read config
    #TODO: need to separate _read_config into specific datatype (image, )
    #basically, now we have only Image "inference needs"
    inference_needs = self._read_config(dct_app_streams, dct_app_plugins)
    
    #2. create jobs pipelines
    lst_pipelines = []
    for job_name, lst_job_data in inference_needs.items():
      res = self._job_engine.create(
        signature=job_name,
        data=lst_job_data
        )
      if res:
        lst_pipelines.append(res)
    return lst_pipelines
  
  def execute_pipelines(self, pipelines):
    max_len = max(len(x) for x in pipelines)
    pipeline_execution = {i:[] for i in range(len(pipelines))}
    
    #TODO: how to implement (such as “dont run tensorflow before pytorch”) ???
    for nr_step in range(max_len):
      for nr_pipe, pipe in enumerate(pipelines):
        if len(pipe) > nr_step:   
          #execute only if pipe still has steps
          step        = pipe[nr_step]
          func_type   = step['TYPE']
          func        = step['EXECUTE']
          input_data  = step['INPUT_DATA']
          
          #check if step data comes from previous step
          if isinstance(input_data, str) and 'STEP_' in input_data:
            data_step = int(input_data.replace('STEP_', ''))
            print(nr_pipe, data_step)
            input_data = pipeline_execution[nr_pipe][data_step]
          
          res = func(input_data)
          pipeline_execution[nr_pipe].append(res)
          
    res = [x[-1] for _,x in pipeline_execution.items()]
    return res
  
if __name__ == '__main__':
  import numpy as np
  from app.config import Config
  cfg_file = 'main_config.txt'
  log = Logger(lib_name='DEVICE', config_file=cfg_file, max_lines=1000)
  config = Config.get_instance(log)
  
  MODELS = log.load_json('playground/inference/inference.txt')
    
  dct_app_streams, _ = config.read_streams_config()
  dct_app_plugins, _ = config.read_plugins_config()
  
  captures_data = {
    x['NAME']: np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
          for _,x in dct_app_streams.items()
    }
  

  # io = InferenceOrchestrator(log=log)
  # io.start_models(MODELS) #this should be called after inference necessities is determined
  # lst_jobs_pipeline = io.create_jobs_pipeline(dct_app_streams, dct_app_plugins)
  # res = io.execute_pipelines(lst_jobs_pipeline)
  
  
  
  
  
      
  
      
  
  