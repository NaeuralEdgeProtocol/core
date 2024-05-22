
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

#local dependencies
from core import constants as ct
from core.main.orchestrator import Orchestrator
from core import Logger

# TODO: change to `from PyE2 import`
from PyE2.utils import load_dotenv

MANDATORY_PACKAGES = {
  'torch'           : '2.0',
  'accelerate'      : '0.2',
  'transformers'    : '4.30',
  'tokenizers'      : '0.14',
}

def maybe_replace_txt(fn):
  result = fn
  if fn.endswith('.txt'):
    fn_dest = fn.replace('.txt', '.json')
    print("Found '{}' as base startup config file. Converting to '{}'...".format(fn, fn_dest), flush=True)
    # duplicate .txt to .json    
    shutil.copy(fn, fn_dest)
    # delete old .txt
    os.remove(fn)
    result = fn_dest
  return result
    
  
def get_id(log):
  config_box_id = log.config_data.get(ct.CONFIG_STARTUP_v2.K_EE_ID, '')
  log.P("Found EE_ID '{}'".format(config_box_id))
  if config_box_id.upper().replace('X','') == '':
    config_box_id_env = os.environ.get('EE_ID')
    log.P("Changing found default config id '{}'...".format(config_box_id))  
    if config_box_id_env is not None and config_box_id != config_box_id_env:
      log.P("E2 configured from env '{}'".format(config_box_id_env), color='m')
      if config_box_id_env == 'E2dkr': # same as in Dockerfile
        config_box_id_env += '-' + log.get_uid(size=4)
      config_box_id = config_box_id_env
    else:
      config_box_id = 'ee-' + log.get_uid(size=4)
      log.P("E2 is not manually configured nor from env. Assuming a random id '{}'".format(config_box_id), color='r')
    #endif no name
    log.config_data[ct.CONFIG_STARTUP_v2.K_EE_ID] = config_box_id
    log.P("  Saving/updating config with new EE_ID '{}'...".format(config_box_id))
    log.update_config_values({ct.CONFIG_STARTUP_v2.K_EE_ID: config_box_id})
  #endif config not ok
  return config_box_id
  
  
def get_config(config_fn):  
  fn = None
  extensions = ['.json', '.txt',]
  for loc in ['.', ct.LOCAL_CACHE]:
    for ext in extensions:
      test_fn = os.path.join(loc, config_fn + ext)
      if os.path.isfile(test_fn):
        fn = maybe_replace_txt(test_fn)
        break
  
  if fn is not None:
    print("Found '{}' as base startup config file.".format(fn), flush=True)
  else:
    fn = "{}.json".format(config_fn)
    print("No startup config file found in base folder", flush=True)
    os.makedirs(ct.LOCAL_CACHE, exist_ok=True)
    fn = os.path.join(ct.LOCAL_CACHE, fn)
    is_config_in_local_cache = os.path.isfile(fn)
    print("Using {}: {}".format(fn, is_config_in_local_cache), flush=True)
    config_string= os.environ.get('EE_CONFIG', './.{}.json'.format(config_fn)) # default to local .config_startup.json
    is_config_string_a_file = os.path.isfile(config_string)
    
    if is_config_in_local_cache:
      print("Found '{}' config file in local cache.".format(fn), flush=True)
    #endif local cache
    elif is_config_string_a_file:
      shutil.copy(config_string, fn)
      print("Using '{}' -> '{} as base startup config file.".format(config_string, fn), flush=True)
    #endif config string is a file
    elif len(config_string) > 3:
      # assume input is json config and we will overwrite the local cache even if it exists
      print("Attempting to process JSON '{}'...".format(config_string), flush=True)      
      config_data = json.loads(config_string)
      if isinstance(config_data, dict):
        with open(fn, 'w') as fh:
          json.dump(config_data, fh)
        print("Saved config JSON to {}".format(fn), flush=True)
      else:
        print("ERROR: EE_CONFIG '{}' is neither config file nor valid json data".format(config_string), flush=True)
        sys.exit(ct.CODE_CONFIG_ERROR)
    else:
      print("ERROR: EE_CONFIG '{}' cannot be used for startup configuration".format(config_string), flush=True)
      sys.exit(ct.CODE_CONFIG_ERROR)
      #endif cache or copy
    #endif JSON or file
  #endif default config exists or not
  return fn


def main():
  CONFIG_FILE = 'config_startup'
  is_docker = str(os.environ.get('AINODE_DOCKER')).lower() in ["yes", "true"]
  if not is_docker:
    load_dotenv()
      
  config_file = get_config(config_fn=CONFIG_FILE)

  l = Logger(
    lib_name='EE',
    base_folder='.',
    app_folder=ct.LOCAL_CACHE,
    config_file=config_file,
    max_lines=1000, 
    TF_KERAS=False
  )
  
  if l.no_folders_no_save:
    l.P("ERROR: local cache not properly configured. Note: This version is not able to use read-only systems...", color='r', boxed=True)
    sys.exit(ct.CODE_CONFIG_ERROR)
  #endif no folders
  
  if l.config_data is None or len(l.config_data) == 0:
    l.P("ERROR: config_startup.txt is not a valid json file", color='r', boxed=True)
    sys.exit(ct.CODE_CONFIG_ERROR)
  else:
    l.P("Running with config:\n{}".format(json.dumps(l.config_data, indent=4)), color='n')

  packs = l.get_packages(as_text=True, indent=4, mandatory=MANDATORY_PACKAGES)
  l.P("Current build installed packages:\n{}".format(packs))

  # DEBUG log environment
  l.P("Environment:")
  for k in os.environ:
    if k.startswith('EE_') or k.startswith('AIXP'):
      l.P("  {}={}".format(k, os.environ[k]))
  # DEBUG end log environment

  if is_docker:
    # post config setup
    docker_env = os.environ.get('AINODE_ENV')
    l.P("Docker base layer environment {}".format(docker_env))
    # test GPU overwrite
  #endif docker post config
  
  config_box_id = get_id(log=l)
  
  try:    
    lock = l.lock_process(config_box_id)

    l.P("Starting Execution Engine Main Loop...\n\n\n.", color=ct.COLORS.MAIN)
    eng = Orchestrator(log=l)
        
    if lock is None:
      msg = "Shutdown forced due to already started local processing node with id '{}'!".format(eng.cfg_eeid)
      eng.P(msg, color='error')
      return_code = eng.forced_shutdown()
    else:
      return_code = eng.main_loop()
      
    l.p('Execution Engine v{} main loop exits with return_code: {}'.format(
      eng.__version__, return_code), color=ct.COLORS.MAIN
    )
    l.maybe_unlock_windows_file_lock(lock)
    exit_code = return_code
      
  except Exception as e:
    l.p('Execution Engine encountered an error: {}'.format(str(e)), color=ct.COLORS.MAIN)
    l.p(traceback.format_exc())
    l.maybe_unlock_windows_file_lock(lock)    
    l.P("Execution Engine exiting with post-exception code: {}".format(ct.CODE_EXCEPTION), color='r')
    exit_code = ct.CODE_EXCEPTION
  #endtry main loop startup
  return exit_code, eng
  
