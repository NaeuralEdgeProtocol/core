import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from naeural_core import Logger

def load_payloads(log, path_folder):
  l = filter(lambda x: x.endswith('.txt'), os.listdir(path_folder))
  l_paths = [os.path.join(path_folder, x) for x in l]
  l_payloads = [log.load_json(fn=x, verbose=False) for x in l_paths]
  return l_paths, l_payloads

def get_alert_objects_confidence(l_paths, l_payloads):
  dct = {'PATH': [], 'PROB': [], 'LBL': []}
  for path, full_payload in zip(l_paths, l_payloads):
    stream_payload = full_payload['PAYLOAD']    
    lst_results = stream_payload['RESULTS']
    for result in lst_results:
      plugin_payload = result['PAYLOAD']
      
      alert_objects = plugin_payload.get('ALERT_OBJECTS', [])
      
      for obj in alert_objects:
        prob = obj['PROB_PRC']
        lbl = obj['TYPE']
        prob = round(prob, 1)
        dct['PATH'].append(path)
        dct['LBL'].append(lbl)
        dct['PROB'].append(prob)
      #endfor
    #endfor
  #endfor
  df = pd.DataFrame(dct)
  return df

def get_objects_confidence(l_paths, l_payloads):
  dct = {'PATH': [], 'PROB': [], 'LBL': []}
  for path, full_payload in zip(l_paths, l_payloads):
    lst_inf = full_payload['INFERENCES']    
    for obj in lst_inf:
      prob = obj['PROB_PRC']
      lbl = obj['TYPE']
      prob = round(prob, 1)
      dct['PATH'].append(path)
      dct['LBL'].append(lbl)
      dct['PROB'].append(prob)
    #endfor
  #endfor
  df = pd.DataFrame(dct)
  return df

def distribution(name, df_prob, thr):
  l_prob = df_prob['PROB'].tolist()
  log.p('Report for {}'.format(name))
  
  log.p('Ploting confidence distribution @ confidence threshold ({}%)'.format(thr))
  log.p('Total number of unique images that contain objects @thr={}: {}'.format(thr, len(df_prob['PATH'].unique())))
  log.p('Total number of objects identified @thr={}: {}'.format(thr, len(l_prob)))
  
  unique, counts = np.unique(l_prob, return_counts=True)
  plt.bar(
    x=unique,
    height=counts
    )
  plt.xlabel('Confidence %')
  plt.ylabel('Number of occurances')
  plt.title('{}: False positives confidence distribution @ thr={}%'.format(name, thr))
  plt.show()
  return

if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='EE_BMW', 
    config_file=cfg_file, 
    max_lines=1000, 
    TF_KERAS=False
    )
  
  log.set_nice_prints()  
  
  folder_name = '20210923_20210924'
  
  path_base = os.path.join(
      log.get_dropbox_drive(),
      '_vapor_data',
      '__sources',
      'images',
      'report_bmw',
      folder_name
      )
  
  l = [os.path.join(path_base, x) for x in os.listdir(path_base) if x != 'orig']  
  l_folders = list(filter(lambda x: os.path.isdir(x), l))
  
  l_folders = [path_base] + l_folders
  
  for i, fld in enumerate(l_folders):
    fld_name = os.path.basename(fld)
    l_paths, l_payloads = load_payloads(
      log=log,
      path_folder=fld
      )
    
    THR = 30
    if i == 0:
      df_prob = get_alert_objects_confidence(l_paths, l_payloads)
    else:
      df_prob = get_objects_confidence(l_paths, l_payloads)
    distribution(fld_name, df_prob, THR)
    
    THR = 50
    df_prob50 = df_prob[df_prob['PROB'] >= THR]
    distribution(fld_name, df_prob50, THR)
    log.p('')
  
  # #obtain report for original payloads
  # if True:
  #   log.p('Obtaining distribution for production')
  #   path_folder = os.path.join(
  #     log.get_dropbox_drive(),
  #     '_vapor_data',
  #     '__sources',
  #     'images',
  #     'report_bmw',
  #     folder_name
  #     )
  #   l_paths, l_payloads = load_payloads(
  #     log=log,
  #     path_folder=path_folder
  #     )
    
  #   THR = 30
  #   df_prob = get_alert_objects_confidence(l_paths, l_payloads)
  #   distribution('DEFAULT', df_prob, THR)
    
  #   THR = 50
  #   df_prob50 = df_prob[df_prob['PROB'] >= THR]
  #   distribution('DEFAULT', df_prob50, THR)
  #   log.p('')
    
  
  # #obtain report after inference with EffDet3
  # if True:
  #   fld = 'EffDet3768x1358BS1'
  #   log.p('Obtaining distribution for {}'.format(fld))
  #   path_folder = os.path.join(
  #     log.get_dropbox_drive(),
  #     '_vapor_data',
  #     '__sources',
  #     'images',
  #     'report_bmw',
  #     folder_name,
  #     fld
  #     )
  #   l_paths, l_payloads = load_payloads(
  #     log=log,
  #     path_folder=path_folder
  #     )
    
  #   THR = 30
  #   df_prob = get_objects_confidence(l_paths, l_payloads)    
  #   distribution(fld, df_prob, THR)
    
  #   THR = 50
  #   df_prob50 = df_prob[df_prob['PROB'] >= THR]
  #   distribution(fld, df_prob50, THR)
  #   log.p('')
  
  # #obtain report after inference with EffDet3
  # if True:
  #   fld = 'EffDet3'
  #   log.p('Obtaining distribution for {}'.format(fld))
  #   path_folder = os.path.join(
  #     log.get_dropbox_drive(),
  #     '_vapor_data',
  #     '__sources',
  #     'images',
  #     'report_bmw',
  #     folder_name,
  #     fld
  #     )
  #   l_paths, l_payloads = load_payloads(
  #     log=log,
  #     path_folder=path_folder
  #     )
    
  #   THR = 30
  #   df_prob = get_objects_confidence(l_paths, l_payloads)    
  #   distribution(fld, df_prob, THR)
    
  #   THR = 50
  #   df_prob50 = df_prob[df_prob['PROB'] >= THR]
  #   distribution(fld, df_prob50, THR)
  #   log.p('')
    
    
  # #obtain report after inference with EffDet5
  # if True:
  #   fld = 'EffDet5'
  #   log.p('Obtaining distribution for {}'.format(fld))
  #   path_folder = os.path.join(
  #     log.get_dropbox_drive(),
  #     '_vapor_data',
  #     '__sources',
  #     'images',
  #     'report_bmw',
  #     folder_name,
  #     fld
  #     )
  #   l_paths, l_payloads = load_payloads(
  #     log=log,
  #     path_folder=path_folder
  #     )
    
  #   THR = 30
  #   df_prob = get_objects_confidence(l_paths, l_payloads)    
  #   distribution(fld, df_prob, THR)
    
  #   THR = 50
  #   df_prob50 = df_prob[df_prob['PROB'] >= THR]
  #   distribution(fld, df_prob50, THR)
  #   log.p('')
    
    
  # #obtain report after inference with EffDet7
  # if True:
  #   fld = 'EffDet7'
  #   log.p('Obtaining distribution for {}'.format(fld))
  #   path_folder = os.path.join(
  #     log.get_dropbox_drive(),
  #     '_vapor_data',
  #     '__sources',
  #     'images',
  #     'report_bmw',
  #     folder_name,
  #     fld
  #     )
  #   l_paths, l_payloads = load_payloads(
  #     log=log,
  #     path_folder=path_folder
  #     )
    
  #   THR = 30
  #   df_prob = get_objects_confidence(l_paths, l_payloads)    
  #   distribution(fld, df_prob, THR)
    
  #   THR = 50
  #   df_prob50 = df_prob[df_prob['PROB'] >= THR]
  #   distribution(fld, df_prob50, THR)
  #   log.p('')