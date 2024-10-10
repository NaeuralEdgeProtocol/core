import numpy as np

from naeural_core import Logger

if __name__ == '__main__':
  import os
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='SB', 
    config_file=cfg_file, 
    max_lines=1000, 
    TF_KERAS=False
    )
  
  NEW_H = 1024
  NEW_W = 1792
  
  path_out = os.path.join(
    log.get_data_folder(), 
    'config',
    'streams_bmw_out'
    )
  os.makedirs(path_out, exist_ok=True)
  
  path_src = os.path.join(
    log.get_data_folder(), 
    'config',
    'streams_bmw'
    )
  
  l = [os.path.join(path_src, x) for x in os.listdir(path_src)]
  
  for path_config in l:
    dct = log.load_json(path_config)
    h_orig = dct['STREAM_CONFIG_METADATA']['TRANSCODER_H']
    w_orig = dct['STREAM_CONFIG_METADATA']['TRANSCODER_W']
    (top, left, bottom, right), (new_h, new_w) = log.center_image_coordinates(
      src_h=h_orig, 
      src_w=w_orig, 
      target_h=NEW_H, 
      target_w=NEW_W
      )
    offset_h = top
    offset_w = left
  
    lst_plugins = dct['PLUGINS']
    for plugin in lst_plugins:
      lst_locations = plugin['LOCATIONS']
      for loc in lst_locations:
        points = loc['POINTS']
        arr_points = np.array(points)
        arr_points[:, 0] = arr_points[:, 0] / w_orig * new_w + offset_w
        arr_points[:, 1] = arr_points[:, 1] / h_orig * new_h + offset_h
        loc['POINTS'] = arr_points.astype(np.int32).tolist()
        
        loc['CONFIDENCE_THRESHOLD'] = 0.3
        loc['ALERT_DATA_COUNT'] = 2
        loc['ALERT_RAISE_CONFIRMATION_TIME'] = 1
        loc['ALERT_LOWER_CONFIRMATION_TIME'] = 7
        loc['ALERT_RAISE_VALUE'] = 0.5
        loc['ALERT_LOWER_VALUE'] = 0.4
        loc['ALERT_REDUCE_VALUE'] = False
        loc['INFERENCE_GRAPHS'] = ['EFF_DET3_768x1358_BS7']
        del loc['VALUES_COUNT']
        del loc['CONFIRMATION_TIME']
        del loc['LOW_DRAWING']
        
    #replace values
    dct['STREAM_CONFIG_METADATA']['TRANSCODER_H'] = NEW_H
    dct['STREAM_CONFIG_METADATA']['TRANSCODER_W'] = NEW_W
    
    log.save_json(
      dct=dct,
      fname=os.path.join(path_out, os.path.basename(path_config))
      )
  #endfor
  
  