import os
import cv2
import pandas as pd

from tqdm import tqdm
from time import time
from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils
from decentrai_inference.graphs import EfficientDet0InferenceGraph, \
    EfficientDet3InferenceGraph, EfficientDet4InferenceGraph

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
COLOR = (255, 0, 0)
COLOR_LABEL = (0, 255, 0)

if __name__ == '__main__':
  # create logger
  cfg_file = 'main_config.txt'
  log = Logger(lib_name='VB_EXP', config_file=cfg_file, max_lines=1000)

  # create painter instance
  painter = DrawUtils(log=log)

  # create graph
  graph_effdet0 = EfficientDet0InferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )
  graph_effdet3 = EfficientDet3InferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )
  graph_effdet4 = EfficientDet4InferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )

  lst_graphs = [
    graph_effdet0,
    graph_effdet3,
    graph_effdet4
  ]

  img_dir = os.path.join(
    log.get_dropbox_drive(),
    '_vapor_data',
    '__sources',
    'images',
    'experiment_people_lidl'
  )
  dct_time = {}
  lst = os.listdir(img_dir)
  for name in tqdm(lst):
    img_path = os.path.join(img_dir, name)

    # load specific image
    img_bgr = painter.read(img_path)
    img_rgb = img_bgr[:, :, ::-1]

    for graph in lst_graphs:
      graph_key = graph.config_key
      # make predictions
      start = time()
      dct_res = graph.predict(img_rgb)
      stop = time()
      if graph_key not in dct_time:
        dct_time[graph_key] = {}
      dct_time[graph_key][name] = stop - start

      # draw inference
      lst_inf = dct_res['INFERENCES'][0]
      lst_inf = [x for x in lst_inf if x['TYPE'] == 'person']
      img_show = painter.draw_inference_boxes(img_bgr.copy(), lst_inf, draw_box_index=True)

      # save image
      path_output = os.path.join(log.get_output_folder(), graph_key)
      os.makedirs(path_output, exist_ok=True)
      path_img = os.path.join(path_output, name)
      painter.save(fn=path_img, image=img_show, folder=None)
  # endfor

  for graph_name, dct_graph_time in dct_time.items():
    df = pd.DataFrame(
      {
        'NAME': list(dct_graph_time.keys()),
        'TIME': list(dct_graph_time.values())
       }
    )
    log.save_dataframe(
      df=df,
      fn=graph_name + '.csv',
      folder='output'
    )
