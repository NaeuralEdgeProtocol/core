import os
import cv2

import tensorflow.compat.v1 as tf
from tqdm import tqdm
from decentra_vision.draw_utils import DrawUtils
from naeural_core import Logger
from decentrai_inference.graphs import FaceInferenceGraph

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
  graph = FaceInferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )

  lst = [
    # 'xperimental/covid1.jpg',
    'xperimental/covid2.jpg'
  ]
  for img_path in tqdm(lst):
    # load specific image
    img_bgr = painter.read(img_path)
    img_rgb = img_bgr[:, :, ::-1]

    # make predictions
    dct_res = graph.predict(img_rgb)
    lst_inf = dct_res['INFERENCES'][0]

    # filter
    img_show = painter.draw_inference_boxes(img_bgr, lst_inf, draw_box_index=True)
