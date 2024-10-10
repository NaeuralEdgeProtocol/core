import os
import cv2

from tqdm import tqdm
from decentra_vision.draw_utils import DrawUtils
from naeural_core import Logger
from decentrai_inference.graphs import LPDetectionInferenceGraph, LPDv2InferenceGraph

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
  graph = LPDv2InferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )

  img_dir = os.path.join(
    log.get_dropbox_drive(),
    '_vapor_data',
    '__sources',
    'images',
    'experiment_lpd_lidl'
  )
  lst = os.listdir(img_dir)
  for name in tqdm(lst):
    img_path = os.path.join(img_dir, name)

    # load specific image
    img_bgr = painter.read(img_path)
    img_rgb = img_bgr[:, :, ::-1]

    # make predictions
    dct_res = graph.predict(img_rgb)
    lst_inf = dct_res['INFERENCES'][0]
    lst_inf = [x for x in lst_inf if x['TYPE'] == 'license_plate']
    lst_inf = sorted(lst_inf, key=lambda x: x['PROB_PRC'])
    lst_inf = lst_inf[:1]
    # filter
    img_show = painter.draw_inference_boxes(img_bgr, lst_inf, draw_box_index=True)
    painter.save(fn=name, image=img_show)
