import os
import cv2

from tqdm import tqdm
from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils
from decentrai_inference.graphs import EfficientDet0InferenceGraph, \
    EfficientDet3InferenceGraph, EfficientDet4InferenceGraph, FaceInferenceGraph

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
  graph_face = FaceInferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )

  img_dir = os.path.join(
    log.get_dropbox_drive(),
    '_vapor_data',
    '__sources',
    'images',
    'experiment_people_lidl2'
  )
  dct_time = {}
  lst = os.listdir(img_dir)
  for name in tqdm(lst):
    img_path = os.path.join(img_dir, name)

    # load specific image
    img_bgr = painter.read(img_path)
    img_rgb = img_bgr[:, :, ::-1]

    dct_inf_effdet0 = graph_effdet0.predict(img_rgb)
    lst_inf_effdet = dct_inf_effdet0['INFERENCES'][0]

    dct_inf_face = graph_face.predict(img_rgb)
    lst_inf_face = dct_inf_face['INFERENCES'][0]
