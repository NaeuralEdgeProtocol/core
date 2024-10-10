import os
import cv2
import numpy as np
import pandas as pd

from decentra_vision.draw_utils import DrawUtils
from naeural_core import Logger
from decentrai_inference.graphs import EfficientDet0InferenceGraph, EfficientDet2InferenceGraph, EfficientDet2640x1132InferenceGraph

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
COLOR = (255, 0, 0)
COLOR_LABEL = (0, 255, 0)

BATCH_SIZE_14 = 14
DOWNSCALE = 50


def save_image(base_path, img):
  # save full image
  # path_full = os.path.join(base_path, 'full', log.now_str() + '.png')
  path_full = os.path.join(base_path, log.now_str() + '.png')
  cv2.imwrite(path_full, img)

  # #save downscaled image
  # width = int(img.shape[1] * DOWNSCALE / 100)
  # height = int(img.shape[0] * DOWNSCALE / 100)
  # dim = (width, height)
  # resized = cv2.resize(img, dim)
  # path_downscaled = os.path.join(base_path, str(DOWNSCALE), log.now_str() + '.png')
  # cv2.imwrite(path_downscaled, resized)
  return


if __name__ == '__main__':
  path_img = 'xperimental/_bmw_images/vlcsnap-2021-09-02-13h23m42s756.png'

  # create logger
  cfg_file = 'main_config.txt'
  config_path = '_local_cache/_data/config/config_inference.txt'

  log = Logger(lib_name='VB', config_file=cfg_file, max_lines=1000)
  painter = DrawUtils(log=log)
  config = log.load_json(config_path)

  path_output = log.check_folder_output('effd0_vs_effd2')
  path_out_d0 = os.path.join(path_output, 'effd0')
  path_out_d2 = os.path.join(path_output, 'effd2')
  path_out_d2_640x1132 = os.path.join(path_output, 'effd2_640x1132')
  for fld in [path_out_d0, path_out_d2, path_out_d2_640x1132]:
    os.makedirs(fld, exist_ok=True)

  # create EffD0
  eff_d0 = EfficientDet0InferenceGraph(
    log=log,
    config_path=config_path
  )

  # create EffD2
  config_graph = config['EFF_DET2']
  models_path = config_graph['MODELS_PATH']
  graph = models_path.format(BATCH_SIZE_14)
  config_graph['BATCH_SIZE'] = BATCH_SIZE_14
  config_graph['GRAPH'] = graph
  eff_d2 = EfficientDet2InferenceGraph(
    log=log,
    config_graph=config_graph
  )

  # create EFF_DET2_640x1132
  config_graph = config['EFF_DET2_640x1132']
  models_path = config_graph['MODELS_PATH']
  graph = models_path.format(BATCH_SIZE_14)
  config_graph['BATCH_SIZE'] = BATCH_SIZE_14
  config_graph['GRAPH'] = graph
  effd2_640x1132 = EfficientDet2640x1132InferenceGraph(
    log=log,
    config_graph=config_graph
  )

  # load specific image
  img_bgr = painter.read(path_img)
  img_rgb = img_bgr[:, :, ::-1]

  np_imgs = np.array([img_rgb for _ in range(BATCH_SIZE_14)])

  dct = {'MODEL': [], 'NR_OBJECTS': []}
  # make predictions with EffDet0
  eff_d0.predict(img_rgb)
  log.start_timer('eff_d0')
  for i in range(20):
    dct_res = eff_d0.predict(img_rgb)
  log.stop_timer('eff_d0')

  lst_inf = dct_res['INFERENCES'][0]
  img_show = painter.draw_inference_boxes(
    image=img_bgr,
    lst_inf=lst_inf,
    draw_box_index=True
  )
  dct['MODEL'].append('eff_d0')
  dct['NR_OBJECTS'].append(len(lst_inf))

  save_image(
    base_path=path_out_d0,
    img=img_show
  )

  lst = [
    ('eff_d2', eff_d2, path_out_d2),
    ('effd2_640x1132', effd2_640x1132, path_out_d2_640x1132)
  ]
  for name, graph, path in lst:
    graph.predict(np_imgs)
    log.start_timer(name)
    for i in range(20):
      dct_res = graph.predict(np_imgs)
    log.stop_timer(name)

    for i in range(BATCH_SIZE_14):
      lst_inf = dct_res['INFERENCES'][i]
      crt_rgb = np_imgs[i]
      crt_bgr = crt_rgb[:, :, ::-1].copy()
      img_show = painter.draw_inference_boxes(
        image=crt_bgr,
        lst_inf=lst_inf,
        draw_box_index=True
      )

      save_image(
        base_path=path,
        img=img_show
      )
      dct['MODEL'].append(name)
      dct['NR_OBJECTS'].append(len(lst_inf))
    # endfor
  # endfor

  log.show_timers()
  df = pd.DataFrame(dct)
  log.p('\n\n{}'.format(df))
