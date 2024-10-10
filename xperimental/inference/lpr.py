import os
import cv2
import string
import numpy as np

from tqdm import tqdm
from decentra_vision.draw_utils import DrawUtils
from naeural_core import Logger
from decentrai_inference.graphs import LPRInferenceGraph

lst_lpr_chars = list(range(10)) + list(string.ascii_uppercase) + [' ']
dct_lpr_chars_idx = {str(k): v for k, v in zip(lst_lpr_chars, range(len(lst_lpr_chars)))}
dct_lpr_idx_chars = {v: k for k, v in dct_lpr_chars_idx.items()}


def decode_lpr(preds):
  lp = [dct_lpr_idx_chars[c] for c in preds]
  return lp


if __name__ == '__main__':
  # create logger
  cfg_file = 'main_config.txt'
  log = Logger(lib_name='VB_EXP', config_file=cfg_file, max_lines=1000)

  # create painter instance
  painter = DrawUtils(log=log)

  # create graph
  graph = LPRInferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )

  img_dir = os.path.join(
    log.get_dropbox_drive(),
    '_vapor_data',
    '__sources',
    'images',
    'experiment_lpr_lidl'
  )
  lst = os.listdir(img_dir)
  for name in tqdm(lst):
    img_path = os.path.join(img_dir, name)

    # load specific image
    img_bgr = painter.read(img_path)
    img_rgb = img_bgr[:, :, ::-1]

    # make predictions
    dct_res = graph.predict(img_rgb)
    np_probs = dct_res['INFERENCES'][0]
    preds = np.argmax(np_probs, axis=-1)
    lpr_chars = decode_lpr(preds)
    lpr = ''.join(lpr_chars)
    painter.save(image=img_bgr, fn=lpr + '.png')
