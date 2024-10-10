import cv2
import numpy as np
import os

import tensorflow as tf

from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils


def cv2_clod(im, model='xperimental/classless_object_detection/model.yml.gz'):
  edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
  # if this line is causing a bug it will be replaced with: rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  # until further testing this will remain
  rgb_im = im[:, :, ::-1]
  edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

  orimap = edge_detection.computeOrientation(edges)
  edges = edge_detection.edgesNms(edges, orimap, r=30)

  edge_boxes = cv2.ximgproc.createEdgeBoxes()
  edge_boxes.setMaxBoxes(100)
  boxes = edge_boxes.getBoundingBoxes(edges, orimap)

  res = []
  for b in boxes:
    L, T, w, h = b
    R = L + w
    B = T + h
    res.append({
      'TLBR_POS': (T, L, B, R),
      'TYPE': 'unk',
      'PROB_PRC': -1,
    }
      )
  return res


def get_test_data(files, resize=None):

  check_files = [os.path.isfile(x) for x in files]
  missing = [files[i] for i, x in enumerate(check_files) if not x]
  if sum(check_files) != len(check_files):
    raise ValueError("Failed to find test images: {}".format(missing))

  lst_imgs = [
    np.ascontiguousarray(cv2.imread(x)) for x in files
  ]

  if resize is not None:
    lst_imgs = [
      cv2.resize(x, (resize[1], resize[0])) for x in lst_imgs
    ]

  lst_imgs = [
    x[:, :, ::-1] for x in lst_imgs
  ]

  return lst_imgs


if __name__ == '__main__':

  model_path = 'xperimental/classless_object_detection/models/mobile_object_localizer'

  files = [
      # 'xperimental/_images/H480_W640/faces1.jpg' ,
      # 'xperimental/_images/H768_W1024/faces4.jpg' ,
      # 'xperimental/_images/H720_W1280/faces2.jpg' ,
      'xperimental/_images/H1080_W1920/faces21.jpg',
      'xperimental/_images/H1080_W1920/faces17.jpg',
    ]

  l = Logger('CLOD', base_folder='.', app_folder='_cache')
  t = l.is_main_thread

  painter = DrawUtils(log=l)

  lst_imgs = get_test_data(files=files)
  np_imgs = np.array([cv2.resize(x, (192, 192)) for x in lst_imgs])
  np_imgs = (np_imgs / 255).astype('float32')

  saved_model = tf.saved_model.load(model_path)
  model_func = saved_model.signatures['default']

  dct_res = model_func(tf.constant(np_imgs))

  for i in range(len(lst_imgs)):
    img = lst_imgs[i][:, :, ::-1]
    scores = dct_res['detection_scores'][i].numpy()
    boxes = dct_res['detection_boxes'][i].numpy()
    boxes[:, 0] = boxes[:, 0] * lst_imgs[0].shape[0]
    boxes[:, 2] = boxes[:, 2] * lst_imgs[0].shape[0]
    boxes[:, 1] = boxes[:, 1] * lst_imgs[0].shape[1]
    boxes[:, 3] = boxes[:, 3] * lst_imgs[0].shape[1]
    inferences = []
    for j, score in enumerate(scores):
      if score > 0.2:
        inferences.append({
          'TLBR_POS': boxes[j].tolist(),
          'PROB_PRC': score,
          'TYPE': 1
        })
    img = painter.draw_inference_boxes(
      image=np.ascontiguousarray(img),
      lst_inf=inferences,
    )
    painter.show(files[i], img)
