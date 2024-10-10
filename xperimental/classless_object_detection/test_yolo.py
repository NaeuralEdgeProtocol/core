import cv2
import numpy as np
import os

from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils

from plugins.serving.architectures.y5.general import (
  scale_coords,
  xywh2xyxy
)

# direct yolo tests

from naeural_core.serving.serving_manager import get_raw_server


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

  files = [
      'xperimental/_images/SHELVES/H1568xW2592/pic1.jpg',
      'xperimental/_images/SHELVES/H1568xW2592/pic2.jpg',
    ]

  l = Logger('CLOD', base_folder='.', app_folder='_local_cache')
  t = l.is_main_thread

  painter = DrawUtils(log=l)

  lst_imgs = get_test_data(files=files)

  svr = get_raw_server(log=l, server_name='th_y5l6s')
  inputs = [np.ascontiguousarray(x) for x in lst_imgs]

  res = svr.forward(inputs)
  preds = res[0]
  all_preds = []
  for i in range(len(preds)):
    good = preds[i, :, 4] > 0.2
    good_preds = preds[i, good]
    boxes = xywh2xyxy(good_preds[:, :4])
    img_boxes = scale_coords(
        img1_shape=svr.cfg_input_size,
        coords=boxes[:, :4].cpu(),
        img0_shape=lst_imgs[0].shape[:2],
      ).round()
    all_preds.append(img_boxes)

  all_preds = [x.cpu().numpy() for x in all_preds]

  if True:
    for i in range(len(lst_imgs)):
      img = lst_imgs[i][:, :, ::-1]
      inferences = [
        {'TLBR_POS': [x[1], x[0], x[3], x[2]]} for x in all_preds[i]
      ]
      img = painter.draw_inference_boxes(
        image=np.ascontiguousarray(img),
        lst_inf=inferences,
        draw_label=False,
      )
      painter.show(files[i], img)
