import os
import cv2

from decentra_vision.draw_utils import DrawUtils
from naeural_core import Logger
import tensorflow.compat.v1 as tf
from decentrai_inference.graphs import EfficientDet0InferenceGraph, YoloInferenceGraph

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
COLOR = (255, 0, 0)
COLOR_LABEL = (0, 255, 0)


def iou(lst_inf, pers1, pers2):
  coords1 = lst_inf[pers1]['TLBR_POS']
  coords2 = lst_inf[pers2]['TLBR_POS']

  [top1, left1, bottom1, right1] = coords1
  [top2, left2, bottom2, right2] = coords2

  top = max(top1, top2)
  left = max(left1, left2)
  right = min(right1, right2)
  bottom = min(bottom1, bottom2)

  h_overlap = bottom - top + 1
  w_overlap = right - left + 1

  area_overlap = h_overlap * w_overlap

  area_union = (right1 - left1 + 1) * (bottom1 - top1 + 1) + \
      (right2 - left2 + 1) * (bottom2 - top2 + 1) - \
      area_overlap
  iou = area_overlap / area_union
  return iou if iou > 0 else 0


if __name__ == '__main__':
  # img_name = 'vlcsnap-2021-07-02-13h26m43s659.png'
  img_name = 'vlcsnap-2021-07-29-09h15m46s252.png'

  # create logger
  cfg_file = 'main_config.txt'
  log = Logger(lib_name='VB', config_file=cfg_file, max_lines=1000)

  # create painter instance
  painter = DrawUtils(log=log)

  # create EffDet0 instance in order to make predicts
  graph = EfficientDet0InferenceGraph(
    log=log,
    config_path='_local_cache/_data/config/config_inference.txt'
  )
  if False:
    graph = YoloInferenceGraph(
      log=log,
      config_path='_local_cache/_data/config/config_inference.txt'
    )

  # load specific image
  path_img = os.path.join(log.get_output_folder(), 'inference', img_name)
  img_bgr = painter.read(path_img)
  img_rgb = img_bgr[:, :, ::-1]

  # make predictions
  dct_res = graph.predict(img_rgb)
  lst_inf = dct_res['INFERENCES'][0]
  # filter
  # lst_inf = [x for x in lst_inf if x['TYPE'] in ['person']]
  birds = [x for x in lst_inf if x['TYPE'] in ['bird']]
  print(birds)
  lst_inf = [x for x in lst_inf]
  img_show = painter.draw_inference_boxes(img_bgr, lst_inf, draw_box_index=True)

  # blur persons
  # for x in lst_inf:
  #   img_show = painter.blur_person(
  #         frame=img_show,
  #         top=x['TLBR_POS'][0],
  #         left=x['TLBR_POS'][1],
  #         bottom=x['TLBR_POS'][2],
  #         right=x['TLBR_POS'][3],
  #         object_type='person'
  #         )

  # tensorflow NMS check
  # print('-----------------')
  # print(tf.image.non_max_suppression_with_scores(
  #       [lst_inf[3]['TLBR_POS'], lst_inf[11]['TLBR_POS']],
  #       [lst_inf[3]['PROB_PRC'], lst_inf[11]['PROB_PRC']],
  #       10,
  #       iou_threshold=0.25,
  #       score_threshold=0.2,
  #       soft_nms_sigma=0.0))
  # print('-----------------')

  print(iou(lst_inf, 3, 11))
  # img_show = cv2.resize(img_show, (1920,1080))
  painter.show('openspace', img_show)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
