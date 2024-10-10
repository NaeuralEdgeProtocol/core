import os
import cv2


from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils
from decentrai_inference.graphs import EfficientDet0InferenceGraph

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
COLOR = (255, 0, 0)
COLOR_LABEL = (0, 255, 0)


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

  # load specific image
  path_movie = os.path.join(
    log.get_dropbox_drive(),
    '_vapor_data/__sources/movies/Animal/URS_6.mp4'
  )

  cap = cv2.VideoCapture(path_movie)
  if not cap.isOpened():
    print("Cannot open camera")
    exit()
  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    # Our operations on the frame come here

    frame_rgb = frame[:, :, ::-1]

    dct_res = graph.predict(frame_rgb)
    lst_inf = dct_res['INFERENCES'][0]
    bears = [x for x in lst_inf if x['TYPE'] in ['bear']]
    if bears:
      img_show = painter.draw_inference_boxes(
        frame,
        bears,
        draw_box_index=False
      )
      path_out = os.path.join(
        log.get_output_folder(),
        '{}_bear.png'.format(log.now_str())
      )
      cv2.imwrite(path_out, img_show)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break
  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
