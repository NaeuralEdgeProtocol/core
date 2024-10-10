import cv2 as cv
import os
import numpy as np
from naeural_core.core_logging import SBLogger
from naeural_core.local_libraries.vision.ffmpeg_utils import FFMPEGUtils
from naeural_core.local_libraries.vision.ffmpeg_writer import FFmpegWriter
from decentra_vision.draw_utils import DrawUtils
from naeural_core.xperimental.video_checking.video_checking import get_rotate_code_from_orientation, correct_rotation
from threading import Thread
from functools import partial
import random
from collections import deque


class ThreadRunner:
  def __init__(self, logger, blur_type, blur_legit):
    self.threads = []
    self.files = []
    self.log = logger
    self.blur_type = blur_type
    self.blur_legit = blur_legit

  def run_thread(self):
    idx = self.current_thread
    fn = self.files[idx]
    out_fn = f'out_{idx}.mp4'
    dummy_blur_movie(
      log=self.log,
      video_path=fn,
      dst_path=out_fn,
      use_good_rotation=True,
      rotate_before_write=False,
      blur_method=partial(random_painter_blur, blur_type=self.blur_type) if self.blur_legit else None
    )
    return

  def run(self, fn):
    thr = Thread(target=self.run_thread, daemon=True)
    self.threads.append(thr)
    self.files.append(fn)
    self.current_thread = len(self.threads) - 1
    thr.start()
    return


def display_video(video_path, use_good_rotation=True, show_orientation=False):
  cap = cv.VideoCapture(video_path)
  if use_good_rotation:
    cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
  cap_orientation = cap.get(cv.CAP_PROP_ORIENTATION_META)
  rotate_code = get_rotate_code_from_orientation(cap_orientation)

  q_pressed = False

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      # if use_good_rotation:
      #   frame = correct_rotation(frame, rotate_code)
      cv.imshow(f'Video{f"_{cap_orientation}" if show_orientation else ""}', frame)
      pressed_key = cv.waitKey(25) & 0xFF
      # Press Q on keyboard to  exit
      if pressed_key == ord('q'):
        q_pressed = True
        break
      elif pressed_key == ord('f'):
        break
    # Break the loop
    else:
      break
  # When everything done, release the video capture object
  cap.release()
  # Closes all the frames
  cv.destroyAllWindows()
  return q_pressed


def _blur_persons(self, frame, lst_persons):
  """
  Blurs detected persons on current frame

  Params:
    frame: current movie frame
    lst_persons: list of persons to be blurred

  Returns:
    frame: frame with persons blurred
  """

  for dct_inf in lst_persons:
    top, left, bottom, right = dct_inf[ct.TLBR_POS]
    height = bottom - top
    height = int(self.cfg_person_blur * height)
    bottom = top + height

    self.start_timer('blur_person')
    frame = self._painter.blur_person(
      frame=frame,
      top=top,
      left=left,
      bottom=bottom,
      right=right
    )
    self.end_timer('blur_person')
  return frame


N_BLURS = 5


def random_painter_blur(frame, painter, blur_type='raw'):
  h, w, _ = frame.shape
  blur_list = []
  for _ in range(N_BLURS):
    lst_w, lst_h = [], []
    lst_h.append(random.randint(0, h - 50))
    lst_w.append(random.randint(0, w - 50))
    lst_h.append(random.randint(0, h - 50))
    lst_w.append(random.randint(0, w - 50))
    blur_list.append(
      [
        min(lst_h),
        min(lst_w),
        max(lst_h),
        max(lst_w)
      ]
    )

  for blur in blur_list:
    if blur_type == 'raw':
      frame = painter.blur_raw(
        frame,
        top=blur[0],
        left=blur[1],
        bottom=blur[2] + 20,
        right=blur[3] + 20
      )
    elif blur_type == 'adaptive':
      frame = painter.blur_adaptive(
        frame,
        top=blur[0],
        left=blur[1],
        bottom=blur[2] + 20,
        right=blur[3] + 20
      )
    else:
      raise ValueError

  return frame


def dummy_blur_movie(log, video_path, dst_path, use_good_rotation=True, rotate_before_write=True, blur_method=None):
  cap = cv.VideoCapture(video_path)
  if use_good_rotation:
    cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
  cap_orientation = cap.get(cv.CAP_PROP_ORIENTATION_META)
  rotate_code = get_rotate_code_from_orientation(cap_orientation)
  reverse_rotate_code = get_rotate_code_from_orientation(cap_orientation, for_undo=True)
  frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

  # frame_size = frame_size[1], frame_size[0]

  out = FFmpegWriter(
    filename=dst_path,
    fps=cap.get(cv.CAP_PROP_FPS),
    frameSize=frame_size,
    log=log,
    overall_options='',  # '-stats_period 10'
    use_shell=True,
    pipe_cmd=''  # "awk 'NR % {} == 0'".format(int(cap.get(cv.CAP_PROP_FPS)) * 30),
  )
  painter = DrawUtils(log=log, timers_section=dst_path)

  period = 1000
  cnt = 0
  dq = deque(maxlen=256)
  while cap.isOpened():
    cnt += 1
    ret, frame = cap.read()
    if cnt % period == 0:
      log.P(f'At frame number {cnt} when making {dst_path}')
    if ret == True:
      if use_good_rotation:
        frame = correct_rotation(frame, rotate_code)

      if blur_method is not None:
        frame = blur_method(frame, painter)
      else:
        frame[100:200, 100:150] = np.array([255, 0, 0])

      if use_good_rotation and rotate_before_write:
        frame = correct_rotation(frame, reverse_rotate_code)

      out.write(frame)
      # dq.append(frame)
      # if len(dq) > 250:
      #   len_dq = len(dq)
      #   for _ in range(len_dq):
      #     out.write(dq.popleft())
    else:
      break
  out.release()
  cap.release()
  return


if __name__ == '__main__':
  log = SBLogger()
  # video_path = '/home/bleo/debug/Test.mp4'
  video_path = 'C:\\Users\\Stefan.saraev\\Downloads\\debug\\test.mp4'
  # video_path = '/home/bleo/debug/lung.mp4'
  path_prefix = 'C:\\Users\\Stefan.saraev\\Downloads\\debug\\shards'
  output_path_prefix = 'C:\\Users\\Stefan.saraev\\Downloads\\debug\\outputs'
  result_path = 'C:\\Users\\Stefan.saraev\\Downloads\\debug\\final.mp4'

  if False:
    thread_runner = ThreadRunner(logger=log, blur_type='adaptive', blur_legit=True)
    N_JOBS = 5
    for _ in range(N_JOBS):
      thread_runner.run(video_path)
    exit(-1)

  utils_obj = FFMPEGUtils()
  print('Splitting video')
  dct_shards = utils_obj.split_video_file(
    path=video_path, nr_chunks=4,
    path_to_output=path_prefix
  )

  paths = dct_shards['output_files']
  # print('Displaying shards')
  # for path in paths:
  #   if display_video(os.path.join(path_prefix, path), use_good_rotation=True, show_orientation=True):
  #     break

  processed_paths = []
  print('Applying `blur`')
  for fn in paths:
    print(f'Start processing {fn}')
    path = os.path.join(path_prefix, fn)
    dst_path = os.path.join(output_path_prefix, fn)
    dummy_blur_movie(log=log, video_path=path, dst_path=dst_path, use_good_rotation=False, rotate_before_write=False)
    processed_paths.append(dst_path)
    print(f'Done processing {fn}')

  # print('Displaying processed shards')
  # for path in processed_paths:
  #   if display_video(os.path.join(path_prefix, path), use_good_rotation=False, show_orientation=True):
  #     break

  print('Concatenating shards')
  utils_obj.concatenate_multiple_video_files(input_paths=processed_paths, output_path=result_path)
