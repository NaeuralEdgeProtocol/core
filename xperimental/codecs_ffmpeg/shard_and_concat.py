import cv2 as cv
import os
import numpy as np
import time
from naeural_core.core_logging import SBLogger
from naeural_core.local_libraries.vision.ffmpeg_utils import FFMPEGUtils
from naeural_core.local_libraries.vision.ffmpeg_writer import FFmpegWriter
from naeural_core.xperimental.video_checking.video_checking import get_rotate_code_from_orientation, correct_rotation


def display_video(video_path, use_good_rotation=True, show_orientation=False, name='Video'):
  cap = cv.VideoCapture(video_path)
  if use_good_rotation:
    cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
  cap_orientation = cap.get(cv.CAP_PROP_ORIENTATION_META)
  rotate_code = get_rotate_code_from_orientation(cap_orientation)

  q_pressed = False

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      if use_good_rotation:
        frame = correct_rotation(frame, rotate_code)
      cv.imshow(f'{name}{f"_{cap_orientation}" if show_orientation else ""}', frame)
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


def dummy_blur_movie(log, video_path, dst_path, use_good_rotation=True, rotate_before_write=True):
  cap = cv.VideoCapture(video_path)
  if use_good_rotation:
    cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
  cap_orientation = cap.get(cv.CAP_PROP_ORIENTATION_META)
  rotate_code = get_rotate_code_from_orientation(cap_orientation)
  reverse_rotate_code = get_rotate_code_from_orientation(cap_orientation, for_undo=True)
  frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

  print(f'ORIENTATION: {cap_orientation}')
  print(f'FPS: {cap.get(cv.CAP_PROP_FPS)}')

  if not rotate_before_write and use_good_rotation and rotate_code is not None:
    frame_size = frame_size[1], frame_size[0]

  out = FFmpegWriter(
    filename=dst_path,
    fps=cap.get(cv.CAP_PROP_FPS),
    frameSize=frame_size,
    log=log
  )
  printed = False

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      if use_good_rotation:
        if not printed:
          print(f'performed rotation for rotate_code {rotate_code}')
        frame = correct_rotation(frame, rotate_code)

      frame[100:200, 100:150] = np.array([255, 0, 0])

      if use_good_rotation and rotate_before_write:
        if not printed:
          print(f'performed rotation for rotate_code {reverse_rotate_code}')
        frame = correct_rotation(frame, reverse_rotate_code)
      printed = True
      out.write(frame)
    else:
      break
  out.release()
  cap.release()
  return

if __name__ == '__main__':
  log = SBLogger()
  video_path = '/home/bleo/debug/Test.mp4'
  video_path = '/home/bleo/Downloads/telefon.mp4'
  video_path = '/home/bleo/debug/Test 2 - Portret 2 - 144p.mp4'
  path_prefix = '/home/bleo/debug/ee_shards'
  output_path_prefix = '/home/bleo/debug/ee_shards_out'
  result_path = f'/home/bleo/debug/final_lung_{time.time()}.mp4'

  USE_GOOD_ROTATION = True
  SHOW_ORIENTATION = True
  ROTATE_BEFORE = False

  utils_obj = FFMPEGUtils()
  if True:
    display_video(video_path, use_good_rotation=USE_GOOD_ROTATION, show_orientation=SHOW_ORIENTATION, name='Original')

    print('Splitting video')
    log.start_timer('split_video')
    dct_shards = utils_obj.split_video_file(
      path=video_path, nr_chunks=4,
      path_to_output=path_prefix
    )
    log.end_timer('split_video', skip_first_timing=False)


    paths = dct_shards['output_files']
    print('Displaying shards')
    for path in paths:
      if display_video(
          os.path.join(path_prefix, path),
          use_good_rotation=USE_GOOD_ROTATION,
          show_orientation=SHOW_ORIENTATION,
          name=path
      ):
        break

    processed_paths = []

    print('Applying `blur`')
    for fn in paths:
      log.start_timer(f'read_video_{fn}')
      print(f'Start processing {fn}')
      path = os.path.join(path_prefix, fn)
      dst_path = os.path.join(output_path_prefix, fn)
      dummy_blur_movie(
        log=log,
        video_path=path,
        dst_path=dst_path,
        use_good_rotation=USE_GOOD_ROTATION,
        rotate_before_write=ROTATE_BEFORE
      )
      processed_paths.append(dst_path)
      log.end_timer(f'read_video_{fn}', skip_first_timing=False)
      print(f'Done processing {fn}')
  else:
    processed_paths = [
      '/home/bleo/debug/ee_shards_out/output.mp4',
      '/home/bleo/debug/ee_shards_out/output1.mp4',
      '/home/bleo/debug/ee_shards_out/output2.mp4',
      '/home/bleo/debug/ee_shards_out/output3.mp4'
    ]
  print('Displaying processed shards')
  for path in processed_paths:
    if display_video(os.path.join(path_prefix, path), use_good_rotation=USE_GOOD_ROTATION, show_orientation=SHOW_ORIENTATION):
      break

  print('Concatenating shards')
  log.start_timer('concatenate_video')
  utils_obj.concatenate_multiple_video_files(input_paths=processed_paths, output_path=result_path)
  log.end_timer('concatenate_video', skip_first_timing=False)
  print('Finished concatenating')

  log.show_timers()