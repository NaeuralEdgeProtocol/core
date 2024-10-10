import os
from time import sleep
import cv2
import numpy as np

from naeural_core.local_libraries.vision import FFmpegWriter

if __name__ == '__main__':

  
  from naeural_core import Logger
  
  l = Logger('h264', base_folder='.', app_folder='_local_cache')
  
  VS_CV2 = False
  VIDEO = '_local_cache/_data/test.avi'
  USE_VIDEO = True
  
  width, height = 1920, 1080
  fps = 15
  TIME = 2000 // fps
  nr_frames = fps * TIME
  imgs = []
  
  if USE_VIDEO == True:
    l.P("Loading video...", color='g')
    cap = cv2.VideoCapture(VIDEO)
    done = False
    while not done:
      ok, img = cap.read()
      if ok:
        imgs.append(img)
      else:
        done = True
    l.P("Done loading video", color='g')
  else:
    for i in range(nr_frames):
      # Build synthetic image for testing ("render" a video frame).
      img = np.full((height, width, 3), 60, np.uint8)
      cv2.putText(img, str(i+1), (width//2-100*len(str(i+1)), height//2+100), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 30, 30), 20)  # Blue number
      imgs.append(img)
      # out1.write(img)
    
  frames = np.array(imgs[:nr_frames])
  
  # single frame
  l.P("Starting single frame process...", color='g')
  l.start_timer('FFmpegWriter', section='FFmpegWriter')
  l.start_timer('FFmpegWriter_create', section='FFmpegWriter')
  out1 = FFmpegWriter(
    filename=os.path.join(l.get_output_folder(), 'test_ffmpeg1.mp4'), 
    fps=fps, 
    frameSize=(width, height),
    log=l,
  )  
  l.stop_timer('FFmpegWriter_create', section='FFmpegWriter')  
  l.start_timer('FFmpegWriter_loop_write', section='FFmpegWriter')
  for frame in frames:
    out1.write(frame)
  l.stop_timer('FFmpegWriter_loop_write', section='FFmpegWriter')
  
  l.start_timer('FFmpegWriter_release', section='FFmpegWriter')
  out1.release()
  l.stop_timer('FFmpegWriter_release', section='FFmpegWriter')
  l.stop_timer('FFmpegWriter', section='FFmpegWriter')

  l.P("Done single frame process.", color='g')
  
  sleep(2)
  
  l.P("Starting batch frames process...", color='g')
  # batch frames
  l.start_timer('FFmpegWriter', section='FFmpegWriter')
  l.start_timer('FFmpegWriter_create', section='FFmpegWriter')
  out2 = FFmpegWriter(
    filename=os.path.join(l.get_output_folder(), 'test_ffmpeg2.mp4'), 
    fps=fps, 
    frameSize=(width, height),
    log=l,
  )  
  l.stop_timer('FFmpegWriter_create', section='FFmpegWriter')  
  l.start_timer('FFmpegWriter_batch_write', section='FFmpegWriter')
  out2.write(frames)
  l.stop_timer('FFmpegWriter_batch_write', section='FFmpegWriter')
  
  l.start_timer('FFmpegWriter_release', section='FFmpegWriter')
  out2.release()
  l.stop_timer('FFmpegWriter_release', section='FFmpegWriter')
  l.stop_timer('FFmpegWriter', section='FFmpegWriter')  

  l.P("Done batch frames process.", color='g')  
  
  if VS_CV2:
    l.start_timer('CV2Writer')
    l.start_timer('CV2Writerr_create')
    out2 = cv2.VideoWriter(
      filename='test_cv2.mp4', 
      fps=fps, 
      fourcc=cv2.VideoWriter_fourcc(*"XVID"),
      frameSize=(width, height),
    )
    l.stop_timer('CV2Writerr_create')
    
    for i in range(TIME*60):
      # Build synthetic image for testing ("render" a video frame).
      img = np.full((height, width, 3), 60, np.uint8)
      cv2.putText(img, str(i+1), (width//2-100*len(str(i+1)), height//2+100), cv2.FONT_HERSHEY_DUPLEX, 10, (255, 30, 30), 20)  # Blue number
      l.start_timer('CV2Writer_write')
      out2.write(img)
      l.stop_timer('CV2Writer_write')
      if (i % 120) == 0:
        l.P("Status: {}".format(out2.isOpened()), color='g')
    l.start_timer('CV2Writer_release')
    out2.release()
    l.stop_timer('CV2Writer_release')
    l.stop_timer('CV2Writer')

  l.show_timers(title=l.get_machine_name())
  
      
      
  
  
