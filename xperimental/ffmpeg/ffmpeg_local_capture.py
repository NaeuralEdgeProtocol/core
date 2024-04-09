# start DELETE
from typing import Any
import cv2
# end DELETE

import subprocess as sp
import numpy as np
import re
import json

from io import BufferedReader
from collections import deque
from threading import Thread

URLS = [
  "__URL__",
  "__URL__",
  "__URL__",
]

class LogReader():
  def __init__(self, log_reader):
    self.log_reader : BufferedReader = log_reader
    self.__buffer = ""
    self.lines = deque(maxlen=1000)
    self.done = False
    self.__metadata_ready = False
    self.__metadata_buffer = ""
    self.__metadata = {
      'encoding': None,
      'fps': None,
      'frame_w': None,
      'frame_h': None      
    }
    self.start()
    return

  def read_log(self):
    result = None
    if len(self.lines) > 0:
      result = self.lines.popleft()
    elif len(self.__buffer) > 0:
      result = self.__buffer
      self.__buffer = ""
    return result

  def start(self):
    self.thread = Thread(target=self._run)
    self.thread.start()
    return
  
  def stop(self):
    self.done = True
    return
    
  def _run(self):
    while not self.done:
      text = self.log_reader.read(100)
      if text:
        self.__buffer += text.decode('utf-8')
        if not self.__metadata_ready:
          self.__metadata_buffer += text.decode('utf-8')
          self.__process_metadata()
        if '\n' in self.__buffer:
          lines = self.__buffer.split('\n')
          self.buffer = lines[-1]
          self.lines += lines[:-1]
      else:
        break
    return
  

  def parse_ffmpeg_log(self, log_line):
    pattern = r"Video: (\w+) .*, (\d+x\d+), (\d+(?:\.\d+)?) fps"
    match = re.search(pattern, log_line)
    if match:
      encoding = match.group(1)
      resolution = match.group(2)
      fps = match.group(3)
      return encoding, resolution, fps
    else:
      return None, None, None

  
  def __process_metadata(self):
    if 'Video:' in self.__metadata_buffer:      
      encoding, resolution, fps = self.parse_ffmpeg_log(self.__metadata_buffer)
      self.__metadata['encoding'] = encoding
      self.__metadata['fps'] = fps
      try:
        self.__metadata['frame_w'] = int(resolution.split('x')[0])
        self.__metadata['frame_h'] = int(resolution.split('x')[1])
      except:
        pass
      self.__metadata_ready = True
    return 
  
  def get_metadata(self):
    return self.__metadata

if __name__ == '__main__':
  
  # SETUP YOUR RTSP URL HERE
  rtsp_url = URLS[2]

  # Desired resolution
  width = 1280
  height = 720
  
  LOG_ON_SCREEN = False

  # FFmpeg command with parametrized resolution
  command = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',  # Force TCP transport for RTSP
    '-i', rtsp_url,
    '-f', 'image2pipe',
    '-s', f'{width}x{height}',  # Set the resolution
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    # '-stats_period', '120',
    '-'
  ]

  if LOG_ON_SCREEN:
    proc = sp.Popen(
      command, 
      stdout=sp.PIPE, 
      bufsize=3 * width * height
    )  
  else:
    proc = sp.Popen(
      command, 
      stdout=sp.PIPE, 
      stderr=sp.PIPE, 
      bufsize=3 * width * height
    )
    log_reader = LogReader(proc.stderr)

  video_reader = proc.stdout
  metadata_shown = False
  frame_count = 0
  
  while True:
    raw_image = video_reader.read(width * height * 3)  
    
    if not LOG_ON_SCREEN:
      log_line = log_reader.read_log()
      if log_line is not None:
        print(log_line)
    
    if not raw_image:
      break
    
    bgr_frame = np.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))  
  
    rgb_frame = np.ascontiguousarray(bgr_frame[:, :, ::-1])
    
    if not metadata_shown and frame_count > 0:
      metadata_shown = True
      metadata = log_reader.get_metadata()
      print("Stream working with following metadata:\n", json.dumps(metadata, indent=2))
      
        
    cv2_frame = bgr_frame
    cv2.imshow('Frame', cv2_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
  proc.terminate()
  cv2.destroyAllWindows()      
