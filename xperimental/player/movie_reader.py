import cv2

from collections import deque
from naeural_core import DecentrAIObject
from naeural_core import Logger

class SimpleMovieReader(DecentrAIObject):
  def __init__(self, path_file, **kwargs):
    self._path_file = path_file    
    super().__init__(**kwargs)
    return
  
  def startup(self):
    super().startup()
    self._capture = None
    self._idx_selected = 0
    self._idx_read = 0
    self._buffer_size = 500
    self._buffer = deque(maxlen=self._buffer_size)
    self._init_capture()
    return
  
  def _get_number_of_frames(self):
    i = 0
    while True:
      has_frame = self._capture.grab()
      if not has_frame:
        break
      i+= 1
    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return i
  
  def _init_capture(self):
    if self._capture:
      self._capture.release()
    
    self._capture  = cv2.VideoCapture(self._path_file)
    self._fps    = int(self._capture.get(cv2.CAP_PROP_FPS))
    self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self._width  = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    self._frame_count = self._get_number_of_frames()
    return
  
  def _read(self, steps):
    steps = max(1, steps)
    self.log.p('Reading {} frames'.format(steps))
    for _ in range(steps):
      has_frame, frame = self._capture.read()
      if has_frame:
        self._idx_read+= 1
        self._idx_selected = self._idx_read
        self._buffer.append((self._idx_read, frame))
    return has_frame, frame
  
  def _read_from_buffer(self, searched_idx):
    has_frame, frame = None, None
    buf = {i:f for i,f in self._buffer}
    if searched_idx in buf:
      has_frame, frame = True, buf[searched_idx]
      self._idx_selected = searched_idx
    return has_frame, frame
  
  def _refill_buffer(self, steps):
    self.log.p('Need to re-initialize movie in order to provide frame {}'.format(steps))
    self._buffer = deque(maxlen=self._buffer_size)
    self._idx_read = 0
    self._idx_selected = 0
    self._init_capture()
    self._read(steps=steps)
    return
  
  def read_next(self):
    searched_idx = self._idx_selected + 1
    has_frame, frame = self._read_from_buffer(searched_idx)
    if not has_frame:
      has_frame, frame = self._read(steps=1)
    return has_frame, frame
  
  def read_previous(self):
    searched_idx = max(1, self._idx_selected - 1)
    has_frame, frame = self._read_from_buffer(searched_idx)    
    if not has_frame:
      self._refill_buffer(steps=searched_idx)
    
    has_frame, frame = self._read_from_buffer(searched_idx)
    return has_frame, frame 
  
  def skip_forward(self, steps):
    searched_idx = min(self._idx_selected + steps, self._frame_count)
    has_frame, frame = self._read_from_buffer(searched_idx)
    if not has_frame:
      has_frame, frame = self._read(steps=steps)
    return has_frame, frame
  
  def skip_backward(self, steps):
    searched_idx = max(1, self._idx_selected - steps)
    has_frame, frame = self._read_from_buffer(searched_idx)
    if not has_frame:
      self._refill_buffer(steps=searched_idx)
    has_frame, frame = self._read_from_buffer(searched_idx)
    return has_frame, frame
  
  def get_idx_selected(self):
    return self._idx_selected
  
  def get_frame_count(self):
    return self._frame_count
  

if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='CAVI', 
    config_file=cfg_file, 
    max_lines=1000, 
    TF_KERAS=False
    )
  
  path = 'C:/Users/ETA/Dropbox/MKT/Blur Alex/CD render/Intrare magazin.avi'
  smr = SimpleMovieReader(
    log=log,
    path_file=path
    )
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  