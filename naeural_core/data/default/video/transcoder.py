"""
TODO: Refactor in order to have a abstract video stream than can:
  - resize & cropping
  - stream splitting (virtual stream) so we can spawn multiple streams from same one
  - unify all video stream under this abstract video stream

"""
import sys

import numpy as np
from naeural_core import constants as ct

from datetime import datetime
from time import sleep
from naeural_core.data.base import DataCaptureThread
from naeural_core.data.mixins_libs import _VideoConfigMixin

from naeural_core.utils.system_shared_memory import NumpySharedMemory

_CONFIG = {
  **DataCaptureThread.CONFIG,
  
  'RECONNECT_AFTER_FAILS' : 1000,
  
  # the cap resolution is NOT determined by the DCT but by the producer of data
  # so we set the default cap resolution to a pretty large number
  'CAP_RESOLUTION' : 100, 
  
  
  'VALIDATION_RULES' : {
    **DataCaptureThread.CONFIG['VALIDATION_RULES'],
    
    'RECONNECT_AFTER_FAILS' :{
      'TYPE' : 'int',
      'MIN_VAL' : 100,
      'MAX_VAL' : 5000,
    }
    
  },
}

class TranscoderDataCapture(DataCaptureThread, _VideoConfigMixin):
  CONFIG = _CONFIG
  def __init__(self, **kwargs):
    self._null_frames = 0
    self._first_null_frame_time = None

    self._in_mem_name = None
    self._in_mem_ctrl_name = None
    self._height, self._width, self._size = None, None, None
    self._crt_frame = 0
    self.shm_data = None
    self.shm_ctrl = None
    self._null_reads = 0  # this tracks read fails after each open (will reset after open)
    self._cont_fails = 0  # this tracks overall continous fails since last one (will reset after good read)
    self._reconnects = 0  # this tracks all reconnects since last connection is down (will reset after good read)
    self._lifetime_reconnects = 0 # tracks lifetime reconnects
    self._first_null_frame_time = None # timestamp of the first null frame in current series

    super(TranscoderDataCapture, self).__init__(**kwargs)
    return

  def startup(self):
    super().startup()
    self._metadata.update(
      fps=None,
      frame_h=None,
      frame_w=None,
      frame_count=None,
      frame_current=None,
      download=None,
    )

    self._in_mem_name = self.cfg_url
    self._in_mem_ctrl_name = self._in_mem_name + '_ctrl'
    self._height = self.cfg_stream_config_metadata[ct.TRANSCODER_H]
    self._width  = self.cfg_stream_config_metadata[ct.TRANSCODER_W]
    self._size = self._height * self._width * 3

    # initial connect
    self._open_shmem()
    self._maybe_configure_frame_crop_resize(
      height=self._height, 
      width=self._width,
    )
    return
  
  def _open_shmem(self):
    # prepare the memory buffer
    self.P("Initializing image shmem ndarray", color='y')
    self.shm_data = NumpySharedMemory(
      mem_name=self._in_mem_name, 
      mem_size=self._size, 
      np_shape=(self._height, self._width, 3), 
      np_type=np.uint8, 
      create=False,
      log=self.log
    )
    
    self.P("Initializing ctrl shmem ndarray", color='y')
    self.shm_ctrl = NumpySharedMemory(
      mem_name=self._in_mem_ctrl_name, 
      mem_size=4, 
      np_shape=(1,), 
      np_type=np.uint32, 
      create=False,
      log=self.log,
    )

    if (not self.shm_data.initialized) or (not self.shm_ctrl.initialized):
      err1 = self.shm_data.error_message
      err2 = self.shm_ctrl.error_message    
      msg = "ERROR in ShareMemory('{}')".format(self._in_mem_name,)
      info = "Image:{}\n Ctrl:{}".format(err1, err2)
      self.P(msg + "\n" + info, color='r')
      msg_type = ct.STATUS_TYPE.STATUS_EXCEPTION
    else:
      msg = "SharedMemory '{}' succesfully initialized (This does not imply data is available)".format(self._in_mem_name)
      msg_type = ct.STATUS_TYPE.STATUS_NORMAL
      info = None
      self.P(msg, color='g')
    #endif

    self._create_notification(
      notif=msg_type,
      msg=msg,
      info=info,
      displayed=True,
    )
            
    return
    
  def _init(self):
    #
    return
  
  def _maybe_reconnect(self):
    # initial connect was done in self.startup()
    if self._null_reads > self.cfg_reconnect_after_fails:
      self.P("{} re-connecting to shared memory due to {} continuous fails".format(
        self.__class__.__name__, self._null_reads,
        ), color='r')
      self._release()
      self._open_shmem()
      self._null_reads = 0   
      self._reconnects += 1  
      self._lifetime_reconnects += 1
    return
  
  def _run_data_aquisition_step(self):
    self.start_timer('read_shm_ctrl')
    frame = None
    # check if data is avalable
    ctrl_data = self.shm_ctrl.read()
    transcoder_count = int(ctrl_data[0]) if ctrl_data is not None else 0
    self.end_timer('read_shm_ctrl')
    
    if transcoder_count > 0: 
      self.start_timer('read_shm_img')
      self._crt_frame = transcoder_count
      # copy from shm 
      frame = self.shm_data.read()
      self.end_timer('read_shm_img')
      
    if frame is not None:
      self.start_timer('write_shm_ctrl')
      # reset control
      self.shm_ctrl.write(0)
      # prepare data
      self._null_reads = 0 # good frame, reset fails since reconnect
      self._cont_fails = 0 # good frame, reset continuous fails
      self._reconnects = 0 # reset continous reconns (but not lifetime)
      
      self._metadata.frame_current = self._crt_frame
      self.end_timer('write_shm_ctrl')
      
      ### universal video stream code
      frame = self._maybe_resize_crop(frame)
      ### end universal video stream code
        
      self._add_inputs(
        [
          self._new_input(img=frame, metadata=self._metadata.__dict__.copy())
        ]
      )
      # no need for sleep as it is implemented in main thread loop
    else:
      if self._cont_fails == 0:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # we do not use Logger for more thread safety
        self._first_null_frame_time = now_str
      self._null_reads += 1
      self._cont_fails += 1
      sleep(0.100) # wait as it seems there is a problem anyway
      if (self._cont_fails % (self.cfg_reconnect_after_fails * 10)) == 0:
        msg = "Received {} NULL reads continously since {} ({} since last reconnect attempt). {} reconnect attempts since last connection lost, {} lifetime reconn attempts".format(
          self._cont_fails,
          self._first_null_frame_time,
          self._null_reads,
          self._reconnects,
          self._lifetime_reconnects,
        )
        self.P(msg, color='r')
        self._create_notification(
          notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
          msg=msg,
          displayed=True,
        )
        self.P("DCT sleeping for 20 seconds as multiple fails occured...")
        sleep(20)
      #endif send notification
    #endif if data or no data
    return
  
  def _release(self):
    try:
      self.P("{} closing shmem '{}'...".format(
        self.__class__.__name__, self._in_mem_name))
      if self.shm_data is not None:
        self.shm_data.shutdown()
      if self.shm_ctrl is not None:
        self.shm_ctrl.shutdown()
      
    except:
      msg = '{} shmem release failed'.format(self.__class__.__name__)
      info = "{}".format(sys.exc_info()[0])
      self.P(msg + "\n" + info, color='error')
      self._create_notification(
          notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
          msg=msg,
          info=info
      )
    return
