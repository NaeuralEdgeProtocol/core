"""
This plugin can be put on any VideoStream pipeline and will monitor the delay between
consecutive frames received from the DCT. If this delay is bigger than a certain threshold
(ALERT_RAISE_VALUE), an alert will be raised. The alert will be lowered when the delay
is less than another threshold (ALERT_LOWER_VALUE) for a period of time
(ALERT_LOWER_CONFIRMATION_TIME seconds).

This plugin can discard duplicate frames, considering frame duplication as delay between
different frames.
"""

# local dependencies
from core.business.base import CVPluginExecutor as BaseClass

__VER__ = '1.0.0'

_CONFIG = {
  **BaseClass.CONFIG,

  'RUN_WITHOUT_IMAGE': True,
  'ALLOW_EMPTY_INPUTS': True,
  'ADD_ORIGINAL_IMAGE': False,

  'MAX_INPUTS_QUEUE_SIZE': 1,
  "PROCESS_DELAY": 0.04,

  'ALERT_DATA_COUNT'              : 1,
  'ALERT_RAISE_CONFIRMATION_TIME' : 0,
  'ALERT_LOWER_CONFIRMATION_TIME' : 3,
  'ALERT_RAISE_VALUE'             : 5, # video loss delay is usually 1-2 seconds, if it is more than 5 seconds, it is a problem
  'ALERT_LOWER_VALUE'             : 0.5,  # revert to normal when delay is less than 0.5 seconds
  'ALERT_MODE'                    : 'max',

  'DISCARD_DUPLICATES': True,

  'VALIDATION_RULES': {
    **BaseClass.CONFIG['VALIDATION_RULES'],
  },
}


class VideoLoss01Plugin(BaseClass):
  CONFIG = _CONFIG

  def startup(self):
    super().startup()

    self.__last_frame_timestamp = None
    self.__last_frame = None
    return

  def _process(self):
    current_time = round(self.time(), 2)

    img = self.dataapi_image()
    if img is not None:
      if self.cfg_discard_duplicates:
        if self.__last_frame is None or not self.np.equal(self.__last_frame, img).all():
          self.__last_frame = img
          self.__last_frame_timestamp = current_time
        # endif we have a new image
      else:
        self.__last_frame_timestamp = current_time
    # endif we have an image

    if self.__last_frame_timestamp is None:
      return
    # endif we did not analyze any image yet

    time_since_last_frame = round(current_time - self.__last_frame_timestamp, 2)
    self.alerter_add_observation(time_since_last_frame)
    
    if self.alerter_status_changed():
      self.create_and_send_payload(
        last_frame_timestamp=self.__last_frame_timestamp,
        last_frame_datetime=self.timestamp_to_str(self.__last_frame_timestamp),
        time_since_last_frame=time_since_last_frame,
      )

    return
