"""
This plugin can be pun on any VideoStream pipeline and will generate statistics
about the quality of the DCT processing.

The plugin will compute, for a given window of time, the maximum delay between
two consecutive frames received from the DCT.

When closed or finished, the plugin will save a CSV file with the following columns:
- timestamp: the timestamp of the window
- delay: the maximum delay in the window
"""

# local dependencies
from naeural_core.business.base import CVPluginExecutor as BaseClass

__VER__ = '1.0.0'

_CONFIG = {
  **BaseClass.CONFIG,

  'RUN_WITHOUT_IMAGE': True,
  'ALLOW_EMPTY_INPUTS': True,
  'MAX_INPUTS_QUEUE_SIZE': 1,

  "PROCESS_DELAY": 1 / 20,
  "ANALYSIS_WINDOW": 5,  # SECONDS

  "SHUTDOWN_AFTER_X_DAYS": 0.9,  # DAYS


  'VALIDATION_RULES': {
    **BaseClass.CONFIG['VALIDATION_RULES'],
  },
}


class CameraDelaysPlugin(BaseClass):
  CONFIG = _CONFIG

  def startup(self):
    super().startup()

    current_time = int(self.time())

    self.__window = [] # list of delays in the current window
    self.__last_window_time = current_time - current_time % self.cfg_analysis_window

    self.__lst_datapoints = []  # list of tuples (timestamp, delay)
    self.__last_image_timestamp = None

    self._saved_dataframe = False
    return

  def _process(self):
    current_time = round(self.time(), 2)
    if current_time - self.start_time > self.cfg_shutdown_after_x_days * 24 * 3600:
      self._maybe_save_dataframe()
      self.cmdapi_update_instance_config(*self.unique_identification, {'FORCED_PAUSE': True}, self.ee_addr)
    # endif we need to shutdown

    img = self.dataapi_image()
    if img is not None:
      self.__last_image_timestamp = current_time
    # endif we have an image

    if self.__last_image_timestamp is None:
      return
    # endif we did not analyze any image yet

    self.__window.append(round(current_time - self.__last_image_timestamp, 2))

    if current_time - self.__last_window_time > self.cfg_analysis_window:
      self.__lst_datapoints.append((self.__last_window_time, max(self.__window)))
      self.__last_window_time = int(current_time) - int(current_time) % self.cfg_analysis_window
      self.__window = []
    # endif we process the window
    return

  def _maybe_save_dataframe(self):
    if self._saved_dataframe:
      return
    # endif we already saved the dataframe

    now = self.now_str(short=True)

    df = self.pd.DataFrame(self.__lst_datapoints, columns=['timestamp', 'delay'])
    self.diskapi_save_dataframe_to_output(df, f'camera_delays_{self.get_stream_id()}_{now}.csv')
    self._saved_dataframe = True
    return

  def on_close(self):
    self._maybe_save_dataframe()
    return super().on_close()
