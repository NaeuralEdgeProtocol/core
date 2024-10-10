import re
import subprocess as sp
from io import BufferedReader
from threading import Thread

from naeural_core.data.base import DataCaptureThread
from naeural_core.data.mixins_libs import _VideoConfigMixin
from naeural_core.utils.plugins_base.plugin_base_utils import LogReader
from naeural_core.utils.thread_raise import ctype_async_raise

_CONFIG = {
  **DataCaptureThread.CONFIG,

  "INITIAL_MAX_TIME_NO_READ": 15,  # seconds
  "MAX_TIME_NO_READ": 3,  # seconds
  "STATS_PERIOD": 120,
  "LOCK_RESOURCE": "ffmpeg-pipe",
  "SHOW_FFMPEG_LOG": False,
  "USE_LOCK_WHEN_RECONNECTING": False,

  "MAX_RETRIES": 2,

  "FRAME_H": None,  # 720,
  "FRAME_W": None,  # 1280,

  "USER": None,
  "PASSWORD": None,

  "FFMPEG_GENERAL_COMMAND_PARAMETERS": {},
  "DEFAULT_FFMPEG_GENERAL_COMMAND_PARAMETERS": {
    'fflags': '+discardcorrupt',  # removed flags  'nobuffer', 'flush_packets'
    'vcodec': 'rawvideo',
  },

  "FFMPEG_STREAM_COMMAND_PARAMETERS": {},
  "DEFAULT_FFMPEG_STREAM_COMMAND_PARAMETERS": {
    'rtsp_transport': 'tcp',  # Force TCP transport for RTSP | Maybe experiment further with UDP
    'max_delay': '500000',
    'analyzeduration': '20M',
    'probesize': '20M',
    'threads': '2',
  },

  "URL": "rtsp://your_rtsp_url",

  'VALIDATION_RULES': {
    **DataCaptureThread.CONFIG['VALIDATION_RULES'],
  },
}


class FfmpegLogReader(LogReader):
  def __init__(self, owner, buff_reader):
    self.__metadata_ready = False
    self.__metadata_buffer = ""
    self.__metadata = {
      'encoding': None,
      'fps': None,
      'frame_w': None,
      'frame_h': None
    }
    super(FfmpegLogReader, self).__init__(owner=owner, buff_reader=buff_reader, size=100)
    return

  def __parse_ffmpeg_log(self, log_line):
    pattern = r"Video: (\w+) .*, (\d+x\d+).*, (\d+(?:\.\d+)?) fps"
    match = re.search(pattern, log_line)
    if match:
      encoding = match.group(1)
      resolution = match.group(2)
      fps = match.group(3)
      return encoding, resolution, fps
    else:
      return None, None, None

  def __process_metadata(self):
    if 'Video:' in self.__metadata_buffer and "fps" in self.__metadata_buffer:
      encoding, resolution, fps = self.__parse_ffmpeg_log(self.__metadata_buffer)
      self.__metadata['encoding'] = encoding
      self.__metadata['fps'] = fps
      try:
        frame_w, frame_h = resolution.split('x')[:2]
        self.__metadata['frame_w'] = int(frame_w)
        self.__metadata['frame_h'] = int(frame_h)
      except:
        pass
      self.__metadata_ready = True
    return

  def on_text(self, text):
    super().on_text(text)
    if not self.__metadata_ready:
      self.__metadata_buffer += text.decode('utf-8')
      self.__process_metadata()
    return

  # Public methods
  def get_metadata(self):
    return self.__metadata if self.__metadata_ready else None

  def read_log(self):
    buf = self.get_next_characters()
    if len(buf) == 0:
      return None
    return buf


class FfmpegFrameReader(LogReader):
  def __init__(self, owner, buff_reader, frame_size):
    super(FfmpegFrameReader, self).__init__(owner=owner, buff_reader=buff_reader, size=frame_size)
    self.frame_size = frame_size
    self.frame_buffer = b""
    self.frames_deques = owner.deque(maxlen=1)
    self.__full_frames = 0
    self.__part_frames = 0
    return

  # Public methods

  @property
  def n_full_frames(self):
    return self.__full_frames

  @property
  def n_part_frames(self):
    return self.__part_frames

  def read_frame(self):
    self.frame_buffer += self.get_next_characters(decode=False)
    while len(self.frame_buffer) >= self.frame_size:
      self.__full_frames += 1
      self.frames_deques.append(self.frame_buffer[:self.frame_size])
      self.frame_buffer = self.frame_buffer[self.frame_size:]
    # end while

    frame = None
    if len(self.frames_deques) > 0:
      frame = self.frames_deques.popleft()
    return frame


class VideoStreamFfmpegDataCapture(DataCaptureThread, _VideoConfigMixin):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self._capture = None
    self._ffmpeg_process = None
    self._ffmpeg_frame_reader = None
    self._ffmpeg_log_reader = None
    self._crt_frame = 0
    self._last_read_time = 0

    self.last_url = None
    self.last_user = None
    self.last_password = None
    self.last_downloaded_file = None
    self.last_hw = None

    self._dct_ffmpeg_errors = {
      'dup': 0,  # Duplicated frames
      'drop': 0,  # Dropped frames
      'dec': 0,  # Decoding errors
      'cor': 0,  # Corrupt frames
      'miss': 0,  # Missed frames
      'con': 0,  # Concealing errors
      'delay': 0,  # Max delay
    }
    super(VideoStreamFfmpegDataCapture, self).__init__(**kwargs)
    return

  def startup(self):
    super().startup()
    self._metadata.update(
      fps=None,
      frame_h=None,
      frame_w=None,
      frame_count=None,
      frame_current=None,
    )
    return

  def get_configured_hw(self):
    return self.cfg_frame_h, self.cfg_frame_w

  def _release(self):
    if self._ffmpeg_frame_reader is not None:
      self._ffmpeg_frame_reader.stop()

    if self._ffmpeg_log_reader is not None:
      self._ffmpeg_log_reader.stop()

    if self._ffmpeg_process is not None:
      self._ffmpeg_process.kill()

      # timeout 5 seconds
      WAIT_TIME = 5

      start_wait_time = self.time()
      while self.time() - start_wait_time < WAIT_TIME and self._ffmpeg_process is not None:
        try:
          result = self._ffmpeg_process.wait(1)
          if result is not None:
            break
        except sp.TimeoutExpired:
          continue
        finally:
          if self.cfg_show_ffmpeg_log:
            self.P("Waiting for ffmpeg process to exit...")
      # end while
      if self.cfg_show_ffmpeg_log:
        self.P("FFmpeg process exited with code {}".format(self._ffmpeg_process.returncode))

    if self._ffmpeg_process is not None:
      del self._ffmpeg_process.stderr
      del self._ffmpeg_process.stdout
      del self._ffmpeg_process

    self._ffmpeg_process = None
    self._ffmpeg_frame_reader = None
    self._ffmpeg_log_reader = None
    return

  def _init(self):
    self._force_video_stream_config()
    return

  def __is_video_stream(self, url):
    is_rtsp = 'rtsp://' in url
    is_rtsps = 'rtsps://' in url
    return is_rtsp or is_rtsps

  def _get_url(self):
    """
    Get the url for the video stream. If the url is a file, we return the path to the file.

    Returns
    -------
    url, is_file: str, bool
      The url and a flag that indicates if the url is a file
    """
    # TODO: check if there are other cases for live streams
    if self.__is_video_stream(self.cfg_url):
      return self.cfg_url, False

    if self.last_downloaded_file is not None:
      return self.last_downloaded_file, True

    # we assume that the url is a direct link to a video file
    filepath = self.maybe_download(
      url=self.cfg_url,
      fn=self.now_str(),
      target="data"
    )

    if filepath is None:
      # the assumption is false, nothing was downloaded
      # we return the url as is, and let ffmpeg handle it
      return self.cfg_url, False

    filepath = self.os_path.abspath(filepath)
    self.last_downloaded_file = filepath

    return filepath, True

  def _maybe_edit_user_password_in_url(self, url):
    parsed_url = self.urlparse(url)

    old_user, old_password = parsed_url.username, parsed_url.password

    host, port = parsed_url.hostname, parsed_url.port

    user = self.cfg_user if self.cfg_user is not None else old_user
    password = self.cfg_password if self.cfg_password is not None else old_password

    self.last_user = user
    self.last_password = password

    self.P(f"User: {old_user} -> {user}")
    self.P(f"Password: {old_password} -> {password}")

    new_url = self.urlunparse

    new_netloc = f"{user}:{password}@{host}"
    new_netloc = f"{new_netloc}:{port}" if port is not None else new_netloc

    new_url = self.urlunparse((parsed_url.scheme, new_netloc, parsed_url.path,
                              parsed_url.params, parsed_url.query, parsed_url.fragment))

    self.P(f"New URL: {new_url}")

    return new_url

  def _create_ffmpeg_subprocess(self):

    url, is_file = self._get_url()

    ffmpeg_command = ['ffmpeg']

    if is_file:
      ffmpeg_command += [
        '-re',  # Read input at native frame rate. Mainly used to simulate a grab device
      ]
    else:
      url = self._maybe_edit_user_password_in_url(url)

    dct_general_command_parameters = {
      **self.cfg_default_ffmpeg_general_command_parameters,
      **self.cfg_ffmpeg_general_command_parameters,
    }

    lst_general_command_parameters = [[f"-{k}", str(v)] for k, v in dct_general_command_parameters.items()]
    general_command_parameters = []
    for lst in lst_general_command_parameters:
      general_command_parameters += lst

    ffmpeg_command += [
        '-i', url,  # f"\"{url}\"",
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-stats_period', str(self.cfg_stats_period),
        # do not ignore errors because we want to see pleasant video stream
        # from documentation, "This option will not result in a video that is pleasing to watch in case of errors"
        # '-err_detect', 'ignore_err',

        # if discarding corrupt frames for more than MAX_TIME_NO_READ seconds, we will reconnect
        # we want to discard corrupt frames because the video stream will contain duplicate frames
        *general_command_parameters,
    ]

    if not is_file:

      dct_stream_command_parameters = {
        **self.cfg_default_ffmpeg_stream_command_parameters,
        **self.cfg_ffmpeg_stream_command_parameters,
      }

      lst_stream_command_parameters = [[f"-{k}", str(v)] for k, v in dct_stream_command_parameters.items()]
      stream_command_parameters = []
      for lst in lst_stream_command_parameters:
        stream_command_parameters += lst

      ffmpeg_command += [
        *stream_command_parameters,
        '-r', str(self.cfg_cap_resolution),
      ]

    # set scaling resolution (we move the scaling to ffmpeg)
    frame_h, frame_w = self.get_configured_hw()
    if frame_h is not None and frame_w is not None:
      ffmpeg_command += [
        '-s', f'{frame_w}x{frame_h}',  # Set the resolution
      ]
    # endif

    ffmpeg_command += [
        '-'
    ]

    if self.cfg_show_ffmpeg_log:
      self.P("Starting ffmpeg process with command: {}".format(" ".join(ffmpeg_command)))

    self._ffmpeg_process = sp.Popen(ffmpeg_command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 7)
    self._ffmpeg_log_reader = FfmpegLogReader(owner=self, buff_reader=self._ffmpeg_process.stderr)

    initial_time = self.time()
    while self.time() - initial_time < self.cfg_initial_max_time_no_read:
      self.sleep(0.01)
      metadata = self._ffmpeg_log_reader.get_metadata()

      if metadata is not None:
        break

    # end while read metadata
    if metadata is None:
      self.P("Failed to read metadata from ffmpeg log. Most likely the stream did not start.", color='r')
      return

    # we do not want to display the metadata in the logs, so we remove it from the log queue
    # we do this to reduce the number of logs we print
    if not self.cfg_show_ffmpeg_log:
      while self._ffmpeg_log_reader.read_log() is not None:
        self.sleep(0.01)

    frame_h, frame_w = metadata['frame_h'], metadata['frame_w']
    configured_h, configured_w = self.get_configured_hw()

    height = frame_h if configured_h is None else configured_h
    width = frame_w if configured_w is None else configured_w

    self._metadata.update(**metadata)

    self._ffmpeg_frame_reader = FfmpegFrameReader(
      owner=self,
      buff_reader=self._ffmpeg_process.stdout,
      frame_size=height * width * 3
    )
    return

  def _on_config_changed(self):
    if self.cfg_url != self.last_url:
      self.has_connection = False  # forces reconnection
      self.last_downloaded_file = None
      self.P("RTSP URI change detected from `{}` to `{}`, reconnecting...".format(
        self.last_url, self.cfg_url), color='r')

    if self.get_configured_hw() != self.last_hw:
      self.has_connection = False  # forces reconnection
      self.P("Video stream acquisition resolution changed from `{}` to `{}`, reconnecting...".format(
        self.last_hw, self.get_configured_hw()), color='r')

    # Trigger a reconnect if the user changes and the url is a video stream
    if self.cfg_user != self.last_user and self.__is_video_stream(self.cfg_url):
      self.has_connection = False  # forces reconnection
      self.P("User change detected from `{}` to `{}`, reconnecting...".format(
        self.last_user, self.cfg_user), color='r')

    # Trigger a reconnect if the password changes and the url is a video stream
    if self.cfg_password != self.last_password and self.__is_video_stream(self.cfg_url):
      self.has_connection = False
      self.P("Password change detected from `{}` to `{}`, reconnecting...".format(
        self.last_password, self.cfg_password), color='r')
    return

  def _read_frame(self):
    frame = self._ffmpeg_frame_reader.read_frame()
    if frame is not None:
      configured_h, configured_w = self.get_configured_hw()
      frame_h, frame_w = self._metadata.frame_h, self._metadata.frame_w
      height = frame_h if configured_h is None else configured_h
      width = frame_w if configured_w is None else configured_w
      try:
        frame = self.np.frombuffer(frame, dtype='uint8').reshape((height, width, 3))
        # frame = self.np.ascontiguousarray(bgr_frame[:, :, ::-1])
      except:
        frame = None
      # endif

    if frame is not None:
      self._last_read_time = self.time()
    return frame

  def _maybe_reconnect(self):
    if self.has_connection and self.time() - self._last_read_time < self.cfg_max_time_no_read:
      return

    self.last_url = self.cfg_url  # a cache of current url
    self.last_hw = self.get_configured_hw()  # a cache of current hw
    self.has_connection = False

    self._dct_ffmpeg_errors = {
      'dup': 0,  # Duplicated frames
      'drop': 0,  # Dropped frames
      'dec': 0,  # Decoding errors
      'cor': 0,  # Corrupt frames
      'miss': 0,  # Missed frames
      'con': 0,  # Concealing errors
      'delay': 0,  # Max delay
    }
    nr_retry = 0
    if self.nr_connection_issues == 0:
      msg = "Connecting to url:"
      color = 'g'
    else:
      msg = "Reconnecting ({}) to url (last read {}/{} seconds):".format(
        self.nr_connection_issues,
        self.time() - self._last_read_time,
        self.cfg_max_time_no_read,
      )
      color = 'r'
    self.P("{} {} (deq size: {})".format(msg, self.cfg_url, self._deque.maxlen), color=color)
    str_e = None
    while nr_retry <= self.cfg_max_retries:
      self.sleep(1)
      nr_retry += 1
      self.nr_connection_issues += 1
      with self.managed_lock_resource(self.cfg_lock_resource, condition=(str(self.cfg_use_lock_when_reconnecting).upper() == 'TRUE')):
        try:
          if self._ffmpeg_process:
            self._release()
          self.P("  Connection retry {}/{} of connect session {}, total retries {}:".format(
            nr_retry, self.cfg_max_retries, self.nr_connection_issues // self.cfg_max_retries, self.nr_connection_issues,
          ))

          self._create_ffmpeg_subprocess()

          if self._ffmpeg_frame_reader is not None:
            frame = None
            initial_time = self.time()
            while frame is None and self.time() - initial_time < self.cfg_initial_max_time_no_read:
              self.sleep(0.01)
              frame = self._read_frame()
            # end while

            if frame is None:
              self.P("    Capture read failed.", color='r')
              continue

            # We comment this out because we can set ffmpeg to scale the image for us
            height, width = frame.shape[:2]

            # universal video stream code
            self._maybe_configure_frame_crop_resize(
              height=height,
              width=width,
            )
            # end universal video stream code

            self.has_connection = True
        except:
          str_e = self.trace_info()
          self.P('`_maybe_reconnect` exception: {}'.format(str_e), color='r')
        # end try-except
      # endwith conditional lock

      if self.has_connection:
        break
    # endwhile

    # random sleep time to de-sync broken streams
    sleep_time = self.np.random.randint(self.cfg_sleep_time_on_error, self.cfg_sleep_time_on_error * 3)

    if self.has_connection:
      self.reset_received_first_data()
      msg = "Ffmpeg DCT '{}' successfully connected. Overall {} reconnects.".format(
        self.cfg_name,
        self.nr_connection_issues,
      )
      notification_type = self.ct.STATUS_TYPE.STATUS_NORMAL
      color = 'g'
    else:
      msg = "Abnormal functioning of Ffmpeg DCT '{}' failed after {} retries, overall {} reconnects. Sleeping DCT for {:.1f}s".format(
        self.cfg_name, self.cfg_max_retries,
        self.nr_connection_issues,
        sleep_time,
      )
      notification_type = self.ct.STATUS_TYPE.STATUS_ABNORMAL_FUNCTIONING
      color = 'e'
    # endif
    self.P(msg, color=color)
    self._create_notification(
      notif=notification_type,
      msg=msg,
      info=str_e,
      video_stream_info=self.deepcopy(self._metadata.__dict__),
      displayed=True,
    )

    if not self.has_connection:
      # execute sleep after notification delivery
      self.sleep(sleep_time)
    return

  def _process_ffmpeg_log(self, log_line):
    dct_key = None
    dct_value = None
    if 'error while decoding' in log_line:
      dct_key = 'dec'
      dct_value = 1
    elif 'corrupt' in log_line:
      dct_key = 'cor'
      dct_value = 1
    elif 'RTP: missed' in log_line:
      dct_key = 'miss'
      dct_value = re.compile(r'RTP: missed (\d+) packets').search(log_line)
      if dct_value is not None:
        dct_value = int(dct_value.group(1))
    elif 'concealing' in log_line:
      dct_key = 'con'
      dct_value = 1
    elif 'max delay' in log_line:
      dct_key = 'delay'
      dct_value = 1
    if 'dup=' in log_line:
      dct_key = 'dup'
      dct_value = re.compile(r'dup=(\d+)').search(log_line)
      if dct_value is not None:
        dct_value = int(dct_value.group(1))
      if dct_value is None or dct_value < self._dct_ffmpeg_errors['dup']:
        dct_value = 0
      else:
        dct_value = dct_value - self._dct_ffmpeg_errors['dup']
    if 'drop=' in log_line:
      dct_key = 'drop'
      dct_value = re.compile(r'drop=(\d+)').search(log_line)
      if dct_value is not None:
        dct_value = int(dct_value.group(1))
      if dct_value is None or dct_value < self._dct_ffmpeg_errors['drop']:
        dct_value = 0
      else:
        dct_value = dct_value - self._dct_ffmpeg_errors['drop']

    if dct_key is not None:
      if dct_value is None:
        dct_value = 1
      self._dct_ffmpeg_errors[dct_key] += dct_value
    return dct_key is None and 'frame=' in log_line

  def _run_data_aquisition_step(self):
    if self.has_connection:
      frame = self._read_frame()

      log_line = self._ffmpeg_log_reader.read_log()

      if log_line is not None and self._process_ffmpeg_log(log_line) and self.cfg_show_ffmpeg_log:
        self.P("FFmpeg log: {}".format(log_line.strip()))

      if frame is not None:

        # universal video stream code
        frame = self._maybe_resize_crop(frame)
        # end universal video stream code

        # metadata
        self._crt_frame += 1
        self._metadata.frame_current = self._crt_frame
        self._metadata.frame_n_full = self._ffmpeg_frame_reader.n_full_frames
        self._metadata.frame_n_partial = self._ffmpeg_frame_reader.n_part_frames

        self._add_img_input(frame)
    else:
      self.P('Capture seems to be closed. Setting `has_connection=False`', color='r')
      self.has_connection = False
    return

  def get_plugin_specific_stats(self):
    return self._dct_ffmpeg_errors
