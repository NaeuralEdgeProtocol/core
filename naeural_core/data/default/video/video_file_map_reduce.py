import os
import uuid
import shutil
from naeural_core import constants as ct
from threading import Thread
from time import time
from datetime import timedelta

from naeural_core.data.base import AbstractMapReduceDataCapture
from naeural_core.data.mixins_libs import _VideoFileMixin
from naeural_core.local_libraries.vision.ffmpeg_utils import FFMPEGUtils

_CONFIG = {
  **AbstractMapReduceDataCapture.CONFIG,
  'VALIDATION_RULES' : {
    **AbstractMapReduceDataCapture.CONFIG['VALIDATION_RULES'],
  },
}

class ThreadWithReturnValue(Thread):
  def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, daemon=None):
    super(ThreadWithReturnValue, self).__init__(
      group=group, target=target,
      name=name, args=args, kwargs=kwargs,
      daemon=daemon
    )
    self._return = None
    return

  def run(self):
    try:
      if self._target:
        self._return = self._target(*self._args, **self._kwargs)
    finally:
      # Avoid a refcycle if the thread is running a function with
      # an argument that has a member that points to the thread.
      del self._target, self._args, self._kwargs
    return

  def join(self, timeout=None):
    super().join(timeout)
    return self._return

class VideoFileMapReduceDataCapture(AbstractMapReduceDataCapture, _VideoFileMixin):
  CONFIG = _CONFIG
  def __init__(self, **kwargs):
    self._upload_threads = []
    self._file_system_manager = None
    self._path_split_dir = None

    self._threads_upload = None
    self._upload_id = None
    self._video_file_duration = None
    self._video_file_segment_time = None
    self._paths_splitted_video_files = None
    self._ffmpeg_utils = None

    self._time_start_upload = None
    super(VideoFileMapReduceDataCapture, self).__init__(**kwargs)
    return

  def startup(self):
    super().startup()
    assert self.nr_chunks is not None
    self._ffmpeg_utils = FFMPEGUtils(caller=self)
    self._metadata.update(
      download=None,
      finished_split=False,
      split_elapsed_time=None,
      upload_elapsed_time=None,
      path=None,
      input_indexes=None,
      video_resolution=None,
    )
    self._file_system_manager = self.shmem['file_system_manager']
    return

  @property
  def cfg_delete_path(self):
    delete_path = False
    if not os.path.exists(str(self.video_file_url)):
      delete_path = True

    delete_path = self.cfg_stream_config_metadata.get('DELETE_PATH', delete_path)
    return delete_path

  @property
  def cfg_video_file_extension(self):
    return self.cfg_stream_config_metadata.get('VIDEO_FILE_EXTENSION', None)

  @property
  def cfg_use_local_system_to_save(self):
    return self.cfg_stream_config_metadata.get('USE_LOCAL_SYSTEM_TO_SAVE', True)

  @property
  def video_file_url(self):
    return self.cfg_url

  @property
  def has_threads_upload(self):
    return self._threads_upload is not None

  @property
  def alive_threads_upload(self):
    if not self.has_threads_upload:
      return False

    lst_bool_thr_is_alive = [thr.is_alive() for thr in self._threads_upload]
    return any(lst_bool_thr_is_alive)

  def _init_map(self):
    self._maybe_download()
    time_start_split = time()
    self._video_file_duration = self._ffmpeg_utils.video_file_duration(path=self._path)
    self._video_resolution = self._ffmpeg_utils.video_resolution(path=self._path)

    self._upload_id = str(uuid.uuid4())
    self._path_split_dir = os.path.split(self._path)[0] + '/{}'.format(self._upload_id)
    os.makedirs(self._path_split_dir, exist_ok=True)

    dct_ret = self._ffmpeg_utils.split_video_file(
      path=self._path,
      nr_chunks=self.nr_chunks,
      path_to_output=self._path_split_dir,
      duration=self._video_file_duration,
      default_ext=self.cfg_video_file_extension
    )

    self._metadata.split_elapsed_time = str(timedelta(seconds=time() - time_start_split))
    self._metadata.video_resolution = self._video_resolution

    output_files = dct_ret['output_files']
    self._video_file_segment_time = dct_ret['segment_time']

    self._metadata.finished_split = True
    self._paths_splitted_video_files = [os.path.join(self._path_split_dir, f) for f in output_files]

    indexes = []
    past_index = 1
    crt_index = 0
    for file_path in self._paths_splitted_video_files:
      self._reset_decord_video_reader(file_path)
      nr_frames = self._get_number_of_frames()
      crt_index += nr_frames
      indexes.append([past_index, crt_index])
      past_index = crt_index+1
    #endfor
    self._metadata.input_indexes = indexes

    self._time_start_upload = time()
    if not self.cfg_use_local_system_to_save:
      self._threads_upload = [None for _ in range(len(output_files))]
      for i in range(len(self._threads_upload)):
        self._threads_upload[i] = ThreadWithReturnValue(
          target=self._file_system_manager.upload,
          kwargs=dict(
            file_path=self._paths_splitted_video_files[i],
            target_path='MASTER_UPLOADS/{}_{}'.format(self._upload_id, output_files[i])
          ),
          name=ct.THREADS_PREFIX + self.cfg_name + '_up_{}'.format(i+1)
        )

        self._threads_upload[i].daemon = True  # Why?
        self._threads_upload[i].start()
      #endfor
    #endif

    return

  def _release_map(self):
    if self.cfg_delete_path:
      os.remove(self._path)
      self.P("In `_release_map` removed file '{}'".format(self._path), color='y')

    if not self.cfg_use_local_system_to_save:
      try:
        shutil.rmtree(self._path_split_dir)
        self.P("In `_release_map` removed tree '{}'".format(self._path_split_dir), color='y')
      except:
        pass
    #endif

    return

  def _maybe_reconnect_map(self):
    return

  def _run_data_aquisition_step_map(self):
    self._metadata.upload_elapsed_time = str(timedelta(seconds=time() - self._time_start_upload))
    if self.alive_threads_upload:

      metadata = self._metadata.__dict__.copy()
      self._add_inputs(
        [
          self._new_input(struct_data=metadata, metadata=metadata)
        ]
      )
      return
    #endif

    if self.has_threads_upload:
      movies = [thr.join() for thr in self._threads_upload]
    else:
      movies = self._paths_splitted_video_files
    #endif
    self._add_inputs(
      [
        self._new_input(struct_data=dict(url_map=movies, workers=self._workers[:len(movies)]))
      ]
    )
    self._finished_map = True
    return
