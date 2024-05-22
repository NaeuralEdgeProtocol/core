# TODO Bleo: WIP
from core.business.base import CVPluginExecutor as BasePlugin

_CONFIG = {
  **BasePlugin.CONFIG,
  'AI_ENGINE': ['lowres_general_detector'],
  'OBJECT_TYPE': ['person'],

  'ALLOW_EMPTY_INPUTS': True,
  'RUN_WITHOUT_IMAGE': True,
  "CLASSES": None,

  'LOG_FAILED_SAVES': False,
  'FORCE_TERMINATE_COLLECT': False,
  'FORCE_OBJECT_TYPE_BALANCED': False,
  'CLOUD_PATH': 'DATASETS/',
  'OBJECTIVE_NAME': None,
  'REPORT_PERIOD': 10*60,
  'COLLECT_UNTIL': None,
  'LABELS_DONE': False,

  'PROCESS_DELAY': 1,
  'MAX_INPUTS_QUEUE_SIZE': 32,

  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES'],
  },
}

__VER__ = '0.1.0.0'


class CVCropTrainingDataPlugin(BasePlugin):
  def on_init(self):
    super(CVCropTrainingDataPlugin, self).on_init()
    self.final_payload = None
    self._source_names = set()
    self._dataset_local_root = None
    self.finished = False
    self._received_input = False
    self.dataset_stats = self.defaultdict(lambda : 0)
    self.count_saved_by_object_type = self.defaultdict(lambda : 0)
    self.dataset_rel_path = self.os_path.join('datasets', self.cfg_objective_name)
    self.dataset_abs_path = self.os_path.join(self.get_output_folder(), self.dataset_rel_path)
    return

  def get_collect_until(self):
    """
    Provides possibility to stop collecting data at a certain datetime
    """
    collect_until = self.cfg_collect_until
    if collect_until is not None:
      collect_until = self.datetime.strptime(collect_until, '%Y-%m-%d %H:%M')
    return collect_until

  def get_ds_classes(self):
    classes = self.cfg_classes
    if classes is None:
      classes = self.cfg_object_type
    return classes if isinstance(classes, list) else [classes]

  @property
  def collect_until_passed(self):
    collect_until = self.get_collect_until()
    if collect_until is not None:
      return (collect_until - self.datetime.now()).total_seconds() < 0
    return False

  @property
  def dataset_object_name(self):
    return self.os_path.join(self.cfg_cloud_path, self.cfg_objective_name + '_RAW.zip')

  @property
  def done_collecting(self):
    return self.final_payload is not None

  def dataset_info_object_filename(self):
    return self.os_path.join(self.dataset_rel_path, 'ADDITIONAL.json')

  def _check_if_can_save_object_type(self, object_type):
    if not self.cfg_force_object_type_balanced:
      return True

    crt_object_type_count = self.count_saved_by_object_type[object_type]
    can_save = True
    for k in self.cfg_object_type:
      if k == object_type:
        continue

      if crt_object_type_count > self.count_saved_by_object_type[k]:
        can_save = False

    return can_save

  def crop_and_save_one_img(self, np_img, inference, source_name, current_interval):
    try:
      self.start_timer('crop_and_save_one_img')
      top, left, bottom, right = list(map(lambda x: int(x), inference['TLBR_POS']))
      object_type = inference['TYPE']
      np_cropped_img = np_img[top:bottom + 1, left:right + 1, :]
      subdir = self.os_path.join(
        self.dataset_rel_path,
        str(object_type),
        source_name,
      )
      if current_interval is not None:
        subdir = self.os_path.join(subdir, current_interval)
      # endif interval given
      fname = f'{object_type}_{self.count_saved_by_object_type[object_type]:06d}_{self.now_str(short=True)}.jpg'
      self.diskapi_save_image_output(
        image=np_cropped_img,
        filename=fname,
        subdir=subdir
      )
      self.stop_timer('crop_and_save_one_img')
    except Exception as e:
      self.stop_timer('crop_and_save_one_img')
      if self.cfg_log_failed_saves:
        self.P(f'Failed save from {source_name} in {current_interval} with exception {e}', color='r')
      return None
    return subdir

  def _crop_and_save_all_images(self):
    dct_imgs = self.dataapi_images()

    for i, np_img in dct_imgs.items():
      lst_inferences = self.dataapi_specific_image_instance_inferences(idx=i)
      inp_metadata = self.dataapi_specific_input_metadata(idx=i)
      source_name = inp_metadata.get('SOURCE_STREAM_NAME', self.dataapi_stream_name())
      current_interval = inp_metadata.get('current_interval', 'undefined')
      self._source_names.add(source_name)
      for infer in lst_inferences:
        object_type = infer.get('TYPE', None)
        if object_type is None:
          self.P("Inference did not return 'TYPE', cannot save the crop", color='r')
          continue
        #endif

        if self._check_if_can_save_object_type(object_type):
          subdir = self.crop_and_save_one_img(
            np_img=np_img, inference=infer, source_name=source_name, current_interval=current_interval
          )
          if subdir is not None:
            self.count_saved_by_object_type[object_type] += 1
            self.dataset_stats[subdir] += 1
          # endif successfully saved
        # endif allowed to save
      # endfor inferences
    # endfor images
    return

  def _generate_progress_payload(self, **kwargs):
    payload = self._create_payload(counts=self.dataset_stats, **kwargs)
    return payload

  def archive_and_upload_ds(self):
    classes_data = {
      'classes': self.get_ds_classes(),
      'name': self.cfg_objective_name,
    }
    classes_path = self.diskapi_save_json_to_output(dct=classes_data, filename=self.dataset_info_object_filename())
    fn_zip = self.diskapi_zip_dir(self.dataset_abs_path)
    ds_url, _ = self.upload_file(
      file_path=fn_zip,
      target_path=self.dataset_object_name,
      force_upload=True
    )
    self.diskapi_delete_file(fn_zip)
    return ds_url

  def stop_gather(self):
    self.P(f'Stopping all the other data sources: {self._source_names}')
    self.cmdapi_stop_current_stream()
    for s in self._source_names:
      self.cmdapi_stop_other_stream_on_current_box(s)
    self.finished = True
    return

  def finalise_process(self):
    ds_url = self.archive_and_upload_ds()
    payload_kwargs = {
      'url': ds_url, 'path': self.dataset_object_name,
    }
    payload = self._generate_progress_payload(**payload_kwargs)
    self.final_payload = payload
    self.add_payload(payload)
    return

  def _process(self):
    # TODO: test data gathering when other pipelines are running
    # We should only gather from the specified streams
    # Step 1: If gathering not finished and input available, crop and save all images.
    if not self.done_collecting and self.dataapi_received_input():
      self._crop_and_save_all_images()
      self._received_input = True

    payload = None
    # Step 2: If report period passed, generate progress payload or resend final payload
    # if the raw dataset was uploaded.
    if self.time() - self.last_payload_time >= self.cfg_report_period:
      payload = self.final_payload if self.done_collecting else self._generate_progress_payload()

    # Step 3: If no more data gathering is needed, but it was not yet finished,
    # finalise the process and send the final payload.
    if (self.collect_until_passed or self.cfg_force_terminate_collect) and not self.done_collecting:
      if not self._received_input:
        return

      self.finalise_process()
      payload = None
    # endif no more data gathering needed

    # Step 4: If the raw data was labeled, stop data sources and current pipeline.
    if self.cfg_labels_done and not self.finished:
      self.stop_gather()
      payload = None
    # endif the data was labeled so we no longer need this plugin active

    return payload
