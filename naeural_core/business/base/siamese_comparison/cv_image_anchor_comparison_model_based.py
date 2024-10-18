from naeural_core.business.base.siamese_comparison.cv_image_anchor_comparison import \
    CvImageAnchorComparisonPlugin as BasePlugin

_CONFIG = {
  **BasePlugin.CONFIG,
  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES'],
  },

  # Developer config
  "ANCHOR_RELOAD_PERIOD": None,
  "SERVING_DATETIME_FORMAT": "%Y/%m/%d_%H:%M:%S",
  "WARNING_ANCHOR_NOT_SYNCHRONIZED_TIME": 10,  # seconds
  'DISCARD_FAILED_ANALYSIS': True,
  #############################

  # Plugin dependant config
  "AI_ENGINE": ["lowres_general_detector"],
  "SERVING_SERIALIZATION_NAME": "th_planos1_refactor",
  #############################

  # Alerter config
  "ALERT_DATA_COUNT": 5,
  "ALERT_RAISE_CONFIRMATION_TIME": 10,
  "ALERT_LOWER_CONFIRMATION_TIME": 15,
  "ALERT_RAISE_VALUE": 50,
  "ALERT_LOWER_VALUE": 30,
  "ALERT_MODE": "mean",
  "ALERT_REDUCE_VALUE": False,

  'RE_RAISE_TIME': 3 * 60,  # seconds
  'FORCED_LOWER': True,
  'FORCED_LOWER_AFTER_RE_RAISE_TIME': 12 * 60,  # seconds
  #############################

  # User config
  "IS_ATM": False,
  "ATM_ANALYSIS_IGNORE_MAX_PERSON_AREA": 101,  # percent
  "ATM_PEOPLE_IN_FRAME_COOLDOWN_FRAMES": 20,  # <= 0 means no cooldown
  "ATM_AI_ENGINE": ["lowres_general_detector"],

  "ANCHOR_MAX_SUM_PERSON_AREA": 0,  # percent
  "ANALYSIS_IGNORE_MAX_PERSON_AREA": 70,  # percent
  "ANALYSIS_IGNORE_MIN_PERSON_AREA": 0,  # percent

  "PERSON_MIN_AREA_THRESHOLD": 0.02,  # between 0 and 1
  "PERSON_MIN_AREA_PROB_THRESHOLD": 0.5,  # between 0 and 1

  "MIN_ENTROPY_THRESHOLD": 4,
  "PEOPLE_IN_FRAME_COOLDOWN_FRAMES": 10,  # <= 0 means no cooldown

  "DEMO_MODE": False,
  "ANCHOR_URL": None,
  'WITNESS_BEFORE_CROP': True,
  #############################
}


class CvImageAnchorComparisonModelBasedPlugin(BasePlugin):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(CvImageAnchorComparisonModelBasedPlugin, self).__init__(**kwargs)

    self.__found_anchor_reload_period = False
    self.__anchor_reload_period = None

    self.__last_warning_anchor_not_synchronized = 0
    return

  @property
  def anchor_reload_period(self):
    # we get the anchor reload period from serving
    if not self.__found_anchor_reload_period:
      dct_inferences = self.inferences
      if dct_inferences is None:
        # we haven't received any inference yet, so we can't get the anchor reload period
        return None
      # endif dct_inferences is None

      # the anchor reload period is in minutes, but we need it in seconds
      self.__anchor_reload_period = dct_inferences[self.ct.ThAnchor.ANCHOR_RELOAD_TIME]
      self.__found_anchor_reload_period = True
    # endif anchor reload period not found
    return self.__anchor_reload_period

  def compare_image_with_anchor(self, image):
    if not self._inference_validation():
      return None, None

    return self.process_model_inference()

  def _inference_validation(self):
    dct_inferences = self.inferences

    if dct_inferences is None:
      return False

    self_inferences = dct_inferences.get(self.str_unique_identification)
    if self_inferences is None:
      return False

    if self_inferences[self.ct.ThAnchor.PREDICTIONS] is None:
      return False

    return self.custom_inference_validation()

  def custom_image_validation(self, image):
    """
    Check if inferences exist.

    Parameters
    ----------
    image : np.ndarray
        Image to be validated.

    Returns
    -------
    bool
        True if inferences exist, False otherwise.
    """

    if not self._inference_validation():
      return False

    dct_inferences = self.inferences[self.str_unique_identification]
    anchor_last_save_time = dct_inferences[self.ct.ThAnchor.ANCHOR_UPDATED_AT]
    anchor_last_save_time = self.datetime.strptime(anchor_last_save_time, self.cfg_serving_datetime_format)
    anchor_last_save_time = self.datetime.timestamp(anchor_last_save_time)

    # the distance should not be more than one second
    if abs(self._anchor_last_save_time - anchor_last_save_time) > 1:
      if self.time() - self.__last_warning_anchor_not_synchronized > self.cfg_warning_anchor_not_synchronized_time:
        msg = "Ignoring inference because anchor reload time is not correct. " + \
              f"Expected {self._anchor_last_save_time}, got {anchor_last_save_time}"
        if self.__last_warning_anchor_not_synchronized == 0:
          self._create_abnormal_notification(
            msg=msg,
            displayed=True,
          )
        else:
          self.P(msg, color='r')

        self.__last_warning_anchor_not_synchronized = self.time()
      return False

    self.__last_warning_anchor_not_synchronized = 0
    return True

  def _update_anchor_callback(self, serving_serialized_data):
    if serving_serialized_data is None:
      serving_serialized_data = {}
    dct_anchor_metadata = serving_serialized_data.get(self.ct.ThAnchor.ANCHOR_METADATA, {})
    anchors = serving_serialized_data.get(self.ct.ThAnchor.ANCHOR_IMAGES, {})

    anchors[self.str_unique_identification] = self._anchor
    serving_serialized_data[self.ct.ThAnchor.ANCHOR_IMAGES] = anchors

    if not self.str_unique_identification in dct_anchor_metadata:
      dct_anchor_metadata[self.str_unique_identification] = {}

    anchor_last_save_time = self.datetime.fromtimestamp(self._anchor_last_save_time)
    dct_anchor_metadata[self.str_unique_identification][self.ct.ThAnchor.ANCHOR_UPDATED_AT] = self.datetime.strftime(anchor_last_save_time,
                                                                                                                     self.cfg_serving_datetime_format)
    serving_serialized_data[self.ct.ThAnchor.ANCHOR_METADATA] = dct_anchor_metadata

    return serving_serialized_data

  def _cleanup_anchor_callback(self, serving_serialized_data):
    if serving_serialized_data is None:
      serving_serialized_data = {}
      return serving_serialized_data

    dct_anchor_metadata = serving_serialized_data.get(self.ct.ThAnchor.ANCHOR_METADATA, {})
    anchors = serving_serialized_data.get(self.ct.ThAnchor.ANCHOR_IMAGES, {})

    anchors.pop(self.str_unique_identification, None)
    serving_serialized_data[self.ct.ThAnchor.ANCHOR_IMAGES] = anchors

    dct_anchor_metadata.pop(self.str_unique_identification, None)
    serving_serialized_data[self.ct.ThAnchor.ANCHOR_METADATA] = dct_anchor_metadata

    return serving_serialized_data

  def custom_save_anchor(self, image):
    if self.cfg_anchor_url is not None:
      files, _ = self.maybe_download(self.cfg_anchor_url)
      if len(files) != 0:
        # TODO: change this to a diskapi call
        anchor = self.PIL.Image.open(self.os_path.join(self.log.get_output_folder(), files[0]))
        if anchor is not None:
          self._anchor = anchor
          self._anchor_last_save_time = self.time()

    self.persistence_serialization_update_serving(
      name=self.cfg_serving_serialization_name,
      update_callback=self._update_anchor_callback,
    )
    return

  def custom_load_anchor(self):
    serving_saved_data = self.persistence_serialization_load_from_serving(name=self.cfg_serving_serialization_name)
    if serving_saved_data is None:
      return None

    anchors = serving_saved_data.get(self.ct.ThAnchor.ANCHOR_IMAGES, {})
    metadata = serving_saved_data.get(self.ct.ThAnchor.ANCHOR_METADATA, {})

    if not self.str_unique_identification in anchors:
      return None

    anchor = anchors[self.str_unique_identification]

    if not self.str_unique_identification in metadata:
      # this should not be possible, but we handle it anyway
      return anchor, 0

    anchor_last_save_time = metadata[self.str_unique_identification][self.ct.ThAnchor.ANCHOR_UPDATED_AT]
    anchor_last_save_time = self.datetime.strptime(anchor_last_save_time, self.cfg_serving_datetime_format)
    anchor_last_save_time = self.datetime.timestamp(anchor_last_save_time)

    self.P("Loaded anchor from serving cache. Last save time: " + metadata[self.str_unique_identification][self.ct.ThAnchor.ANCHOR_UPDATED_AT], boxed=True)
    return anchor, anchor_last_save_time

  @property
  def inferences(self):
    """
    Get the inferences from the dataapi.
    This method should be something like this:
    ```
    return self.dataapi_images_inferences()['TH_ANCHOR_MODEL_HERE'][0]
    ```

    Returns
    -------
    dct_inferences: dict | None
        The inferences from the dataapi.
    """
    raise NotImplementedError

  def custom_inference_validation(self):
    """
    Custom inference validation. This method can be used to add custom inference validation logic.
    This method is called after the inferences are loaded from the dataapi.
    This method guarantees that `self.inferences` are not None and predictions are not None as well.

    Returns
    -------
    bool
        True if the inferences are valid, False otherwise.
    """
    return True

  def process_model_inference(self):
    """
    Parse the model inferences and return the result.

    Returns
    -------
    (result, debug_result) : Tuple[float, dict] | None
        result: The inferred result that will be put in alerter
        debug_result: The debug information to be added to the witness
    """
    raise NotImplementedError

  def _on_close(self):
    """
    Called at shutdown time.

    Returns
    -------
    None.

    """
    super(CvImageAnchorComparisonModelBasedPlugin, self)._on_close()
    self.persistence_serialization_update_serving(
      name=self.cfg_serving_serialization_name,
      update_callback=self._cleanup_anchor_callback,
    )
    return
