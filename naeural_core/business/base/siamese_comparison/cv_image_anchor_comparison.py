# local dependencies
from naeural_core.business.base import CVPluginExecutor as BasePlugin

__VER__ = '1.0.0'


_CONFIG = {
  **BasePlugin.CONFIG,
  "VALIDATION_RULES": {
    **BasePlugin.CONFIG["VALIDATION_RULES"],
  },

  # Developer config
  "AI_ENGINE": ["lowres_general_detector"],
  "ALLOW_COMPARISON_WITH_NO_ANCHOR": False,
  "DISCARD_FAILED_ANALYSIS": False,
  "MAX_INPUTS_QUEUE_SIZE": 1,  # slow plugin - must process only current state/input
  "WARNING_ANCHOR_SAVE_FAIL_SEND_INTERVAL": 60,  # seconds
  "IMG_ORIG": False,
  'ADD_ORIGINAL_IMAGE': False,
  #############################

  # Alerter config
  "ALERT_DATA_COUNT": 5,
  "ALERT_RAISE_CONFIRMATION_TIME": 20,
  "ALERT_LOWER_CONFIRMATION_TIME": 15,
  "ALERT_RAISE_VALUE": 60,
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
  "ANCHOR_RELOAD_PERIOD": 10 * 60,  # second
  #############################

}


class CvImageAnchorComparisonPlugin(BasePlugin):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(CvImageAnchorComparisonPlugin, self).__init__(**kwargs)
    return

  def startup(self):
    super(CvImageAnchorComparisonPlugin, self).startup()
    self.__last_capture_image = None
    self._anchor_last_save_time = 0
    self._force_save_anchor = False
    self._anchor = None
    self._people_in_frame_cooldown_counter = 0

    self._last_anchor_validation_failed_time = 0

    self._alerter_re_raised = False
    self._alerter_forced_lower = False

    self.alerter_maybe_create("__last_witness_cache__")

    if self.cfg_forced_lower:
      assert self.cfg_re_raise_time is not None, "Forced lower cannot be configured with no re raise time!"
      assert self.cfg_forced_lower_after_re_raise_time is not None, "Invalid forced lower after re raise time '{}'".format(
        self.cfg_forced_lower_after_re_raise_time
      )

    self._maybe_load_anchor()
    return

  @property
  def cfg_analysis_ignore_max_person_area(self):
    # cfg_is_atm can be created after this one
    if self._instance_config.get("IS_ATM", False):
      # increase the max person area for ATM, because people will occupy the full frame most of the time
      return self._instance_config.get('ATM_ANALYSIS_IGNORE_MAX_PERSON_AREA', 101)
    return self._instance_config.get('ANALYSIS_IGNORE_MAX_PERSON_AREA', 70)

  @property
  def cfg_people_in_frame_cooldown_frames(self):
    # cfg_is_atm can be created after this one
    if self._instance_config.get("IS_ATM", False):
      # increase the cooldown for ATM, because people can stay in front of the camera in strange positions
      return self._instance_config.get('ATM_PEOPLE_IN_FRAME_COOLDOWN_FRAMES', 20)
    return self._instance_config.get('PEOPLE_IN_FRAME_COOLDOWN_FRAMES', 10)

  @property
  def cfg_ai_engine(self):
    # cfg_is_atm can be created after this one
    if self._instance_config.get("IS_ATM", False):
      return self._instance_config.get('ATM_AI_ENGINE', self._instance_config.get('AI_ENGINE', []))
    return self._instance_config.get('AI_ENGINE', [])

  def _can_save_newer_anchor(self):
    """
    Check if we can save a new anchor.
    We check if the anchor is expired, if there are alerts raised, if the anchor is not saved yet, etc.

    Returns
    -------
    bool
        True if we can save a newer anchor, False otherwise
    """
    if self._force_save_anchor or self._anchor is None:
      # check for anchor None because in the context of model based comparison,
      # we need to get information from the serving process (like the anchor reload period)
      # and because of that we will add values in the alerter
      return True

    is_alert = self.alerter_is_alert()
    last_alert_value = self.alerter_get_last_value()

    # the alerter should be lowered and the last value in the alerter should not be higher than the raise value
    # if we compare with the lower value, we might hit a situation where the plugin can never recover from a 
    # faulty anchor without the need of a forced lower
    alerter_ok = not is_alert and (last_alert_value is None or last_alert_value < self.cfg_alert_raise_value)

    if self.anchor_reload_period is None:
      # we will never reload the anchor
      return False

    anchor_expired = self.time() - self._anchor_last_save_time > self.anchor_reload_period

    return anchor_expired and alerter_ok

  def _maybe_save_anchor(self, image, object_detection_inferences):
    if not self._can_save_newer_anchor():
      return

    valid, reason = self._validate_image_as_anchor(image, object_detection_inferences)
    if valid:
      self.P("Saving anchor. {}".format(reason))
      self._anchor = image
      self._anchor_last_save_time = self.time()
      self._force_save_anchor = False
      self.custom_save_anchor(image)

      self._last_anchor_validation_failed_time = 0
    else:
      current_time = self.time()
      if current_time - self._last_anchor_validation_failed_time >= self.cfg_warning_anchor_save_fail_send_interval:
        self.P("Anchor validation failed: {}".format(reason), color="r")
        self._last_anchor_validation_failed_time = current_time
      return
    pass

  def _maybe_load_anchor(self):
    if self._anchor is not None:
      return

    result = self.custom_load_anchor()
    if result is None:
      return

    anchor, anchor_last_save_time = result

    self._anchor = anchor
    self._anchor_last_save_time = anchor_last_save_time
    self._last_anchor_validation_failed_time = 0

    return

  def _get_intersection_tlbr(self, t1, l1, b1, r1, t2, l2, b2, r2):
    """
    Get the intersection between two TLBRs

    Parameters
    ----------
    t1 : int
        Top coordinate of the first TLBR
    l1 : int
        Left coordinate of the first TLBR
    b1 : int
        Bottom coordinate of the first TLBR
    r1 : int
        Right coordinate of the first TLBR
    t2 : int
        Top coordinate of the second TLBR
    l2 : int
        Left coordinate of the second TLBR
    b2 : int
        Bottom coordinate of the second TLBR
    r2 : int
        Right coordinate of the second TLBR

    Returns
    -------
    (t, l, b, r) : Tuple[int, int, int, int]
        The intersection TLBR
    """
    t = max(t1, t2)
    l = max(l1, l2)
    b = min(b1, b2)
    r = min(r1, r2)

    if t >= b or l >= r:
      return None

    return t, l, b, r

  def _intersect_tlbrs_with_target_zone(self, object_detector_inferences):
    """
    Intersect the inferences with the target zone. Edit the inferences tlbr to be bounded by the target zone.

    Parameters
    ----------
    object_detector_inferences : List[dict]
        The inferences to be intersected with the target zone

    Returns
    -------
    List[dict]
        The intersected inferences
    """

    # if no target zone defined, then we do not intersect
    if self._coords_type == self.ct.COORDS_NONE:
      return object_detector_inferences

    _t, _l, _b, _r = self._top, self._left, self._bottom, self._right

    lst_intersect_inferences = []
    for dct_inference in object_detector_inferences:
      t, l, b, r = dct_inference['TLBR_POS']
      intersection_tlbr = self._get_intersection_tlbr(_t, _l, _b, _r, t, l, b, r)
      if intersection_tlbr is None:
        continue

      lst_intersect_inferences.append({
        **dct_inference,
        'TLBR_POS': intersection_tlbr,
      })

    return lst_intersect_inferences

  def _maybe_update_people_cooldown_counter(self, image, object_detector_inferences):
    people_ok, _ = self.__validate_image_as_anchor_check_people(image, object_detector_inferences)
    if people_ok:
      if self._people_in_frame_cooldown_counter > 0:
        self._people_in_frame_cooldown_counter -= 1
      # endif counter > 0
    else:
      self._people_in_frame_cooldown_counter = self.cfg_people_in_frame_cooldown_frames
    return

  def __validate_image_as_anchor_check_people(self, image, object_detection_inferences):
    people_areas = self.__people_areas_prc(image, object_detection_inferences)

    total_area = sum(people_areas) * 100

    return total_area <= self.cfg_anchor_max_sum_person_area, "People area {:.02f}".format(total_area)

  def __validate_image_as_anchor_check_people_cooldown(self):
    people_cooldown_ok = self._people_in_frame_cooldown_counter <= 0
    people_cooldown_validation_msg = "People cd {} frames".format(self._people_in_frame_cooldown_counter)
    return people_cooldown_ok, people_cooldown_validation_msg

  def __validate_image_as_anchor_check_entropy(self, image):
    entropy = self.image_entropy(image)
    return entropy >= self.cfg_min_entropy_threshold, f"Entropy {entropy:.02f}"

  def _validate_image_as_anchor(self, image, object_detection_inferences):
    # Saving anchor. People: 0 (0 frames), Entopy:  7.54
    people_ok, people_validation_msg = self.__validate_image_as_anchor_check_people(image, object_detection_inferences)
    people_cooldown_ok, people_cooldown_validation_msg = self.__validate_image_as_anchor_check_people_cooldown()
    entropy_ok, entropy_validation_msg = self.__validate_image_as_anchor_check_entropy(image)

    anchor_ok = people_ok and people_cooldown_ok and entropy_ok

    aggregated_messages = ", ".join([people_validation_msg, people_cooldown_validation_msg, entropy_validation_msg])

    return anchor_ok, aggregated_messages

  def _maybe_force_lower(self):
    """
    Check if the alert needs to be force lowered. This method assumes the alerter is already re-raised.

    Parameters
    ----------
    is_alert : bool
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if self.cfg_forced_lower is None or not self.cfg_forced_lower:
      return False

    if self.alerter_time_from_last_change() is None:
      # this should never happen
      return False

    # this time does not take into account the re-raise time
    time_since_last_raise = self.alerter_time_from_last_change()

    force_lower_threshold = self.cfg_forced_lower_after_re_raise_time + self.cfg_re_raise_time

    force_lower_time_elapsed = time_since_last_raise > force_lower_threshold

    return force_lower_time_elapsed

  def _maybe_re_raise_alert(self):
    """
    Check if the alert needs to be re-raised. This method assumes the alerter is already raised.

    Returns
    -------
    bool
        True if the alert needs to be re-raised, False otherwise
    """
    if self.cfg_re_raise_time is None:
      return False

    if self.alerter_time_from_last_change() is None:
      return False

    time_since_last_raise = self.alerter_time_from_last_change()
    re_raise_time_elapsed = time_since_last_raise > self.cfg_re_raise_time

    return re_raise_time_elapsed and not self._alerter_re_raised

  def _custom_alerter_status_changed(self):
    """
    Check if the alerter status changed. This is a custom implementation for the alerter status change
    that takes into account the re-raise and force lower logic

    Returns
    -------
    bool, current_alerter_status
        True if the alerter status changed, False otherwise
    """
    # get the current state of the alerter
    is_alert = self.alerter_is_alert()

    is_new_raised = self.alerter_is_new_raise()
    is_new_lower = self.alerter_is_new_lower()

    if is_new_raised:
      # if status changed from lower to raised
      self._alerter_re_raised = False
      return True, "Raised"

    if is_new_lower:
      self._alerter_forced_lower = False
      self._alerter_re_raised = False
      return True, "Lower"

    # the logic is the following:
    # 1. raise the alert if necessary, T=0
    # 2. if the alert is raised and T > T_reraise, then re-raise the alert
    # 3. if alert is raised and re-raised and T > T_reraise + T_forced_lower, then force lower the alert

    if not is_alert:
      # if the alerter is not raised, then we do not need to check for re-raise or force lower
      self._alerter_forced_lower = False
      self._alerter_re_raised = False
      return False, "OK"

    need_to_re_raise = self._maybe_re_raise_alert()

    if need_to_re_raise:
      self._alerter_re_raised = True
      return True, "Re-Raised"

    if self._alerter_re_raised:
      need_to_force_lower = self._maybe_force_lower()

      if need_to_force_lower:
        self._alerter_forced_lower = True
        return True, "Forced Lower"

    return False, "Alert"

  def _draw_witness_image(self, img, object_detector_inferences, debug_results, **kwargs):
    for dct_inference in object_detector_inferences:
      if dct_inference['TYPE'] == 'person':
        t, l, b, r = dct_inference['TLBR_POS']
        img = self._painter.draw_detection_box(
            image=img,
            top=t,
            left=l,
            bottom=b,
            right=r,
            label=f"conf {100*dct_inference['PROB_PRC']:.0f}%",
            color=self.consts.GREEN if self.__get_person_prc_area(img, dct_inference) is not None else self.consts.RED,
        )
    return img

  def _create_witness_image(self, original_image, debug_results, object_detector_inferences, **kwargs):
    return self.get_witness_image(
      img=original_image,
      draw_witness_image_kwargs={
        'object_detector_inferences': object_detector_inferences,
        'debug_results': debug_results,
      }
    )

  def _get_payload(self, alerter_status, **kwargs):
    # choose the cache based on the alerter status
    cache_witness = self.get_current_witness_kwargs(demo_mode=self.cfg_demo_mode)
    debug_info = cache_witness['debug_info']
    original_image = cache_witness['original_image']
    object_detector_inferences = cache_witness['object_detector_inferences']
    debug_results = cache_witness['debug_results']

    for dct_debug_info in debug_info:
      self.add_debug_info(**dct_debug_info)
    self.add_debug_info({'Alerter Status': alerter_status})

    # create witness
    img_witness = self._create_witness_image(
      original_image=original_image,
      debug_results=debug_results,
      object_detector_inferences=object_detector_inferences,
    )

    # reset alerter if forced lower
    status_changed = (
      self.alerter_status_changed() or
      (self._alerter_re_raised and not self._alerter_forced_lower) or
      self._alerter_forced_lower
    )

    payload_kwargs = {
      'is_re_raise': self._alerter_re_raised if not self._alerter_forced_lower else False,
      'is_forced_lower': self._alerter_forced_lower,
      'is_new_re_raise': self._alerter_re_raised if not self._alerter_forced_lower else False,
      'is_new_forced_lower': self._alerter_forced_lower,
      'is_new_lower': self.alerter_is_new_lower() or self._alerter_forced_lower,
      'is_alert_status_changed': status_changed,
    }

    if self._alerter_re_raised and not self._alerter_forced_lower:
      alert_first_raise_time = self.time() - self.alerter_time_from_last_change()
      alert_first_raise_datetime = self.datetime.fromtimestamp(alert_first_raise_time)
      str_alert_first_raise = self.datetime.strftime(alert_first_raise_datetime, "%Y-%m-%d %H:%M:%S.%f")

      payload_kwargs['status'] = f'Alert re-raised (first raise at {str_alert_first_raise})'
    elif self._alerter_forced_lower:
      alert_first_raise_time = self.time() - self.alerter_time_from_last_change()
      alert_first_raise_datetime = self.datetime.fromtimestamp(alert_first_raise_time)
      str_alert_first_raise = self.datetime.strftime(alert_first_raise_datetime, "%Y-%m-%d %H:%M:%S.%f")

      alert_re_raise_time = self.time() - self.alerter_time_from_last_change() + self.cfg_re_raise_time
      alert_re_raise_datetime = self.datetime.fromtimestamp(alert_re_raise_time)
      str_alert_re_raise = self.datetime.strftime(alert_re_raise_datetime, "%Y-%m-%d %H:%M:%S.%f")

      payload_kwargs['status'] = f'Alert forced-lower (first raise at {str_alert_first_raise} and re-raised at {str_alert_re_raise})'

    if alerter_status == "Forced Lower":
      self._anchor = None
      self._force_save_anchor = True
      self._maybe_save_anchor(original_image, object_detector_inferences)
      self.alerter_hard_reset()
      self._alerter_re_raised = False
      self._alerter_forced_lower = False

    # create payload
    payload = self._create_payload(
      img=[img_witness, original_image, self._anchor],

      # we no longer send the original image because we have IMG_ORIG set to True
      # img=[img_witness, self._anchor],
      **payload_kwargs,
    )

    return payload

  def __get_person_prc_area(self, image, dct_inference):
    if self._coords_type == self.ct.COORDS_NONE:
      H, W = image.shape[:2]
      image_area = H * W
    else:
      image_area = (self._bottom - self._top) * (self._right - self._left)

    area_prc = None
    if dct_inference['TYPE'] == 'person':
      t, l, b, r = dct_inference['TLBR_POS']
      area = (r - l) * (b - t)
      area_prc = area / image_area

      area_too_small = area_prc < self.cfg_person_min_area_threshold
      confident_in_detection = dct_inference['PROB_PRC'] >= self.cfg_person_min_area_prob_threshold
      if area_too_small and not confident_in_detection:
        # ignore small detections with low confidence
        area_prc = None

    return area_prc

  def __people_areas_prc(self, image, lst_obj_inferences, round_digits=None):
    people_areas_prc = [self.__get_person_prc_area(image, dct_inference) for dct_inference in lst_obj_inferences]
    people_areas_prc = [area for area in people_areas_prc if area is not None]

    if round_digits is not None:
      people_areas_prc = [round(area, round_digits) for area in people_areas_prc]

    return people_areas_prc

  def _validate_image_for_analysis(self, image, lst_obj_inferences):
    return self._validate_detections_for_analysis(image, lst_obj_inferences) and self.custom_image_validation(image)

  def _validate_detections_for_analysis(self, image, lst_obj_inferences):
    people_areas = self.__people_areas_prc(image, lst_obj_inferences)

    # check if all persons are smaller than a percentage
    all_persons_small = all([area * 100 <= self.cfg_analysis_ignore_min_person_area for area in people_areas])

    # check if there is a person bigger than a percentage
    exists_person_big = any([area * 100 >= self.cfg_analysis_ignore_max_person_area for area in people_areas])

    return all_persons_small or exists_person_big

  def _prepare_debug_info_for_witness(self, img, object_detector_inferences, debug_results):
    human_readable_last_anchor_time = self.datetime.fromtimestamp(self._anchor_last_save_time)
    human_readable_last_anchor_time = self.datetime.strftime(human_readable_last_anchor_time, "%Y-%m-%d %H:%M:%S")

    frame_entropy = self.image_entropy(img) if img is not None else 0  # image should never be None
    anchor_entropy = self.image_entropy(self._anchor) if self._anchor is not None else 0  # anchor can be None

    debug_info = [
      {'value': "A <{} F <{} >{}: {}  (E {:.02f}) IS_ATM={}".format(
        self.cfg_anchor_max_sum_person_area,
        self.cfg_analysis_ignore_min_person_area,
        self.cfg_analysis_ignore_max_person_area,
        [area * 100 for area in self.__people_areas_prc(img, object_detector_inferences, 2)],
        frame_entropy,
        self.cfg_is_atm,
      )},
      {'value': "Last Anchor Time: {}   Anchor Save Period (s): {} Anchor E: {:.02f}".format(
        human_readable_last_anchor_time, self.anchor_reload_period, anchor_entropy)},
    ]
    if self.cfg_demo_mode:
      if self.cfg_forced_lower:
        debug_info.append({
          'value': {
              'last_raise': self.alerter_time_from_last_change(),
              're-raise_time': self.cfg_re_raise_time,
              'force_lower_time': self.cfg_forced_lower_after_re_raise_time + self.cfg_re_raise_time,
          }})
      elif self.cfg_re_raise_time is not None:
        debug_info.append({
          'value': {
              'last_raise': self.alerter_time_from_last_change(),
              're-raise_time': self.cfg_re_raise_time,
          }})
    debug_info.append({'value': debug_results})
    return debug_info

  def _update_cache_witness_image(self, img, object_detector_inferences, debug_info, result, debug_results):
    self.update_witness_kwargs(
      witness_args={
        'debug_info': debug_info,
        'object_detector_inferences': object_detector_inferences,
        'debug_results': debug_results,
        'original_image': img,
      },
      pos=self.alerter_get_current_frame_state(result),
    )
    return

  def _update_cache_last_witness_image(self, img, object_detector_inferences, debug_info, debug_results):
    self.update_witness_kwargs(
      witness_args={
        'debug_info': debug_info,
        'object_detector_inferences': object_detector_inferences,
        'debug_results': debug_results,
        'original_image': img,
      },
      alerter="__last_witness_cache__",
      pos=0
    )
    return

  def get_last_witness_image(self):
    cache_witness = self.get_current_witness_kwargs(pos=0, alerter="__last_witness_cache__")
    debug_info = cache_witness['debug_info']
    original_image = cache_witness['original_image']
    object_detector_inferences = cache_witness['object_detector_inferences']
    debug_results = cache_witness['debug_results']

    for dct_debug_info in debug_info:
      self.add_debug_info(**dct_debug_info)

    last_witness_image = self._create_witness_image(
      original_image=original_image,
      debug_results=debug_results,
      object_detector_inferences=object_detector_inferences,
    )

    return last_witness_image

  def _print_alert_status_changed(self, current_alerter_status):
    if current_alerter_status not in ["OK", "Alert"]:
      color = None
      if current_alerter_status in ["Raised", "Re-Raised", "Forced Lower"]:
        color = "r"
      self.P("Alerter status changed to: {}".format(current_alerter_status), color=color)
    return

  def _process(self):
    payload = None

    self.__last_capture_image = self.dataapi_image()

    object_detector_inferences = self.dataapi_inferences()[self._get_detector_ai_engine()][0]
    object_detector_inferences = self._intersect_tlbrs_with_target_zone(object_detector_inferences)

    # update the people cooldown counter
    # reset the counter to config value if we have people in the image
    # decrease the counter if we do not have people in the image
    self._maybe_update_people_cooldown_counter(self.__last_capture_image, object_detector_inferences)

    # save the anchor if just started (anchor is None) or if we are forced to save it (force_save_anchor is True)
    # otherwise save a new anchor after comparison
    if self._anchor is None or self._force_save_anchor:
      self._maybe_save_anchor(self.__last_capture_image, object_detector_inferences)

    # if we do not have an anchor, we wait for it
    if self._anchor is None:
      return

    # process the current image
    result, debug_results = self.compare_image_with_anchor(self.__last_capture_image)

    ###### START DEBUG INFO AND CACHING ######
    debug_info = self._prepare_debug_info_for_witness(
      img=self.__last_capture_image,
      object_detector_inferences=object_detector_inferences,
      debug_results=debug_results,
    )

    # prepare the witness image used for the GET_LAST_WITNESS command
    self._update_cache_last_witness_image(
      img=self.__last_capture_image,
      object_detector_inferences=object_detector_inferences,
      debug_info=debug_info,
      debug_results=debug_results,
    )

    if self.cfg_demo_mode:
      # save results in the cache -- this will be used for the witness image when the alerter status changes
      self._update_cache_witness_image(
        img=self.__last_capture_image,
        object_detector_inferences=object_detector_inferences,
        debug_info=debug_info,
        result=result,
        debug_results=debug_results,
      )
    # endif
    ###### END DEBUG INFO AND CACHING ######

    # if result is None, then the comparison failed
    if result is None:
      # we do not have a valid comparison, so
      # we do not go further with the alerter and payload logic
      return

    result_similar = result < self.cfg_alert_raise_value
    validation_ok = self._validate_image_for_analysis(self.__last_capture_image, object_detector_inferences)

    if not result_similar and not validation_ok:
      # found people in image and comparison shows a change
      # since we cannot decide if people are the cause of the change
      # we skip adding the observation to the alerter
      return
    else:
      # because we consider this observation in the alerter 
      # we update the cache with the current image
      # before we add the observation to the alerter
      # otherwise, if (
      #   result < self.cfg_alert_raise_value and
      #   result > self.cfg_alert_lower_value and
      #   and alerter_is_new_raise
      # )
      # the current frame will be added in the alertable cache,
      # even though (result < self.cfg_alert_raise_value)
      self._update_cache_witness_image(
        img=self.__last_capture_image,
        object_detector_inferences=object_detector_inferences,
        debug_info=debug_info,
        result=result,
        debug_results=debug_results,
      )

      # either found people in image and comparison shows no change
      # or no people in the image and comparison shows a change
      # or no people in the image and comparison shows no change
      # we add the observation to the alerter
      self.alerter_add_observation(result)
    # endif

    alerter_status_changed, current_alerter_status = self._custom_alerter_status_changed()

    if alerter_status_changed or self.cfg_demo_mode:
      self._print_alert_status_changed(current_alerter_status)

      payload = self._get_payload(
        alerter_status=current_alerter_status,
      )
    # endif

    if result_similar and validation_ok:
      # no people in the image and comparison shows no change
      # we can save the anchor

      # this method validates both the comparison and the people in the image
      # so it might not be necessary to validate the image again
      self._maybe_save_anchor(self.__last_capture_image, object_detector_inferences)
    # endif

    return payload

  def _on_command(self, data, get_last_witness=None, reset_anchor=None, **kwargs):
    super(CvImageAnchorComparisonPlugin, self)._on_command(data, **kwargs)

    if (isinstance(data, str) and data.upper() == 'GET_LAST_WITNESS') or get_last_witness:
      last_witness = self.get_last_witness_image()
      self.add_payload_by_fields(
        img=[last_witness, self.__last_capture_image, self._anchor],
        command_params=data,
      )
    if (isinstance(data, str) and data.upper() == 'RESET_ANCHOR') or reset_anchor:
      self.P("Received command to reset anchor. This will also reset the alerter.")
      self._anchor = None
      self._anchor_last_save_time = 0  # is this necessary?
      self._force_save_anchor = True
      self.alerter_hard_reset()
    return

  # Can be overridden by the user
  @property
  def anchor_reload_period(self):
    """
    The serving anchor reload period. This is the period after which the anchor will be reloaded.
    This property can be used to add custom logic for the anchor reload period.
    (Like in the model based comparison, where this period is taken from the serving config)

    Returns
    -------
    int
        The anchor reload period
    """
    return self.cfg_anchor_reload_period

  def custom_load_anchor(self):
    """
    Custom anchor loading. This method can be used to add custom anchor loading logic.

    Returns
    -------
    (anchor, anchor_save_time) : (np.ndarray, int) | None
        anchor: The anchor image
        anchor_save_time: The last time the anchor was saved
    """
    # method must return an image that will be used as anchor and the last anchor reload time
    return None

  def custom_save_anchor(self, image):
    """
    Custom anchor saving. This method can be used to add custom anchor saving logic.

    Parameters
    ----------
    image : np.ndarray
        The image to be saved as anchor
    """
    return

  def custom_image_validation(self, image):
    """
    Custom image validation. This method can be used to add custom validation rules.
    This method is called after the default image validation.

    Parameters
    ----------
    image : np.ndarray
        The image to be validated

    Returns
    -------
    bool
        True if the image is valid, False otherwise
    """
    return True

  def compare_image_with_anchor(self, image):
    """
    Compare the current image with the anchor image

    Parameters
    ----------
    image : np.ndarray
        The image to be compared with the anchor

    Returns
    -------
    (result, debug_results) : Tuple[float, dict] | None
        result: The result of the comparison
        debug_results: The debug information to be added to the witness
    """
    return None
