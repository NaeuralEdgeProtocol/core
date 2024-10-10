import numpy as np
from naeural_core import constants as ct
from datetime import datetime
from naeural_core.utils.centroid_object_tracker import CentroidObjectTracker
from decentra_vision import geometry_methods as gmt

class _MultiClassTrackerMixin(object):
  def __init__(self):
    self._object_trackers = {}
    super(_MultiClassTrackerMixin, self).__init__()

  def _match_tracking_results(self, img_inference, img_tracking_results, valid_class):
    """
    Matches all inferences to the returned objects of the tracker, then updates the inference dictionary with required
      info
    """
    for inference in img_inference:
      for track_id, object_info in img_tracking_results.items():
        if np.all(np.array(self.get_inference_track_tlbr(inference)).astype(np.int32) == np.array(object_info['rectangle']).astype(np.int32))\
                and self.__get_track_class(inference) == valid_class:
          inference[ct.TRACK_ID] = track_id
          inference[ct.TRACKING_STARTED_AT] = object_info['first_update']
          inference[ct.TRACKING_ALIVE_TIME] = (
                  datetime.strptime(object_info['last_update'], ct.BASE_CT.EE_DATE_TIME_FORMAT) - datetime.strptime(object_info['first_update'], ct.BASE_CT.EE_DATE_TIME_FORMAT)
          ).seconds
          inference[ct.APPEARANCES] = object_info['appearances']
        #endif
      #endfor
    #endfor

    return

  def __get_track_class(self, inference):
    """
    Returns the class of the object that will be tracked.
    If the object has a meta_type it will be used, otherwise the type will be used.
    In case neither is present, None will be returned.
    Parameters
    ----------
    inference - dict, inference dictionary

    Returns
    -------
    res - str, class of the object that will be tracked
    """
    if self.const.META_TYPE in inference:
      return inference[self.const.META_TYPE]
    if self.const.TYPE in inference:
      return inference[self.const.TYPE]
    return None

  def get_tracking_type(self, inf):
    """
    Public method for accessing the tracking type of inference
    Parameters
    ----------
    inf - dict, inference dictionary

    Returns
    -------
    res - str, tracking type of the inference
    """
    return self.__get_track_class(inf)

  def get_inference_track_tlbr(self, inference):
    """
    Returns the TLBR that will be used for tracking an inference
    This is used in order for the developer to be able to use a different TLBR for tracking
    than the actually detected one (maybe an enlarged one)
    Parameters
    ----------
    inference - dict, inference dictionary

    Returns
    -------
    res - list, list of 4 ints representing the TLBR that will be used for tracking
    """
    return inference.get(ct.TLBR_POS_TRACK, inference[ct.TLBR_POS])

  def _track_objects(self, dct_inference, img_shape=None):
    """
    Method used to track inferences. Different meta_types (if defined) or types (if meta types not defined) use different
      trackers
    """
    if self.cfg_tracking_enabled:
      self.start_timer('obj_track')
      detector_ai_engines = self._get_detector_ai_engines()
      inferences = []
      for engine, engine_inferences in dct_inference.items():
        if engine in detector_ai_engines:
          inferences += engine_inferences

      valid_classes = set(
        self.log.flatten_2d_list(
          [[self.__get_track_class(x) for x in inference if self.__get_track_class(x) is not None] for inference in inferences]
        )
      )
      for class_name in self._object_trackers.keys():
        if class_name not in valid_classes:
          valid_classes.add(class_name)
        # endif class not present in current frame
      # endfor all the previously seen classes
      for valid_class in valid_classes:
        if valid_class not in self._object_trackers.keys():
          if img_shape is None:
            img_shape = self.dataapi_image().shape
          self._object_trackers[valid_class] = CentroidObjectTracker(
            object_tracking_mode=self.cfg_tracking_mode,
            linear_max_age=self.cfg_linear_max_age,
            linear_max_distance=np.sqrt(img_shape[0] * img_shape[1]) / self.cfg_linear_max_distance_ratio,
            linear_max_relative_distance=self.cfg_linear_max_relative_dist,
            sort_min_hits=self.cfg_sort_min_hits,
            sort_max_age=self.cfg_sort_max_age,
            sort_min_iou=self.cfg_sort_min_iou,
            max_dist_scale=self.cfg_linear_max_dist_scale,
            center_dist_weight=self.cfg_linear_center_dist_weight,
            hw_dist_weight=self.cfg_linear_hw_dist_weight,
            linear_reset_minutes=self.cfg_linear_reset_minutes,
            moved_delta_ratio=0.005 ## TODO;
          )
        # endif

        for img_inference in inferences:
          filtered_img_inferences = [
            inference for inference in img_inference
            if self.__get_track_class(inference) == valid_class
          ]
          np_inferences = np.array([
            self.get_inference_track_tlbr(inference) for inference in filtered_img_inferences
          ])
          img_tracking_results = self._object_trackers[valid_class].update_tracker(
            np_inferences
          )
          self._match_tracking_results(img_inference, img_tracking_results, valid_class)
          self._object_trackers[valid_class].add_to_type_history(filtered_img_inferences)
        #endfor
      self.end_timer('obj_track')
    #endif
    return dct_inference

  def _get_tracker(self, object_type):
    """
    :param object_type: type or metatype of the tracker we need
    :return: tracker object used for the specified type/metatype
    """
    if object_type in self._type_to_meta_type_map:
      object_type = self._type_to_meta_type_map[object_type]

    return self._object_trackers.get(object_type, None)

  def _get_object_appearances(self, object_id, object_type):
    """
    Return number of appearances for an object
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_object_history(object_id) if tracker is not None else 0

  def _track_in_zone_objects(self, dct_inference):
    """
    Method used to both update the time in target zone and add it to
    the inference dictionary.
    Parameters
    ----------
    dct_inference - dictionary of inference

    Returns
    -------
    dct_inference - dictionary of inference containing the updated time in target zone
    """
    if self.cfg_tracking_enabled:
      self.start_timer('obj_track_in_zone')
      inferences = dct_inference.get(self._get_detector_ai_engine(), [])

      for obj_class, obj_tracker in self._object_trackers.items():
        for img_inference in inferences:
          filtered_img_inferences = [
            inference for inference in img_inference
            if self.__get_track_class(inference) == obj_class
          ]
          obj_tracker.update_in_zone_history(filtered_img_inferences)
          for inference in filtered_img_inferences:
            inference[ct.TIME_IN_TARGET] = self.trackapi_in_zone_total_seconds(
              object_id=inference[ct.TRACK_ID],
              object_type=obj_class
            )
          # endfor filtered_img_inferences
        # endfor img_inference
      # endfor class trackers

      self.end_timer('obj_track_in_zone')
    # endif tracking enabled
    return dct_inference

  def _get_object_history(self, object_id, object_type):
    """
    Return centroid object history
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_object_history(object_id) if tracker is not None else []

  def _get_object_type_history(self, object_id, object_type):
    """
    :param object_id:
    :param object_type:
    :return:
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_object_type_history(object_id) if tracker is not None else []

  def _get_object_type_history_deque(self, object_id, object_type):
    """
    :param object_id:
    :param object_type:
    :return:
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_object_type_history_deque(object_id) if tracker is not None else []

  def _get_type_history_info(self, object_id, object_type):
    type_history = self._get_object_type_history(object_id, object_type)
    type_history_deque = self._get_object_type_history_deque(object_id, object_type)
    deque_info = [class_name[0] for class_name in type_history_deque]
    dct_string = " ".join([":".join([k[:2], str(v)]) for k, v in type_history.items()])
    info_str = f'Freq: {dct_string} | Last: {"".join(deque_info)}'
    return info_str

  def _get_object_class_count(self, object_id, object_type, object_subtype):
    """
    :param object_id:
    :param object_type:
    :param object_subtype:
    :return:
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_class_count(object_id, object_subtype) if tracker is not None else []

  def _get_object_non_class_count(self, object_id, object_type, object_subtype):
    """
    :param object_id:
    :param object_type:
    :param object_subtype:
    :return:
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_non_class_count(object_id, object_subtype) if tracker is not None else []

  def _get_object_class_ratio(self, object_id, object_type, object_subtype):
    """
    :param object_id:
    :param object_type:
    :param object_subtype:
    :return:
    """
    tracker = self._get_tracker(object_type)
    return tracker.get_class_ratio(object_id, object_subtype) if tracker is not None else []

  def _get_object_absolute_orientation(self, object_id, object_type):
    """
    Returns a vector from the original object position to the current object position
    """
    tracker = self._get_tracker(object_type)
    assert tracker is not None, "No tracker found for object type '{}'".format(object_type)
    original_position = tracker.get_original_position(object_id)
    last_position = tracker.get_object_history(object_id)[-1]
    return (original_position, last_position)

  def _get_current_orientation(self, object_id, object_type, number_of_points=3):
    """
    Returns a vector that has the orientation of the most recent change of steps
    """
    tracker = self._get_tracker(object_type)
    assert tracker is not None, "No tracker found for object type '{}'".format(object_type)
    positions = tracker.get_object_history(object_id)[-number_of_points:]
    point1 = positions[0]
    point2 = positions[-1]
    return (point2[0] - point1[0], point2[1] - point1[1])

  def _distance_point_to_line(self, point, line):
    """
    Returns the distance from a point to a line
    """
    line_point_1, line_point_2 = line

    result = (
            (line_point_2[0] - line_point_1[0]) * (line_point_1[1] - point[1]) - \
            (line_point_1[0]-point[0]) * (line_point_2[1] - line_point_1[1])
    ) / (
      np.sqrt((line_point_2[0] - line_point_1[0]) ** 2 + (line_point_2[1] - line_point_1[1]) ** 2)
    )
    return result

  def _get_zone_points(self, line):
    """
    Generates two points on different parts of a line
    """
    center = (line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2.
    var_x = (line[0][0] - line[1][0]) // 2
    var_y = (line[0][1] - line[1][1]) // 2

    zone_1_point = int(center[0] + var_y), int(center[1] - var_x)
    zone_2_point = int(center[0] - var_y), int(center[1] + var_x)

    return zone_1_point, zone_2_point

  def get_movement_relative_to_line(
      self, object_id, object_type,
      line, zone1_point=None,
      zone2_point=None, threshold=10,
      start_point=None
  ):
    """
    Returns the point direction movement relative to a line (if no points are given, they are automatically generation).

    If the object moved from point A to point B (relative to a line) it returns a tuple with the order (PointA, PointB),
      otherwise it returns the tuple reversed (PointB, PointA)
    """
    first, last = self._get_object_absolute_orientation(
      object_id,
      object_type
    )
    # In case no starting point is provided we will consider the first position known of the current object as the start
    if start_point is None:
      start_point = first

    distance_1 = self._distance_point_to_line(start_point, line)
    distance_2 = self._distance_point_to_line(last, line)

    distance_diff = distance_1 - distance_2
    if abs(distance_diff) < threshold:
      return None

    distance_sign = distance_diff < 0

    if zone1_point is None or zone2_point is None:
      zone1_point, zone2_point = self._get_zone_points(line)

    distance_zone1 = self._distance_point_to_line(zone1_point, line)
    distance_zone2 = self._distance_point_to_line(zone2_point, line)

    zone1_sign = distance_zone1 < 0
    zone2_sign = distance_zone2 < 0

    assert zone1_sign != zone2_sign, "Zone points are in the same delimited area"

    if distance_sign == zone1_sign:
      return (zone1_point, zone2_point)

    return (zone2_point, zone1_point)

  def get_line_passing_direction(
      self, object_id, object_type,
      line, zone1_point=None,
      zone2_point=None, start_point=None,
      eps=0.00001
  ):
    """
    This method will compute the direction of an object if the specified object went from Zone1
    to Zone2, where Zone1 and Zone 2 ar separated by a specified line.
    Parameters
    ----------
    object_id - int, id of the specified object
    object_type - str, type of the specified object
    line - list, list of 2 points that describes the line separating the 2 zones
    zone1_point - list or None, list of 2 int/floats describing a point from Zone1
    - if this is None both zone1_point and zone2_point will be auto-generated
    zone2_point - list or None, list of 2 int/floats describing a point from Zone2
    - if this is None both zone1_point and zone2_point will be auto-generated
    start_point - list or None, list of 2 int/floats describing the starting point of the current object
    - if this is None, we will consider the first appearance of the current object as the start

    Returns
    -------
      None if the object stayed in one zone
      (0, 1) or (1, 0) if the object passed through the line,
      with (0, 1) indicating it passing from Zone1 to Zone2
      and (1, 0) otherwise
    """
    first, last = self._get_object_absolute_orientation(
      object_id,
      object_type
    )
    # In case no starting point is provided we will consider the first position known of the current object as the start
    if start_point is None:
      start_point = first

    distance_1 = self._distance_point_to_line(start_point, line)
    distance_2 = self._distance_point_to_line(last, line)

    if distance_1 * distance_2 >= 0:
      return None

    A, B = self.get_movement_relative_to_line(
      object_id=object_id,
      object_type=object_type,
      line=line,
      zone1_point=zone1_point,
      zone2_point=zone2_point,
      threshold=0,
      start_point=start_point
    )

    distance = gmt.euclidian_distance(self.np.array(A), self.np.array(zone1_point))
    # to check if A and zone1_point are the same point we check if the distance between them is below eps
    if distance < eps:
      # A and zone1_point are the same so the object passed from Zone1 to Zone2
      return 0, 1
    # A and zone1_point are not the same so the object passed from Zone2 to Zone1
    return 1, 0

  """TRACKAPI SECTION"""
  if True:
    def trackapi_in_zone_total_seconds(self, object_id, object_type):
      """
      Public method for accessing the total seconds spent in the target zone by a specified object.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - int, total number of seconds spent in the target zone
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_in_zone_total_seconds_additional(object_id=object_id)

    def trackapi_in_zone_history(self, object_id, object_type):
      """
      Public method for accessing the history in the target zone of a specified object.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - list, list of intervals that the specified object was in the target zone.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_in_zone_history(object_id=object_id)

    def trackapi_centroid_history(self, object_id, object_type):
      """
      Public method for accessing the centroid history of a specified object.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - list, list of points that signify the provided object's centroid on each appearance.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_object_history(object_id=object_id)

    def trackapi_type_history(self, object_id, object_type):
      """
      If meta-types are not used than this will just provide the object's number of appearances.
      Public method for accessing the type history of a specified object as a frequency dictionary.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - dict, dictionary providing the number of times the current object appeared as a certain class.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_object_type_history(object_id=object_id)

    def trackapi_type_history_deque(self, object_id, object_type):
      """
      If meta-types are not used than this will just provide a list full of the same type.
      Public method for accessing the type history of a specified object as a deque.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - deque, list of the type that the current object was at each appearance
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_object_type_history_deque(object_id=object_id)

    def trackapi_most_seen_type(self, object_id, object_type):
      """
      Public method for accessing the most seen type of specified object.
      If meta-types are not used than this will just provide the object's type.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - str, most seen type of the specified object
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_most_seen_type(object_id=object_id)

    def trackapi_class_count(self, object_id, object_type, class_name):
      """
      If meta-types are not used than this will just provide the number of appearances.
      Public method for accessing how many times the object was a certain class.
      Parameters
      ----------
      object_id - int
      object_type - str
      class_name - str or list

      Returns
      -------
      res - int, how many times the object was a certain class.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_class_count(object_id=object_id, class_name=class_name)

    def trackapi_non_class_count(self, object_id, object_type, class_name):
      """
      If meta-types are not used than this will just provide 0.
      Public method for accessing how many times the object was not a certain class.
      Parameters
      ----------
      object_id - int
      object_type - str
      class_name - str or list

      Returns
      -------
      res - int, how many times the object was not a certain class.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_non_class_count(object_id=object_id, class_name=class_name)

    def trackapi_class_ratio(self, object_id, object_type, class_name):
      """
      If meta-types are not used than this will just provide 1.
      Public method for accessing the ratio between how many times the object was
      a certain class and the total number of appearances.
      Parameters
      ----------
      object_id - int
      object_type - str
      class_name - str or list

      Returns
      -------
      res - float, ratio of class appearances.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_class_ratio(object_id=object_id, class_name=class_name)

    def trackapi_original_position(self, object_id, object_type):
      """
      Public method for accessing the original position of a specified object.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - list, centroid of the current object on its first appearance.
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_original_position(object_id=object_id)

    def trackapi_max_movement(self, object_id, object_type, steps=None, method='l2'):
      """
      Public method for accessing the maximum distance between the
      original position of a specified object and all of its following centroids.
      Parameters
      ----------
      object_id - int
      object_type - str
      steps - int or None, how much further back to check for movement
      - if None this will compute the max movement for the entire history
      - if x - int, this will compute the max movement for the last k steps
      method - str, method used for computing the distance
      - if 'l1' this will return the 'l1' distance
      - if 'l2' this will return the 'l2' distance

      Returns
      -------
      res - int or float, max distance the specified object was from its original position

      """
      tracker = self._get_tracker(object_type)
      return tracker.get_object_max_movement(object_id=object_id, steps=steps, method=method)

    def trackapi_last_rectangle(self, object_id, object_type):
      """
      Public method for accessing the last seen rectangle of a specified object.
      Parameters
      ----------
      object_id - int
      object_type - str

      Returns
      -------
      res - list, last seen rectangle of the specified object in format [top, left, bottom, right]
      """
      tracker = self._get_tracker(object_type)
      return tracker.get_last_rectangle(object_id=object_id)

  """END TRACKAPI SECTION"""
