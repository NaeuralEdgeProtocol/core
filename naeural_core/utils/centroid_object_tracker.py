from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import numpy as np

from naeural_core.utils.sort import *
from datetime import datetime
from naeural_core import constants as ct
from decentra_vision import geometry_methods as gmt


class CentroidObjectTracker(object):
  def __init__(
      self, object_tracking_mode=0, linear_max_age=4, linear_max_distance=300, linear_max_relative_distance=1.2,
      max_dist_scale=1.4, center_dist_weight=1, hw_dist_weight=0.8, sort_min_hits=1, sort_max_age=3, sort_min_iou=0,
      moved_delta_ratio=0.005, linear_reset_minutes=60, **kwargs
  ):
    """
    :param object_tracking_mode: int
      0: linear - default
      1: sort - not recommended
      2: both - Only for debug

    :param linear_max_age:
      Used for linear. For how long (no frames) to remember an object if it not found in the scene

    :param linear_max_distance:
      Max value of the distance function between 2 objects in order to have the same id

    :param linear_max_relative_distance:
      Max relative distance allowed between 2 consecutive appearances of the same object.
      The distance between the current candidate and the last centroid will be scaled to the maximum dimension(width or height)
      of both the current candidate and the last rectangle. If that distance is too high, the candidate will not be able
      to get the checked object id.

    :param max_dist_scale:
      Parameter used to compute maximum distance per distance type.  Formula is:
        distance_type_max = max_distance * dist_weight / (hw_dist_weight + center_dist_weight) * max_dist_scale.
      Should have values between 1 and 2

    :param center_dist_weight:
      The weight of the centroid distance in computing the total distance

    :param hw_dist_weight:
     The weight of the sizes distance in computing the total distance

    :param sort_min_hits: int,
      Number of identifications to be assigned an id

    :param sort_max_age: int,
      Number of untracked frames until the object id is no loger tracked.

    :param sort_min_iou: float,
      The minimum IOU between objects from different frames to be considered tracking candidates

    :param moved_delta_ratio: To be implemented

    :param linear_reset_minutes: int
      The maximum number of minutes for an object to be tracked.

    :param kwargs:
    """
    self.nextObjectID = 0
    self.objects = OrderedDict()
    self.disappeared = OrderedDict()
    self.sort_tracker = None

    self.center_dist_weight = center_dist_weight
    self.hw_dist_weight = hw_dist_weight
    self.linear_max_distance = linear_max_distance
    self.linear_max_relative_distance = linear_max_relative_distance
    self.linear_max_distance_center = linear_max_distance * center_dist_weight / (hw_dist_weight + center_dist_weight) * max_dist_scale
    self.linear_max_distance_hw = linear_max_distance * hw_dist_weight / (hw_dist_weight + center_dist_weight) * max_dist_scale

    self.linear_reset_minutes = linear_reset_minutes

    self.object_tracking_mode = object_tracking_mode
    self.linear_max_age = linear_max_age
    self.sort_min_hits = sort_min_hits
    self.sort_max_age = sort_max_age
    self.sort_min_iou = sort_min_iou

    self.objects_history = OrderedDict()
    self.moved_delta_ratio = moved_delta_ratio ### TODO: implement
    super(CentroidObjectTracker, self).__init__(**kwargs)
    return

  def _maybe_init(self):
    if self.sort_tracker:
      return

    self.sort_tracker = Sort(
      min_hits=self.sort_min_hits,
      max_age=self.sort_max_age,
      iou_threshold=self.sort_min_iou
    )
    return

  def register(self, centroid, rectangle):
    """
    Creates a new tracked object and assigns it an id
    """
    self.objects[self.nextObjectID] = {
      'appearances': 0,
      'centroid': centroid,
      'rectangle': [rectangle[0], rectangle[1], rectangle[2], rectangle[3]],
      'color': ct.RED,
      'first_update': datetime.now().strftime(ct.BASE_CT.EE_DATE_TIME_FORMAT),
      'last_update': datetime.now().strftime(ct.BASE_CT.EE_DATE_TIME_FORMAT),
      'centroid_history': deque([centroid], maxlen=1000),
      'original_position': centroid,
      'type_history': {'total': 0},
      'type_history_deque': deque(maxlen=100),
      'in_zone_history': deque(maxlen=1000),
      'in_zone_total_seconds': 0
    }
    self.disappeared[self.nextObjectID] = 0
    self.nextObjectID += 1

  def deregister(self, objectID):
    """
    Removes a tracked object and moves its data to the `objects_history` dict
    """
    # to deregister an object ID we delete the object ID from
    # both of our respective dictionaries
    self.objects_history[objectID] = self.objects[objectID]
    del self.objects[objectID]
    del self.disappeared[objectID]

  def reset_old_objects(self):
    """
    Removes objects that have been tracked for more than `self.linear_reset_minutes` minutes
    """
    to_be_removed = []
    for object_key in self.objects.keys():
      if (datetime.now() - datetime.strptime(self.objects[object_key]['first_update'], ct.BASE_CT.EE_DATE_TIME_FORMAT)).seconds / 60 > self.linear_reset_minutes:
        #self.deregister(object_key)
        to_be_removed.append(object_key)
      #endif
    #endfor
    for object_key in to_be_removed:
      self.deregister(object_key)

  def update_tracker(self, rectangles):
    """
    The main tracking method.

    Takes as parameter an array of bounding boxes, and matches them with past tracked objects
    """
    res = None
    if self.object_tracking_mode == 0:
      res = self.update_linear(rectangles)
    elif self.object_tracking_mode == 1:
      res = self.update_sort(rectangles)
    elif self.object_tracking_mode == 2:

      linear_results = self.update_linear(rectangles)
      sort_results = self.update_sort(rectangles)
      print("   WARNING - TRACKING MODE 2 SHOULD BE USED ONLY FOR DEBUGGING PURPOSES")

      res = {**linear_results, **sort_results}

    else:
      raise NotImplementedError("Tracking mode {} not implemented".format(self.object_tracking_mode))
    # endif
    return res

  def update_sort(self, rectangles):
    """
    Tracks objects using the sort algorithm
    https://github.com/abewley/sort

    """
    self._maybe_init()

    if len(rectangles.shape) > 1:
      rectangles = np.concatenate([rectangles, np.ones((rectangles.shape[0], 1))], axis=-1)
    else:
      rectangles = np.empty((0, 5))

    res = self.sort_tracker.update(rectangles)
    dct_objects = {
      str(int(x[4])) + "_SORT": {'rectangle': [x[0], x[1], x[2], x[3]],
                            'color': ct.DARK_GREEN
                            } for x in res
    }
    return dct_objects

  def update_linear(self, rectangles):
    """
    Tracks objects using the linear algorithm:

    1. Remove old tracked objects
    2. If there is no new objects to be tracked, increments past objects age and removes them if needed and return
    3. Compute new objects centroids
    4. If there is no past tracked object, register all new objects and return
    5. Else we try to match all new objects to past tracked objects.
    5.1. Compute the HW distance (similarity) from all current boxes to all past ones
    5.2. Compute the Centroid distance from all current boxes to all past one
    5.3. Compute a weighted score of the 2 distances
    5.4. For each current object try to match it to the closest past object
    5.4.1. If the total distance is too big, or any of the distance components are too big, or the past object was
        already matched, then continue with no valid match found
    5.4.2. Else assign the current object the past object track id, and update the tracked object information
    5.5. Increase age and remove if necessary for all unmatched old objects
    5.6. Register new objects for all unmatched new objects
    """
    self.reset_old_objects()
    if len(rectangles) == 0:
      for objectID in list(self.disappeared.keys()):
        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.linear_max_age:
          self.deregister(objectID)
      return self.objects
    # endif

    # Compute centroids
    inputCentroids = np.zeros((len(rectangles), 2), dtype="int")
    for (i, (startX, startY, endX, endY)) in enumerate(rectangles):
      cX = int((startX + endX) / 2.0)
      cY = int((startY + endY) / 2.0)
      inputCentroids[i] = (cX, cY)

    usedRows = set()
    usedCols = set()
    unusedRows = set()
    unusedCols = set()
    objectIDs = list(self.objects.keys())

    if len(self.objects) == 0:
      for i in range(0, len(inputCentroids)):
        self.register(inputCentroids[i], rectangles[i])
    else:
      objectCentroids = [x['centroid'] for x in self.objects.values()]

      # Compute distance from past object centroids to current ones
      objectRectangles = np.array([x['rectangle'] for x in self.objects.values()])

      ## H/W - Box similarity
      objects_Hs = objectRectangles[:, 2] - objectRectangles[:, 0]
      inputs_Hs = rectangles[:, 2] - rectangles[:, 0]
      h_D = np.abs(objects_Hs.reshape((-1,1)) - inputs_Hs.reshape((-1,1)).T) # Transpose for broadcasting

      objects_Ws = objectRectangles[:, 3] - objectRectangles[:, 1]
      inputs_Ws = rectangles[:, 3] - rectangles[:, 1]
      w_D = np.abs(objects_Ws.reshape((-1,1)) - inputs_Ws.reshape((-1,1)).T) # Transpose for broadcasting

      hw_D = np.minimum(h_D, w_D)
      centroid_D = dist.cdist(np.array(objectCentroids), inputCentroids)

      D = (hw_D * self.hw_dist_weight + centroid_D * self.center_dist_weight) / (self.hw_dist_weight + self.center_dist_weight)


      # Get the closest past object to each of the current ones
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]

      for (row, col) in zip(rows, cols):
        ### Linear max distance is in pixels. To be modified for normed coords. The other 2 are auto computed
        if D[row, col] > self.linear_max_distance:
          continue
        if centroid_D[row, col] > self.linear_max_distance_center:
          continue
        if hw_D[row, col] > self.linear_max_distance_hw:
          continue
        if row in usedRows or col in usedCols:
          continue

        # Compute the maximum size(width or height) of both the current detection and the one we are trying
        # to match it with.
        max_size = max(
          objectRectangles[row][2] - objectRectangles[row][0],  # last height
          objectRectangles[row][3] - objectRectangles[row][1],  # last width
          rectangles[col][2] - rectangles[col][0],  # current height
          rectangles[col][3] - rectangles[col][1]  # current width
        )
        # check if the distance between the 2 centroids scaled using the previously computed max size is too large.
        if centroid_D[row, col] / max_size > self.linear_max_relative_distance:
          continue

        objectID = objectIDs[row]
        self.objects[objectID]['centroid_history'].append(inputCentroids[col])
        self.objects[objectID]['last_update'] = datetime.now().strftime(ct.BASE_CT.EE_DATE_TIME_FORMAT)
        self.objects[objectID]['appearances'] = len(self.objects[objectID]['centroid_history'])
        self.objects[objectID]['centroid'] = inputCentroids[col]
        self.objects[objectID]['rectangle'] = [rectangles[col][0], rectangles[col][1], rectangles[col][2],
                                               rectangles[col][3]]

        self.disappeared[objectID] = 0
        usedRows.add(row)
        usedCols.add(col)
      # endfor

      unusedRows = set(range(0, D.shape[0])).difference(usedRows)
      unusedCols = set(range(0, D.shape[1])).difference(usedCols)

      for row in unusedRows:
        objectID = objectIDs[row]

        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.linear_max_age:
          self.deregister(objectID)
        # endif
      # else:
      for col in unusedCols:
        self.register(inputCentroids[col], rectangles[col])
      # endfor
    # endif

    returned_objects = self.objects.copy()
    for row in unusedRows:
      try:
        returned_objects.pop(objectIDs[row])
      except Exception as e:
        pass

    return returned_objects

  def get_object_appearances(self, object_id):
    """
    Returns an object's number of appearances
    """
    if object_id in self.objects:
      return self.objects[object_id]['appearances']
    return self.objects_history[object_id]['appearances']

  def get_in_zone_history_deque(self, object_id):
    """
    Returns an object's in-zone history as deque
    """
    if object_id in self.objects:
      return self.objects[object_id]['in_zone_history']
    return self.objects_history[object_id]['in_zone_history']

  def get_in_zone_history(self, object_id):
    """
    Returns an object's in-zone history as list
    """
    return list(self.get_in_zone_history_deque(object_id=object_id))

  def get_in_zone_total_seconds(self, object_id):
    """
    Returns and object's in-zone total seconds without the last interval if the last interval is opened.
    Parameters
    ----------
    object_id - int, id of the specified object

    Returns
    -------
    res - int, number of seconds that the specified object was in the target zone
    """
    if object_id in self.objects:
      return self.objects[object_id]['in_zone_total_seconds']
    return self.objects_history[object_id]['in_zone_total_seconds']

  def set_in_zone_total_seconds(self, object_id, value):
    """
    Updates the total number of seconds that the specified object situated in the target zone.
    Parameters
    ----------
    object_id - int, id of the specified object
    value - new total seconds value

    Returns
    -------

    """
    if object_id in self.objects:
      self.objects[object_id]['in_zone_total_seconds'] = value
    else:
      self.objects_history[object_id]['in_zone_total_seconds'] = value
    return

  def get_in_zone_total_seconds_additional(self, object_id):
    """
    Returns and object's in-zone total seconds including the last interval.
    Parameters
    ----------
    object_id

    Returns
    -------
    res - int, number of seconds that the specified object was in the target zone
    """
    additional_seconds = 0
    in_zone_history = self.get_in_zone_history(object_id=object_id)
    if len(in_zone_history) > 0 and len(in_zone_history[-1]) < 2:
      # if the last in zone interval was not close that means the object is still in the target zone
      additional_seconds = (datetime.now() - in_zone_history[-1][0]).seconds
    # endif opened interval
    return self.get_in_zone_total_seconds(object_id=object_id) + additional_seconds

  def maybe_close_in_zone_interval(self, object_id):
    """
    This method will be used for objects that were seen before, but now are not.
    In case the last interval of in_zone_history is still opened (it has just the start)
    the current time will be added to it in order to mark it as closed and the total in_zone time
    will also be updated.
    Parameters
    ----------
    object_id - int, id of the specified object

    Returns
    -------

    """
    history_deque = self.get_in_zone_history_deque(object_id=object_id)
    if len(history_deque) > 0:
      if len(history_deque[-1]) < 2:
        last_start = history_deque[-1][0]
        now = datetime.now()
        current_total_time = self.get_in_zone_total_seconds(object_id=object_id)
        self.set_in_zone_total_seconds(object_id=object_id, value=current_total_time + (now - last_start).seconds)
        history_deque[-1].append(now)
      # endif last interval opened
    # endif non-empty in_zone history
    return

  def update_in_zone_history(self, in_zone_objects):
    """
    This method will update the in-zone time of all objects
    Parameters
    ----------
    in_zone_objects - list of dictionaries that describe objects left in an arbitrary determined target zone

    Returns
    -------

    """
    # First we mark all the objects that are now in zone and, if it's the case, open a new in_zone interval
    marked = set()
    for obj in in_zone_objects:
      track_id = obj[ct.TRACK_ID]
      if track_id not in marked:
        marked.add(track_id)
      # endif
      current_in_zone_history = self.get_in_zone_history_deque(object_id=track_id)
      if len(current_in_zone_history) < 1 or len(current_in_zone_history[-1]) > 1:
        current_in_zone_history.append([datetime.now()])
      # endif new interval
    # endfor in_zone_objects

    # Second we check to see if any object that was not seen in the current frame has an open interval
    for obj_id in self.objects.keys():
      if obj_id not in marked:
        self.maybe_close_in_zone_interval(object_id=obj_id)
      # endif unseen object
    # endfor objects
    return

  def get_object_history(self, object_id):
    """
    Returns an object's centroid history
    """
    if object_id in self.objects:
      return list(self.objects[object_id]['centroid_history'])
    return list(self.objects_history[object_id]['centroid_history'])

  def get_object_max_movement(self, object_id, steps=None, method='l2'):
    """
    Method for getting the maximum distance a certain object was from its original position.
    Parameters
    ----------
    object_id - int, id of the specified object
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
    lst_centroids = self.get_object_history(object_id=object_id)
    original_position = np.array(self.get_original_position(object_id=object_id))
    if steps is not None and isinstance(steps, int):
      lst_centroids = lst_centroids[len(lst_centroids) - steps:]
      original_position = np.array(lst_centroids[0])
    # endif steps valid

    centroids = np.array(lst_centroids)

    distances = gmt.distance_points_point(points=centroids, point=original_position, method=method)

    return np.max(distances)

  def get_object_type_history(self, object_id):
    """
    Returns an object's type history summary
    """
    if object_id in self.objects:
      return self.objects[object_id]['type_history']
    return self.objects_history[object_id]['type_history']

  def get_object_type_history_deque(self, object_id):
    """
    Returns an object's centroid history deque
    """
    if object_id in self.objects:
      return self.objects[object_id]['type_history_deque']
    return self.objects_history[object_id]['type_history_deque']

  def add_to_type_history(self, inferences):
    """
      TODO: docstring
      """
    for inference in inferences:
      type_history_deque = self.get_object_type_history_deque(inference[ct.TRACK_ID])
      type_history_deque.append(inference[ct.TYPE])
      type_history = self.get_object_type_history(inference[ct.TRACK_ID])
      type_history['total'] += 1
      if inference[ct.TYPE] not in type_history.keys():
        type_history[inference[ct.TYPE]] = 0
      type_history[inference[ct.TYPE]] += 1
    # endfor
    return

  def get_most_seen_type(self, object_id):
    """
    Returns the most seen type of object
    Parameters
    ----------
    object_id - int, id of the specified object

    Returns
    -------
    res - str, the most seen type of the specified object
    """
    type_history = self.get_object_type_history(object_id)
    best_type, best_count = '', 0
    for type_, count in type_history.items():
      # Skip the total count
      if type_ == 'total':
        continue
      if count > best_count:
        best_type, best_count = type_, count
    return best_type

  def get_class_count(self, object_id, class_name, return_complement=False, return_ratio=False):
    """
    Method for obtaining how many times an object was a certain type.
    Parameters
    ----------
    object_id - int, id of the specified object
    class_name - str or list, specified type
    if str it will be considered the name of a type.
    if list it will be considered a list of types.
    return_complement - bool
    if False this will return the number of times the object was the specified type.
    if True this will return the number of times the object was not the specified type.
    return_ratio - bool
    if True this will return the result normalised.
    if False this will return the result not normalised.

    Returns
    -------
    res - how many times the specified object was the specified type.
    """
    type_history = self.get_object_type_history(object_id)
    if isinstance(class_name, list):
      res = sum([type_history.get(x, 0) for x in class_name])
    else:
      res = type_history.get(class_name, 0)

    if return_complement:
      res = type_history['total'] - res

    if return_ratio:
      res = res / type_history['total']
    return res

  def get_non_class_count(self, object_id, class_name):
    """
    Method for obtaining how many times an object was not a certain type.
    Parameters
    ----------
    object_id - int, id of the specified object
    class_name - str or list, specified type
    if str it will be considered the name of a type
    if list it will be considered a list of types

    Returns
    -------
    res - how many times the specified object was not the specified type.
    """
    return self.get_class_count(object_id, class_name, return_complement=True)

  def get_class_ratio(self, object_id, class_name):
    """
    Method for obtaining the ratio between how many times the object was
      a certain class and the total number of appearances.
    Parameters
    ----------
    object_id - int, id of the specified object
    class_name - str or list, specified type
    if str it will be considered the name of a type
    if list it will be considered a list of types

    Returns
    -------
    res - how many times the specified object was not the specified type.
    """
    return self.get_class_count(object_id, class_name, return_ratio=True)

  def get_original_position(self, object_id):
    """
    Returns an object's original (first occurrence) centroid.
    Parameters
    ----------
    object_id - int, id of the specified object
    Returns
    -------
    res - tuple, original position of the specified object
    """
    if object_id in self.objects:
      return self.objects[object_id]['original_position']
    return self.objects_history[object_id]['original_position']

  def get_last_rectangle(self, object_id):
    """
    Returns an object's last seen rectangle.
    Parameters
    ----------
    object_id - int, id of the specified object
    Returns
    -------
    res - list, last seen rectangle of the specified object in format [top, left, bottom, right]
    """
    if object_id in self.objects:
      return self.objects[object_id]['rectangle']
    return self.objects_history[object_id]['rectangle']
