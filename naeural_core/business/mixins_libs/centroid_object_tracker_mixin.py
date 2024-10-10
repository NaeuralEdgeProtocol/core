from scipy.spatial import distance as dist
from collections import OrderedDict

from naeural_core.utils.sort import *
from naeural_core import constants as ct


class _CentroidObjectTrackerMixin(object):
  def __init__(self):
    self.nextObjectID = 0
    self.objects = OrderedDict()
    self.disappeared = OrderedDict()
    self.sort_tracker = None
    super(_CentroidObjectTrackerMixin, self).__init__()
    return

  @property
  def cfg_tracking_mode(self):
    return self._instance_config.get('TRACKING_MODE', 0)

  @property
  def cfg_linear_max_age(self):
    return self._instance_config.get('LINEAR_MAX_AGE', 2)

  @property
  def cfg_sort_min_hits(self):
    return self._instance_config.get('SORT_MIN_HITS', 1)

  @property
  def cfg_sort_max_age(self):
    return self._instance_config.get('SORT_MAX_AGE', 3)

  @property
  def cfg_sort_min_iou(self):
    return self._instance_config.get('SORT_MIN_IOU', 0)

  @property
  def cfg_linear_reset_minutes(self):
    return self._instance_config.get('linear_reset_minutes', 60)

  def _maybe_init(self):
    if self.sort_tracker:
      return
    
    self.sort_tracker = Sort(
      min_hits=self.cfg_sort_min_hits,
      max_age=self.cfg_sort_max_age,
      iou_threshold=self.cfg_sort_min_iou
    )
    return

  def register(self, centroid, rectangle):
    self.objects[self.nextObjectID] = {
      'centroid': centroid,
      'rectangle': [rectangle[0], rectangle[1], rectangle[2], rectangle[3]],
      'color': ct.RED
    }
    self.disappeared[self.nextObjectID] = 0
    self.nextObjectID += 1

  def deregister(self, objectID):
    # to deregister an object ID we delete the object ID from
    # both of our respective dictionaries
    del self.objects[objectID]
    del self.disappeared[objectID]

  def update_tracker(self, rectangles):
    self.start_timer('update_tracker')
    res = None
    if self.cfg_tracking_mode == 0:
      res = self.update_linear(rectangles)
    elif self.cfg_tracking_mode == 1:
      res =  self.update_sort(rectangles)
    elif self.cfg_tracking_mode == 2:

      linear_results = self.update_linear(rectangles)
      sort_results = self.update_sort(rectangles)
      self.P("   WARNING - TRACKING MODE 3 SHOULD BE USED ONLY FOR DEBUGGING PURPOSES")

      res = {**linear_results, **sort_results}

    else:
      self.stop_timer('update_tracker')
      raise NotImplementedError("Tracking mode {} not implemented".format(self.cfg_tracking_mode))
    #endif
    self.stop_timer('update_tracker')
    return res

  def update_sort(self, rectangles):
    self._maybe_init()
    self.start_timer('update_sort')
    
    if len(rectangles.shape) > 1:
      rectangles = np.concatenate([rectangles, np.ones((rectangles.shape[0], 1))], axis=-1)
    else:
      rectangles = np.empty((0,5))

    res = self.sort_tracker.update(rectangles)
    dct_objects = {
      str(x[4]) + "_SORT": {'rectangle': [x[0], x[1], x[2], x[3]],
                  'color': ct.DARK_GREEN
                  } for x in res
    }
    self.stop_timer('update_sort')
    return dct_objects

  def update_linear(self, rectangles):
    self.start_timer('update_linear')
    if len(rectangles) == 0:

      for objectID in list(self.disappeared.keys()):
        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.cfg_linear_max_age:
          self.deregister(objectID)
      self.stop_timer('update_linear')
      return self.objects
    #endif
    
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
      D = dist.cdist(np.array(objectCentroids), inputCentroids)

      # Get the closest past object to each of the current ones
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]

      # cols = D.min(axis=0).argsort()
      # rows = D.argmin(axis=0)[cols]

      for (row, col) in zip(rows, cols):
        if row in usedRows or col in usedCols:
          continue
        objectID = objectIDs[row]
        self.objects[objectID]['centroid'] = inputCentroids[col]
        self.objects[objectID]['rectangle'] = [rectangles[col][0], rectangles[col][1], rectangles[col][2], rectangles[col][3]]
        self.disappeared[objectID] = 0
        usedRows.add(row)
        usedCols.add(col)
      #endfor
      unusedRows = set(range(0, D.shape[0])).difference(usedRows)
      unusedCols = set(range(0, D.shape[1])).difference(usedCols)


      for row in unusedRows:
        objectID = objectIDs[row]

        self.disappeared[objectID] += 1
        if self.disappeared[objectID] > self.cfg_linear_max_age:
          self.deregister(objectID)
        #endif
      # else:
      for col in unusedCols:
        # print("registering col {}".format(col))
        self.register(inputCentroids[col], rectangles[col])
      #endfor
    #endif

    returned_objects = self.objects.copy()
    for row in unusedRows:
      try:
        returned_objects.pop(objectIDs[row])
      except Exception as e:
        pass
    self.stop_timer('update_linear')
    return returned_objects

  def _draw_witness_image(self, img, dct_persons, **kwargs):

    for res in dct_persons:
      top, left, bottom, right = dct_persons[res]['rectangle']
      lbl = "Person id: {}".format(res)
      color = ct.RED

      img = self._painter.draw_detection_box(
        image=img,
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        label=lbl,
        prc=None,
        color=dct_persons[res]['color']
      )
    return img
