import numpy as np
import collections

from naeural_core import DecentrAIObject
from datetime import datetime as dt
from copy import deepcopy

class UTC:
  COUNT = 'COUNT'
  ASSOCIATED_OBJECT = 'ASSOC_OBJECT'
  MAIN_ATTR = 'MAIN_ATTR'
  ALL_ATTR = 'ALL_ATTR'
  LAST_SEEN = 'LAST_SEEN'


class UniversalTracker(DecentrAIObject):
  def __init__(self, log,
               func_distance,
               max_distance,
               flush_delta_seconds=None,
               section=None,
               **kwargs):
    """
    Parameters:
    -----------
    log : Logger, mandatory

    func_distance : callback, mandatory
      Callback that takes two params x and y and return certain distance between x and y.

    max_distance : int, mandatory
      The maximum distance (returned by `func_distance`) to consider in order to get top k similar entities

    flush_delta_seconds : int, optional
      How many seconds should pass between 2 "database" flushes
      Default to None (600)

    """

    if flush_delta_seconds is None:
      flush_delta_seconds = 600

    self.flush_delta_seconds = flush_delta_seconds
    self.func_distance = func_distance
    self.max_distance = max_distance

    ### Objects = what you are tracking (cars, people etc)
    ### Attributes = identification element of the objects (license plates for cars, faces for peolpe etc)

    # dict{String : {UTC.COUNT : Integer, UTC.ASSOCIATED_OBJECT : Integer}}
    self.dct_attr_metadata = {}

    # dict{Integer : {UTC.MAIN_ATTR : String, UTC.ALL_ATTR : Set{String}, UTC.LAST_SEEN : Datetime}
    self.dct_obj_metadata = {}

    self._dct_hashed_attrs = {}
    self._last_flush_time = dt.now()
    self._nr_objects = 0
    self._section = section
    super().__init__(log=log, prefix_log='[TRACK]', **kwargs)
    return

  def _initial_attr_metadata(self):
    return {UTC.COUNT : 0, UTC.ASSOCIATED_OBJECT : None}

  def _initial_obj_metadata(self):
    return {UTC.MAIN_ATTR : None, UTC.ALL_ATTR : set(), UTC.LAST_SEEN : None}

  def start_timer(self, tmr_id):
    self.log.start_timer(
      sname=tmr_id,
      section=self._section
    )
    return

  def end_timer(self, tmr_id, skip_first_timing=True, periodic=False):
    self.log.end_timer(
      sname=tmr_id,
      section=self._section,
      skip_first_timing=skip_first_timing,
      periodic=periodic
    )
    return

  def adjust_attribute(self, attribute, confidence=1):
    self.start_timer('adjust_attribute')
    if not isinstance(attribute, collections.abc.Hashable):
      hattribute = self.log.hash_object(attribute, size=4)
      self._dct_hashed_attrs[hattribute] = attribute
      attribute = hattribute
    #endif

    obj = self._attach_attribute_to_object(attribute)
    self._increment_counter(attribute, increase_value=confidence)
    self._update_object_last_seen(obj)
    adjusted_attr = self._maybe_adjust_object_main_attribute(obj)
    self._maybe_flush_unseen_objects()
    self.end_timer('adjust_attribute')
    return obj, adjusted_attr

  def _increment_counter(self, attribute, increase_value=1):
    new_count = self.dct_attr_metadata[attribute][UTC.COUNT] + increase_value
    self.dct_attr_metadata[attribute][UTC.COUNT] = new_count
    return

  def _update_object_last_seen(self, obj):
    self.dct_obj_metadata[obj]['LAST_SEEN'] = dt.now()
    return

  def _maybe_is_good_attribute(self, attribute):
    dct_main_attr_to_obj = {v[UTC.MAIN_ATTR] : k for k,v in self.dct_obj_metadata.items()}
    obj = dct_main_attr_to_obj.get(attribute, None)
    if obj is not None:
      self._update_object_last_seen(obj)
      return True

    return False

  # def _get_top_k_similar_attributes(self, attribute, k=5):
  #   self.log.start_timer('bk_tree_find')
  #   found = self.bk_tree.find(attribute, n=self.max_distance)
  #   attrs = list(map(lambda x: x[1], found[:k]))
  #   self.log.end_timer('bk_tree_find')
  #   return attrs

  def _get_top_k_similar_attributes_v2(self, attribute, k=5):
    self.start_timer('_get_top_k_similar_attributes_v2')
    db_attrs = np.array(list(self.dct_attr_metadata.keys()))
    distances = []
    for db_attr in db_attrs:
      x = self._dct_hashed_attrs.get(attribute, attribute)
      y = self._dct_hashed_attrs.get(db_attr, db_attr)
      distances.append(self.func_distance(x, y))

    sorted_indices = np.argsort(np.array(distances))
    sorted_distances = np.array(distances)[sorted_indices]

    mask = sorted_distances <= self.max_distance
    good_indices = sorted_indices[:np.sum(mask)][:k]
    good_attrs = db_attrs[good_indices]
    self.end_timer('_get_top_k_similar_attributes_v2')
    return good_attrs

  def _add_new_object_id(self):
    index = self._nr_objects
    self._nr_objects += 1
    return index

  def get_tracker_attribute(self, attribute):
    self.start_timer('get_tracker_attribute')
    res = None
    if attribute in self.dct_attr_metadata.keys():
      obj = self.dct_attr_metadata[attribute][UTC.ASSOCIATED_OBJECT]
      res = self.dct_obj_metadata[obj][UTC.MAIN_ATTR]
    else:
      top_k_similar_attributes = self._get_top_k_similar_attributes_v2(attribute, k=5)
      if len(top_k_similar_attributes) > 0:
        possible_objects = [self.dct_attr_metadata[x][UTC.ASSOCIATED_OBJECT] for x in top_k_similar_attributes]
        counter_possible_objects = collections.Counter(possible_objects)
        obj = counter_possible_objects.most_common(1)[0][0]
        res = self.dct_obj_metadata[obj][UTC.MAIN_ATTR]
      # endif
    # endif
    if res is None:
      res = attribute
    self.end_timer('get_tracker_attribute')
    return res


  def _attach_attribute_to_object(self, attribute):
    if attribute in self.dct_attr_metadata:
      return self.dct_attr_metadata[attribute][UTC.ASSOCIATED_OBJECT]

    self.start_timer('_attach_attribute_to_object')
    top_k_similar_attributes = self._get_top_k_similar_attributes_v2(attribute, k=5)
    self.dct_attr_metadata[attribute] = self._initial_attr_metadata()

    if len(top_k_similar_attributes) == 0:
      obj = self._add_new_object_id()
      self.dct_obj_metadata[obj] = self._initial_obj_metadata()
    else:
      possible_objects = [self.dct_attr_metadata[x][UTC.ASSOCIATED_OBJECT] for x in top_k_similar_attributes]
      counter_possible_objects = collections.Counter(possible_objects)
      obj = counter_possible_objects.most_common(1)[0][0]
    #endif

    self.dct_obj_metadata[obj][UTC.ALL_ATTR].add(attribute)
    self.dct_attr_metadata[attribute][UTC.ASSOCIATED_OBJECT] = obj
    self.end_timer('_attach_attribute_to_object')
    return obj

  def _maybe_adjust_object_main_attribute(self, obj):
    self.start_timer('_maybe_adjust_object_main_attribute')
    def argmax(iterable):
      return max(enumerate(iterable), key=lambda x: x[1][1])[0]

    crt_main_attr = self.dct_obj_metadata[obj][UTC.MAIN_ATTR]
    crt_main_attr_count = None
    if crt_main_attr is not None:
      crt_main_attr_count = self.dct_attr_metadata[crt_main_attr][UTC.COUNT]

    set_all_attr = self.dct_obj_metadata[obj][UTC.ALL_ATTR]
    counts = [(attr, self.dct_attr_metadata[attr][UTC.COUNT]) for attr in set_all_attr]

    max_attr, max_count = counts[argmax(counts)]
    self.end_timer('_maybe_adjust_object_main_attribute')

    if max_count == crt_main_attr_count:
      return crt_main_attr
    else:
      self.dct_obj_metadata[obj][UTC.MAIN_ATTR] = max_attr
      return max_attr

  def _maybe_flush_unseen_objects(self):
    delta_seconds = (dt.now() - self._last_flush_time).seconds
    if delta_seconds >= self.flush_delta_seconds:
      nr_flushed = self._flush_unseen_objects()
      self._last_flush_time = dt.now()
      self.P("Flushed {} objects".format(nr_flushed))
    return

  def _flush_unseen_objects(self):
    _LOCAL_DEBUG = False

    keys = deepcopy(list(self.dct_obj_metadata.keys()))
    nr_flushed = 0
    for obj in keys:
      last_seen = self.dct_obj_metadata[obj][UTC.LAST_SEEN]
      set_all_attr = self.dct_obj_metadata[obj][UTC.ALL_ATTR]
      if (dt.now() - last_seen).seconds >= self.flush_delta_seconds:
        nr_flushed += 1
        if _LOCAL_DEBUG:
          self.P("Flush obj {} with attrs: {}".format(obj, set_all_attr))
        for attr in set_all_attr:
          self.dct_attr_metadata.pop(attr)
        self.dct_obj_metadata.pop(obj)
    #endfor
    return nr_flushed


if __name__ == '__main__':
  from naeural_core import Logger
  from naeural_core.utils.pybktree import levenshtein_distance
  from scipy.spatial.distance import cosine
  log = Logger(lib_name='LPRT', base_folder='.', app_folder='_local_cache')

  if True:
    cars_tracker = UniversalTracker(
      log=log,
      func_distance=levenshtein_distance,
      max_distance=2,
      flush_delta_seconds=3
    )

    infered_lp = [
      'B45ESO',
      'B45ESOO',
      'B45ESOO',
      'B45ESO',
      'B45ESO',
      'CT67AMI',
      'CT67AMY',
      'CT6AMY',
      'C767AMY',
      'DJ30XBC',
      'DJ30XBCC',
      'DJ30XBC',
      'DJ30XBC',
      'B44MAE',
      'B44ADP',
      'CT18AAA',
      'MD9012e4'
    ]

    for lp in infered_lp:
      vcar, adjusted_lp = cars_tracker.adjust_attribute(lp)
      log.P("For lp {:>10} found {:>10} (car {})".format(lp, adjusted_lp, vcar))
    #endfor
  #endif

  if True:
    people_tracker = UniversalTracker(
      log=log,
      func_distance=cosine,
      max_distance=0.025,
      flush_delta_seconds=3
    )

    infered_face_emb = np.array([
      [0.10, 0.20, 0.30, 0.40],
      [0.11, 0.20, 0.30, 0.40],
      [0.11, 0.23, 0.31, 0.45],
      [0.09, 0.20, 0.20, 0.39],

      [0.50, 0.60, 0.70, 0.80],
      [0.55, 0.61, 0.72, 0.79]
    ])

    for face_emb in infered_face_emb:
      person, adjusted_face_emb = people_tracker.adjust_attribute(face_emb)
      log.P("For face emb {} found {} (person {})".format(face_emb, adjusted_face_emb, person))
    #endfor
  #endif

