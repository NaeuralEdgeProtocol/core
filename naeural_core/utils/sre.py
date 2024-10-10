# -*- coding: utf-8 -*-
"""
Usage:
  
  Base objects:
    `state`: 
      current state of the observed equipment/entity
    
    `prev`: 
      previous state of the observed equipment/entity
    
    `hist`: 
      the queue of states (including current state) of the observed equipment/entity
    
    
  Properties for `state` and `prev`:
    
    `<obj>.<prop>.val`: 
      the <obj> state value for <prop>
        
    `<obj>.<prop>.changed`: 
      true if value was changed in <obj> state for <prop>

    `<obj>.<prop>.updated`: 
      true if value was update in <obj> state for <prop> (even if not changed)
    
    `<obj>.<prop>.time`: 
      timestamp of the last update for <prop> (including update with same value `0` to `0`)
      this field permits operations with `change_time` field such as 
      `(state.<prop>.time - prev.<prop>.change_time) > 10 and state.<prop>.changed`
      where the conditions is `True` if the last change of <prop> is more than 10 seconds ago
    
    `<obj>.<prop>.change_time`: 
      timestamp when <prop> last changed value
      
  
  Properties for `hist`:
    `hist(sec)`:
      this will return a object (key-value pair) for each of the properties based on the
      historical states in the past `sec` seconds
      
    `hist.<prop>.val`:
      returns a list of states (up to 10_000 states) that can be sliced with operations such
      as `hist.<prop>.val[-5:]` (last 5 states including current) or `hist.<prop>.val[-3]` 
      (pre-previous state) or `hist.<prop>.val[-1]` (current state)
      
      
    `hist.<prop>.past(sec)`:
      returns a series-like _Series object containing state values for <prop> for the past `sec` seconds
  
  
  Properties of `_Series` objects:
    
    `sum`: 
      returns the sum of non-null values
      
    `count`:
      returns the count of all non-null values
      
    `valcount(val)`
      function that returns count of all `val` values (non-null)
      
    `min`:
      returns the min of non-null values
      
    `max`:
      returns the max of non-null values
    
    `mean`:
      returns the mean of non null values
      
    `nullcount`:
      returns the number of all null values
      
    `empty`:
      True if there are no observations (including nulls)
      
  Examples:

    1. Check if a prop has been repeatedly updated with same value:
      
      `state.<prop>.updated and not stat.<prop>.changed`
    

    2. Direct history check - check if 3 activations in last 2 sec and current status is non zero
       due to equality this rule will not raise a second alert when another update is added 
       (eg. 4th activation in the interval). You can re-raise with `sum >= 3`:
      
      `hist.<prop>.past(sec=2).sum == 3 and state.f3.val == 1` 
    
    
    3. Example of 2 event hits in a row if initial change was at least 1.5 sec ago:
      
      `state.<prop>.val == 1 and prev.<prop>.val==1 and (state.<prop>.time - prev.<prop>.change_time) > 1.5`


"""

import numpy as np
import re
from copy import deepcopy

from collections import deque, OrderedDict
from time import time, strftime, localtime, strptime

from naeural_core import DecentrAIObject


MAX_DEQUE = 10_000
NAME = 'name'
ENV = 'env'
RULES = 'rules'
BASE_PROPS = [NAME]

DEBUG_DICT = {1:False}

__VER__ = '1.5.2'

class _Series:
  def __init__(self, vals):
    if not isinstance(vals, (list, deque)):
      raise ValueError('`vals` must be a list')
    self.np = np.array([x for x in vals if x is not None])
    self.vals = list(vals)
    return
  
  @property
  def sum(self):
    return self.np.sum()
  
  @property
  def min(self):
    return self.np.min()

  @property
  def max(self):
    return self.np.max()
  
  @property
  def mean(self):
    return self.np.mean()  

  @property
  def count(self):
    return self.np.ravel().shape[0]
  
  @property
  def empty(self):
    return len(self.vals) == 0
  
  @property
  def nullcount(self):
    return len([x for x in self.vals if x is None])
  
  def valcount(self, val):
    return (self.np == val).sum()
  
  def __repr__(self):
    return str(self.vals)
  

class _StateProperty(DecentrAIObject):
  """
   simple single-feature object that can track (if series) historical values
   important properties are:
     val: current step value 
     time: time of current step
     change_time: time when it was changed
     updated: was updated this current step
     changed: was changed this current step
  """
  def __init__(self, name,series=False, **kwargs):
    self._name = name

    self.is_series = series
    super().__init__(**kwargs)
    return
  
  @property
  def name(self):
    return str(self._name)
  
  @property
  def val(self):    
    return list(vars(self)[self._name]) if self.is_series else vars(self)[self._name]
  
  @property
  def array(self):
    return _Series(self.val)
  
  
  @property
  def time(self):
    return self._timestamp
  
  @time.setter
  def time(self, t):
    if isinstance(t, str):
      t = self.log.str_to_time(t)
    if isinstance(self._timestamp, (deque, list)):
      self._timestamp.append(t)
    else:
      self._timestamp = t
    return  
              
  @property
  def time_str(self):
    if isinstance(self._timestamp, (deque, list)):
      return [self.log.time_to_str(x) for x in self._timestamp]
    else:
      return self.log.time_to_str(self._timestamp)  
    
  @property
  def elapsed_from_update(self):
    # time from last update
    now_time = time()
    if isinstance(self.time, (deque, list)):
      return round(now_time - self.time[-1], 3)
    else:
      return round(now_time - self.time, 3)

  @property
  def elapsed_from_change(self):
    # time from last update
    now_time = time()
    if isinstance(self.time, (deque, list)):
      return round(now_time - self.time[-1], 3)
    else:
      return round(now_time - self.change_time, 3)
    
  
  def startup(self):
    self.change_time = None
    if self.is_series:
      vars(self)[self._name] = deque(maxlen=MAX_DEQUE)
      self._timestamp = deque(maxlen=MAX_DEQUE)
    else:
      vars(self)[self._name] = None
      self._timestamp = None
    self.changed = False # keeps track of actual change in value
    self.updated = False # keeps track of updates - even if it is the same value
    return 
    
  
  def update(self, val, timestamp):
    # update state or add to series
    if isinstance(vars(self)[self._name], deque):
      vars(self)[self._name].append(val)
      # nextupdate timestamp with current addition even if it None
    else:
      if val is None:
        self.changed = False
        self.updated = False
        return
      else:
        self.updated = True # update no matter what
        self.changed = False
        if vars(self)[self._name] != val:
          # if different the is changed
          self.changed = True
          self.change_time = timestamp
        # update variable
        vars(self)[self._name] = val
      # enf else we have data
    # end if hist or single
    self.time = timestamp # this is ok for hists
    return

  def past(self, sec, restrict_range=False):
    if not isinstance(self._timestamp, (deque, list)):
      raise ValueError('Tried to retrieve last period on non-hist state')
    np_ts = np.array(self._timestamp)
    time_range = round(time() - np_ts.min(),2)
    if sec >  time_range and restrict_range:
      raise ValueError('Tried to retrieve a last period of {}s beyond of hist {}s'.format(sec, time_range))
    max_time = time() - sec
    valid = np_ts >= max_time
    if valid.sum() > 0:
      first_time = np.argwhere(valid).ravel()[0]
      vals = self.val[first_time:]
    else:
      vals = []
    res = _Series(vals)
    return res
  
  
  def __repr__(self):
    if isinstance(self._timestamp, (deque, list)):
      return "<{}: val:{}, time:{}>".format(
        self.name, self.val, 
        [self.log.time_to_str(x) for x in self.time], 
      )
    else:
      return "<{}: val:{}, time:{}, updated: {}, changed:{}, change_time: {}>".format(
        self.name, self.val, 
        self.log.time_to_str(self.time) if self.time is not None else None, 
        self.updated, self.changed, 
        self.log.time_to_str(self.change_time)  if self.change_time is not None else None,
      )



class _EntityState(DecentrAIObject):
  """
  Defines a entity state based on multiple features
  EntityState is used as various state definitions in EntityEnv
  """
  def __init__(self, name, is_series, properties, **kwargs):
    assert (isinstance(properties, list) and 
            len(properties) > 0 and 
            isinstance(properties[0], str)
            ), "Properties must be defined as a list of strings"    
    vars(self)[NAME] = name   
    # better set properties explicitly so we can then do sanity check 
    # of inputs ad update time
    self.properties = list(OrderedDict({x:0 for x in properties}).keys())
    self.is_series = is_series
    super().__init__(**kwargs)
    return
  
  @property
  def changed(self):
    return [x for x in self.properties if vars(self)[x].changed]

  @property
  def time(self):
    return self._timestamp
  
  @property
  def time_str(self):
    return self.log.time_to_str(self._timestamp)
  
  
  @property
  def properties_dict(self):
    return {k:vars(self)[k].val for k in self.properties}
    
  def startup(self):
    for prop in self.properties:
      new_prop = _StateProperty(
        name=prop, 
        series=self.is_series,
        log=self.log
      )
      vars(self)[prop] = new_prop
    return
  
  
  def update(self, timestamp, **properties):
    self._timestamp = timestamp
    for prop in properties:
      if prop not in self.properties:
        raise ValueError("'{}'<{}> does not have property '{}' (props are {})".format(
          self.name, self.__class__.__name__, prop, self.properties))
    for prop in self.properties:
      val = properties.get(prop)
      vars(self)[prop].update(val=val, timestamp=timestamp)
    return
  
  
  def copy_from(self, state):
    if state.is_series:
      raise ValueError("Cannot copy from a hist state")
    for prop in state.properties:
      val = vars(state)[prop].val
      timestamp = vars(state)[prop].time
      vars(self)[prop].update(val=val, timestamp=timestamp)
    if hasattr(state, '_timestamp'):
      self._timestamp = state._timestamp
    return
  
  
  def __repr__(self):
    s = '<{}:'.format(self.name)
    for p in self.properties:
      v = getattr(self, p, "N/A")
      s += ' {}={}'.format(p, v.val)
    s += '>'
    return s
  
  def __call__(self, sec):
    if not self.is_series:
      raise ValueError('Cannot call `{}({})` on non-series object'.format(
        self.__class__.__name__, sec
        )
      )
    dct_res = {k:vars(self)[k].past(sec).vals for k in self.properties}   
    return dct_res
  
    

  
class EntityEnv(DecentrAIObject):
  """
  The entity states - current, past and history
  """
  def __init__(self, name, properties, **kwargs):
    self.version = __VER__
    self.name = name    
    self.properties = list(OrderedDict({x:0 for x in properties}).keys())
    super().__init__(**kwargs)
    return
  
  def startup(self):
    self.hist = _EntityState(
      name=self.name,
      is_series=True,
      properties=self.properties,
      log=self.log,
    )
    self.prev = _EntityState(
      name=self.name,
      is_series=False,
      properties=self.properties,
      log=self.log,
    )
    self.state = _EntityState(
      name=self.name,
      is_series=False,
      properties=self.properties,
      log=self.log,
    )
    return
  

  @staticmethod
  def get_ops(s, return_props=False):
    """
    Returns all the ops checking for rule validity. Can return all props instead of
    all ops
    """
    ops = [x for x in re.split('[*\-,>=<=><! \[\]]| in | and | or | not ', s) if len(x)>0 and x not in ['in', 'and', 'or', 'not']]
    props = set()
    for op in ops:
      op = op.replace('(','').replace(')','')
      opr = op.split(".")
      if len(opr) >= 3:
        if opr[0] not in ['state', 'prev', 'hist']:
          raise ValueError("Unknown op '{}' in '{}'".format(op, s))
        props.add(opr[1])
      elif len(opr) in [1,2]:
        if not opr[0].isnumeric():
          raise ValueError("Unknown value '{}' in '{}'".format(op, s))
      else:
        raise ValueError("Wrong op '{}' in '{}'".format(op, s))
    if return_props:
      return list(props)
    else:
      return ops  
    
  
  def step(self, timestamp, **properties):
    
    # copy current state to prev
    self.prev.copy_from(self.state)
    
    # add state to history
    self.hist.update(
      timestamp=timestamp,
      **properties,
    )
    
    # finally update actual current state
    self.state.update(
      timestamp=timestamp,
      **properties,
    )
    return
  
  def __repr__(self):
    s = ''
    mods = []
    for p in self.properties:
      v = getattr(self.state, p, None)
      s += '{}={} '.format(p, v.val)
      if v.changed:
        mods.append(p)
    s = '<{} S({}):['.format(self.name, ",".join(mods)) + s
    s = s[:-1]
    s += '] P:['
    for p in self.properties:
      v = getattr(self.prev, p,None)
      s += '{}={} '.format(p, v.val)
    s = s[:-1]
    s += ']>'
    return s
  
  def evaluate(self, str_eval):
    str_eval = str_eval.replace('\n', '')
    try:
      ops = self.get_ops(str_eval)
      state = self.state
      prev = self.prev
      hist = self.hist
      res = eval(str_eval)
    except Exception as e:
      res = "Eval of '{}' has failed: {}".format(str_eval, e)
      self.P(res, color='r')
    return res  
  
  def multiple_evaluate(self, lst_evals):
    res = []
    for str_eval in lst_evals:
      res.append(self.evaluate(
        str_eval=str_eval
        )
      )      
    return res
  
  
class SRE(DecentrAIObject):
  def __init__(self, **kwargs):
    self.version = __VER__    
    self.envs = {}
    super().__init__(**kwargs)
    return
  
  def _get_rules(self, entity_id):
    assert entity_id in self.envs, "No environment created for '{}'".format(entity_id)
    return self.envs[entity_id][RULES]
  
  def _get_env(self, entity_id):
    assert entity_id in self.envs, "No environment created for '{}'".format(entity_id)
    return self.envs[entity_id][ENV]
  
  @staticmethod
  def get_props_from_rule(str_rule):
    str_rule = str_rule.replace('\n', '')
    return EntityEnv.get_ops(s=str_rule, return_props=True)
  
  def add_entity(self, entity_id, entity_props, entity_rules=[]):
    """
    Adds a new entity environment with (optional) its own rules

    Parameters
    ----------
    entity_id : str
      The name/ID of the entity.
      
    entity_props : list[str]
      the properties such as 'f1', 'f2', etc.
      
    entity_rules : list[str], optional
      The list of rules - eg. "state.f1.val == 0 and state.f2.val == 0 and prev.f2.val==1". The default is [].

    Returns
    -------
    None.

    """
    env = EntityEnv(
      name=entity_id,
      properties=entity_props,
      log=self.log,
    )
    
    self.envs[entity_id] = {
      ENV : env,
      RULES : entity_rules,
    }
    return

  def get_envs(self):
    return list(self.envs.keys())
  
  
  def assign_rules(self, entity_id, entity_rules):
    """
    Assigns a set of rules (list) to a existing entity environment

    Parameters
    ----------
    entity_id : str
      the entity env name.
      
    entity_rules : list[str]
      the list of rules.

    Returns
    -------
    None.

    """
    assert entity_id in self.envs, "No environment created for '{}'".format(entity_id)
    self.envs[entity_id][RULES] = entity_rules
    return
  
  def step(self, entity_id, run_rules=True, **props):
    """
    Add a data step to a environment and run rules by default

    Parameters
    ----------
    entity_id : str
      name of the entity.
      
    run_rules: bool
      Run the existing rules (if any). Default `True`
      
    **props : dict of prop=val
      all the current properties - all missing will be None by default.

    Returns
    -------
    Bool: evaluation result

    """
    assert entity_id in self.envs, "No environment created for '{}'".format(entity_id)
    t = props.pop('timestamp') if 'timestamp' in props else time()
    env = self._get_env(entity_id)
    env.step(
      timestamp=t,
      **props,
    ) 
    result = None
    rules = self._get_rules(entity_id)
    if len(rules) > 0 and run_rules:
      result = env.multiple_evaluate(rules)
    return result
  
  def multi_step(self, dct_data_by_entity, run_rules=True):
    """
    Similar with `step` but applies to all entity envs

    Parameters
    ----------

    run_rules: bool
      Run the existing rules (if any). Default `True`

    dct_data_by_entity : dict of entity_id:dct_props
      Input dict of dicts.

    Returns
    -------
    Bool: evaluation result by env

    """
    # maybe multi-threaded env 
    dct_data = deepcopy(dct_data_by_entity)
    dct_result = {}
    for entity_id, dct_entity_data in dct_data.items():
      dct_result[entity_id] = self.step(
        entity_id=entity_id,
        run_rules=run_rules,
        **dct_entity_data
      )
    return dct_result
  
  
   
  
if __name__ == '__main__':
  ## DEMO SCRIPT ###
  
  from time import sleep
  from naeural_core import Logger
  
  l = Logger('TST', base_folder='.', app_folder='_local_cache')
  
  SIMPLE_TEST = True
  props_1 = ['f1','f2','f3']
  props_2 = ['f1','f2','f3']
  vals_1 = [
    [0,None,1],
    [None,1,0],
    [0,0,None],
    [None,None,1],
    [1,None,None],
    [None,None,0],
    [0,1,0],
    [None,None,1],
    [None,None,1],
    [None,None,1],
    [None,None,1],
  ]
   
  vals_2 = [
    [None,1,0],
    [None,0,None],
    [None,None,1],
    [0,None,1],
    [1,None,None],
    [None,None,0],
    [0,1,0],
    [None,None,1],
    [None,None,1],
    [None,None,1],
    [None,None,1],
    [1,None,None],
  ]
  
  vals_2 = vals_2[:len(vals_1)]  
  
  if SIMPLE_TEST:
    evals = [
      """ 
      state.f1.val == 0 and
      prev.f1.val == 0 and
      state.f1.updated and
      state.f2.val == 0 and
      prev.f2.val == 1 and
      prev.f3.val == 0 and
      prev.f3.changed
      """,
      "o prostie"
      ]
    vals_1 = vals_1[:4]
  else:
    evals = [
      # simple
      'state.f1.val == 0 and state.f2.val == 0 and prev.f2.val==1', 
      
      
      # simple with expiration timer
      """
         state.f2.val == 1 and prev.f2.val==0 and 
         (state.f2.time - prev.f2.time) > 1 and 
         (state.f2.elapsed_from_change < 2)
      """, 
      
      # 3 hits in a row in last 3 updates
      'state.f3.val == 1 and hist.f3.val[-2] == 1 and hist.f3.val[-3] == 1', 
      
      # 2 hits in a row if initial change was at least 1.5 sec ago
      'state.f3.val == 1 and prev.f3.val==1 and (state.f3.time - state.f3.change_time) > 1.5', 
      
      
      # direct history check - check if 3 activations in last 2 sec and current status is non zero
      # due to equality this rule will not raise a second alert when another update is added 
      # (eg. 4th activation in the interval). You can re-raise with `sum >= 3`
      'hist.f3.past(2).sum == 3 and state.f3.val != 0', 
          
    ]
    
 
  
  # create engine
  eng = SRE(log=l)
  
  l.p(eng.get_props_from_rule(evals[0]))
  
  # add a couple of devices
  eng.add_entity(
    entity_id='dev_test_1', 
    entity_props=props_1,
    entity_rules=evals,
  )
  if not SIMPLE_TEST:
    eng.add_entity(
      entity_id='dev_test_k', 
      entity_props=props_2,
      entity_rules=evals,
    )
  
  for i in range(len(vals_1)):
    if i == len(vals_1) - 1:
      DEBUG_DICT[1] = True
    v1 = vals_1[i]
    v2 = vals_2[i]
    
    data_1 = dict(zip(props_1,v1))
    data_1['timestamp'] = time()
    data_2 = dict(zip(props_2,v2))
    data_2['timestamp'] = time()
    
    data = {
      'dev_test_1' : data_1,
    }
    if not SIMPLE_TEST:
      data['dev_test_k'] = data_2
    
    results = eng.multi_step(
      dct_data_by_entity=data,
    )  
    
    
    for dev in results:
      dev_res = results[dev]
      for res, cnd in zip(dev_res, evals):
        l.P("Status:\nDevice:   {}\nInput({}): {}\nRule:     {}\nResult:   {}".format(
          dev,i,data[dev],
          cnd.replace('\n',' ').replace('  ','').strip(),
          res, 
          ), 
          color='y' if res else None
        )
      l.P("*"*80)
    l.P("*"*80)
    sleep(1)
  
  env = eng.envs['dev_test_1']['env']
  sleep(4)
  hist_1s = env.hist(sec=7)  
  l.P(hist_1s)
  for p in props_1:
    l.P(vars(env.hist)[p].past(7))
  sleep(2)
  hf1 = env.hist.f1.past(7)
  l.P(hf1)
  
  
      
  def tt():
    _res = eng.multi_step(
      dct_data_by_entity=data,
    )  
    
    