
import uuid


from datetime import datetime, timedelta, timezone
from collections import defaultdict
from copy import deepcopy
from threading import Lock


from core import constants as ct
from core.utils import Singleton


EPOCH_MANAGER_VERSION = '0.1.0'


FN_NAME = 'epochs_status.pkl'
FN_SUBFOLDER = 'network_monitor'
FN_FULL = FN_SUBFOLDER + '/' + FN_NAME

EPOCHMON_MUTEX = 'epochmon_mutex'

GENESYS_EPOCH_DATE = '2024-03-10 00:00:00'

NODE_ALERT_INTERVAL = 3600

class EPCT:
  NAME = 'name'
  ID = 'id'
  EPOCHS = 'epochs'
  ALERTS = 'alerts'
  LAST_ALERT_TS = 'last_alert_ts'
  CURRENT_EPOCH = 'current_epoch'
  HB_TIMESTAMPS = 'hb_dates'
  HB_COUNT = 'hb_count'

_NODE_TEMPLATE = {
  EPCT.NAME           : None,
  EPCT.EPOCHS         : defaultdict(int),
  EPCT.ALERTS         : 0,
  EPCT.LAST_ALERT_TS  : 0,
  
  EPCT.CURRENT_EPOCH  : {
    EPCT.ID               : None,
    EPCT.HB_TIMESTAMPS   : set(),
  },
}

def _get_node_template(name):
  data = deepcopy(_NODE_TEMPLATE)
  data[EPCT.NAME] = name
  return data

class EpochsManager(Singleton):
  
  def build(self, owner, debug_date=None, debug=False):
    self.__genesis_date = self.log.str_to_date(GENESYS_EPOCH_DATE).replace(tzinfo=timezone.utc)
    self.owner = owner
    self.__current_epoch = None
    self.__data = {}
    self.__debug = debug
    self._set_dbg_date(debug_date)
    return

  @property
  def data(self):
    return self.__data
  
  @property
  def genesis_date(self):
    return self.__genesis_date
  
  
  def _set_dbg_date(self, debug_date):
    if debug_date is not None:
      if isinstance(debug_date, str):
        debug_date = self.log.str_to_date(debug_date).replace(tzinfo=timezone.utc)
    self._debug_date = debug_date
    return
  
  
  def P(self, msg, **kwargs):
    self.log.P(msg, **kwargs)
    return
  
  
  
  def start_timer(self, name):
    self.log.start_timer(name, section='epoch')
    return
  
  def stop_timer(self, name):
    self.log.stop_timer(name, section='epoch')
    return
  
  def get_node_name(self, node_addr):
    """ 
    Given a node address, returns the name of the node.
    """
    return self.owner.network_node_eeid(node_addr)
  
  def __get_max_hb_per_epoch(self):
    max_hb = 0
    eeid = self.owner.node_name
    interval = self.owner.network_node_hb_interval(eeid=eeid)
    if interval is None:
      raise ValueError("Heartbeat interval not found for node: {}".format(eeid))
    nr_hb = 24 * 3600 // interval
    return nr_hb

      
  def _save_status(self):
    self.P("Saving epochs status...")
    
    # start critical section
    self.log.lock_resource(EPOCHMON_MUTEX)  
    _data_copy = deepcopy(self.__data)
    self.log.unlock_resource(EPOCHMON_MUTEX)
    # end critical section
    
    self.log.save_pickle_to_data(
      data=_data_copy, 
      fn=FN_NAME,
      subfolder_path=FN_SUBFOLDER,
    )
    return
  
  
  def _load_status(self):
    exists = self.log.get_data_file(FN_FULL) is not None
    if exists:
      self.P("Previous epochs state found. Loading epochs status...")
      epochs_status = self.log.load_pickle_from_data(
        fn=FN_NAME,
        subfolder_path=FN_SUBFOLDER
      )
      if epochs_status is not None:
        self.__data = epochs_status
        self.P("Epochs status loaded with ....", boxed=True)
      else:
        self.P("Error loading epochs status.", color='r')
      return  
    
  def get_epoch_id(self, date : any):
    """
    Given a date as string or datetime, returns the epoch id - ie the number of days since the genesis epoch.

    Parameters
    ----------
    date : str or date
      The date as string that will be converted to epoch id.
    """
    if isinstance(date, str):
      # remove milliseconds from string
      date = date.split('.')[0]
      date = self.log.str_to_date(date)
      date = date.replace(tzinfo=timezone.utc) 
    elapsed = (date - self.__genesis_date).days
    return elapsed
  
  def epoch_to_date(self, epoch_id=None):
    """
    Given an epoch id, returns the date as string.

    Parameters
    ----------
    epoch_id : int
      the epoch id
    """
    if epoch_id is None:
      epoch_id = self.get_time_epoch()
    date = self.__genesis_date + timedelta(days=epoch_id)
    str_date = datetime.strftime(date, format="%Y-%m-%d")
    return str_date
  
  def date_to_str(self, date):
    """
    Converts a date to string.
    """
    return datetime.strftime(date, format=ct.HB.TIMESTAMP_FORMAT_SHORT)
    
  
  def get_current_date(self):
    if self._debug_date is not None:
      return self._debug_date
    else:
      return datetime.now(timezone.utc)
        
  def get_time_epoch(self):
    """
    Returns the current epoch id.
    """
    return self.get_epoch_id(self.get_current_date())
  
  def get_hb_utc(self, hb):
    """
    Generates a datetime object from a heartbeat and returns the UTC datetime.

    Parameters
    ----------
    hb : dict
      the hb object

    Returns
    -------
    datetime.datetime
    """
    ts = hb[ct.PAYLOAD_DATA.EE_TIMESTAMP]
    tz = hb.get(ct.PAYLOAD_DATA.EE_TIMEZONE, "UTC+0")        
    remote_datetime = datetime.strptime(ts, ct.HB.TIMESTAMP_FORMAT)
    offset_hours = int(tz.replace("UTC", ""))
    utc_datetime = remote_datetime - timedelta(hours=offset_hours)
    return utc_datetime.replace(tzinfo=timezone.utc)
  
  
  
  def __reset_timestamps(self, node_addr):
    """
    Resets the current epoch timestamps for a node.

    Parameters
    ----------
    node_addr : str
      The node address.
    """
    self.__data[node_addr][EPCT.CURRENT_EPOCH][EPCT.HB_TIMESTAMPS] = set()
    self.__data[node_addr][EPCT.CURRENT_EPOCH][EPCT.ID] = self.get_time_epoch()
    return


  def __reset_all_timestamps(self):
    for node_addr in self.__data:
      self.__reset_timestamps(node_addr)
    return
  
  
  def __calculate_avail_seconds(self, timestamps, time_between_heartbeats=10):
    """
    This method calculates the availability of a node in the current epoch based on the timestamps.

    Parameters
    ----------
    timestamps : set
      The set of timestamps for the current epoch.

    time_between_heartbeats: int
      Mandatory time between heartbeats in seconds.
    
    Returns
    -------
    float
      The availability percentage.
    """
    avail_seconds = 0
    nr_timestamps = len(timestamps)
    
    if nr_timestamps > 1:      
      for i in range(1, nr_timestamps):
        delta = (timestamps[i] - timestamps[i - 1]).seconds
        if delta <= (time_between_heartbeats + 5) and delta > (time_between_heartbeats / 2):
          # the delta between timestamps is less than the max heartbeat interval
          # while being more than half the heartbeat interval (ignore same heartbeat)
          avail_seconds += time_between_heartbeats
        #endif delta between timestamps is less than the max heartbeat interval
      #endfor each hb timestamp
    #endif there are more than 1 timestamps
    return avail_seconds    


  def __recalculate_current_epoch_for_node(self, node_addr, time_between_heartbeats=10):
    """
    This method recalculates the current epoch availability for a node. 
    It should be used when the epoch changes just before resetting the timestamps.

    Parameters
    ----------
    node_addr : str
      The node address.
    """
    node_data = self.__data[node_addr]
    current_epoch_data = node_data[EPCT.CURRENT_EPOCH]
    timestamps = current_epoch_data[EPCT.HB_TIMESTAMPS]
    current_epoch = current_epoch_data[EPCT.ID]
    # now the actual calculation
    lst_timestamps = sorted(list(timestamps))
    avail_seconds = self.__calculate_avail_seconds(
      lst_timestamps, time_between_heartbeats=time_between_heartbeats
    )
    max_possible = 24 * 3600
    prc_available = round(avail_seconds / max_possible, 4)
    record_value = int(prc_available * 255)
    self.__data[node_addr][EPCT.EPOCHS][current_epoch] = record_value
    
    if self.__debug:
      try:
        node_name = self.__data[node_addr][EPCT.NAME]
        start_date, end_date = None, None
        if len(lst_timestamps) >= 1:
          start_date = self.date_to_str(lst_timestamps[0])
          end_date = self.date_to_str(lst_timestamps[-1])
        str_node_addr = node_addr[:8] + '...' + node_addr[-3:]
        self.P("{}:{} availability in epoch {} was: {} ({:.2f}%) from {} to {}".format(
          node_name, str_node_addr, current_epoch, 
          record_value, prc_available * 100, start_date, end_date
        ))
      except Exception as e:
        self.P("Error calculating availability for node: {}".format(node_addr), color='r')
        self.P(str(e), color='r')
    return prc_available
    
    
    
  def recalculate_current_epoch_for_all(self):
    self.log.lock_resource(EPOCHMON_MUTEX)
    self.P("Recalculating epoch {} availability for all nodes during epoch {}...".format(
      self.__current_epoch, self.get_time_epoch()
    ))
    self.start_timer('recalc_all_nodes_epoch')
    for node_addr in self.__data:
      self.start_timer('recalc_node_epoch')
      self.__recalculate_current_epoch_for_node(node_addr)
      self.stop_timer('recalc_node_epoch')
    self.stop_timer('recalc_all_nodes_epoch')
    self.log.unlock_resource(EPOCHMON_MUTEX)
    return


  def maybe_close_epoch(self):
    """
    This method checks if the current epoch has changed and if so, it closes the current epoch and 
    starts a new one. Closing the epoch implies recalculating the current epoch node availability 
    for all nodes and then resetting the timestamps.
    """
    result = False # assume no epoch change
    current_epoch = self.get_time_epoch()
    if self.__current_epoch is None:
      self.__current_epoch = current_epoch
      self.P("Starting epoch: {}".format(self.__current_epoch))
    elif current_epoch != self.__current_epoch:
      if current_epoch != (self.__current_epoch + 1):
        self.P("Epoch is not valid. Current epoch: {}, Last epoch: {}".format(current_epoch, self.__current_epoch))
      else:
        self.P("Closing epoch: {}".format(self.__current_epoch))
        self.recalculate_current_epoch_for_all()
        self.P("Starting epoch: {}".format(current_epoch))
        self.__current_epoch = current_epoch      
        self.__reset_all_timestamps()
        self._save_status()
        result = True
      #endif epoch is not the same as the current one
    #endif current epoch is not None
    return result
  

  def register_data(self, node_addr, hb):
    """
    This method registers a heartbeat for a node in the current epoch.
    
    Parameters
    ----------
    node_addr : str
      The node address.
      
    hb : dict
      The heartbeat dict.
      
    """
    local_epoch = self.get_time_epoch()   
    # maybe first epoch for node_addr
    if node_addr not in self.__data:
      node_name = self.get_node_name(node_addr)
      self.__data[node_addr] = _get_node_template(node_name)
      self.__reset_timestamps(node_addr)
    #endif node not in data
    dt_remote_utc = self.get_hb_utc(hb)
    # check if the hb epoch is the same as the current one
    remote_epoch = self.get_epoch_id(dt_remote_utc)     
    if remote_epoch == local_epoch:
      # the remote epoch is the same as the local epoch so we can register the heartbeat
      self.log.lock_resource(EPOCHMON_MUTEX)
      # add the heartbeat timestamp for the current epoch
      self.__data[node_addr][EPCT.CURRENT_EPOCH][EPCT.HB_TIMESTAMPS].add(dt_remote_utc)
      self.log.unlock_resource(EPOCHMON_MUTEX)
    else:
      self.P("Received invalid epoch {} from node {} on epoch {}".format(
        remote_epoch, node_addr, local_epoch
      ))
    #endif remote epoch is the same as the local epoch
    self.maybe_close_epoch()
    return
  
  
  def get_node_state(self, node_addr):
    """
    Returns the state of a node in the current epoch.

    Parameters
    ----------
    node_addr : str
      The node address.
    """
    if node_addr not in self.__data:
      return None
    return self.__data[node_addr]
  
  
  def get_node_epochs(self, node_addr, autocomplete=False):
    """
    Returns the epochs availability for a node.

    Parameters
    ----------
    node_addr : str
      The node address.
    """
    if node_addr not in self.__data:
      return None
    dct_state = self.get_node_state(node_addr)
    dct_epochs = dct_state[EPCT.EPOCHS]
    if autocomplete:
      for epoch in range(1, self.get_time_epoch()):
        if epoch not in dct_epochs:
          dct_epochs[epoch] = 0
    return dct_epochs
  
  
  def get_node_epoch(self, node_addr, epoch_id=None, as_percentage=False):
    """
    This method returns the percentage a node was alive in a given epoch.
    The data is returned from already calculated values.

    Parameters
    ----------
    node_addr : str
      The node address.
    epoch_id : int
      The epoch id. Defaults to the last epoch

    Returns
    -------
    float
      The value between 0 and 1 representing the percentage of the epoch the node was alive.
    """
    if node_addr not in self.__data:
      return 0
    if epoch_id is None:
      epoch_id = self.get_time_epoch() - 1
    if epoch_id < 1 or epoch_id >= self.get_time_epoch():
      raise ValueError("Invalid epoch requested: {}".format(epoch_id))
    # get the epochs data
    epochs = self.get_node_epochs(node_addr)
    if epochs is None:
      return 0    
    if as_percentage:
      return round(epochs[epoch_id] / 255, 4)
    return epochs[epoch_id]


  def get_node_previous_epoch(self, node_addr, as_percentage=False):
    """
    Returns the last epoch the node was alive.

    Parameters
    ----------
    node_addr : str
      The node address.
    """
    if node_addr not in self.__data:
      return 0
    last_epoch = self.get_time_epoch() - 1
    return self.get_node_epoch(node_addr, epoch_id=last_epoch, as_percentage=as_percentage)
  
  def get_node_last_epoch(self, node_addr, as_percentage=False):
    """
    Alias for get_node_previous_epoch.
    """
    return self.get_node_previous_epoch(node_addr, as_percentage=as_percentage)


  def get_node_first_epoch(self, node_addr):
    """
    Returns the first epoch the node was alive.

    Parameters
    ----------
    node_addr : str
      The node address.
    """
    if node_addr not in self.__data:
      return -1
    epochs = list(self.get_node_epochs(node_addr).keys())
    min_epoch = min(epochs)
    return min_epoch
    
    
    
  
  


if __name__ == '__main__':
  from core.core_logging import Logger
  from core.main.net_mon import NetworkMonitor
  
  FN_NETWORK = 'c:/Dropbox (Personal)/_DATA/netmon_db.pkl'
  
  l = Logger('EPOCH', base_folder='.', app_folder='_local_cache')
  
  DATES = [
    '2024-03-21 12:00:00',
    '2024-03-22 12:00:01',
    '2024-03-23 12:00:01',
    '2024-03-24 12:00:01',
    '2024-03-25 12:00:01',
  ]
  
  # make sure you have a recent (today) save network status
  eng = EpochsManager(log=l, owner=1234, debug_date=DATES[0], debug=True)
  eng2 = EpochsManager(log=l, owner=None, debug_date=DATES[1])
  assert id(eng) == id(eng2)
    
  
  if True:
    netmon = NetworkMonitor(
      log=l, node_name='aid_hpc', node_addr='0xai_A_VwF0hrQjqPXGbOVJqSDqvkwmVwWBBVQV3KXscvyXHC',
      epoch_manager=eng
    )
  else:
    netmon = NetworkMonitor(
      log=l, node_name='aid_hpc', node_addr='0xai_A_VwF0hrQjqPXGbOVJqSDqvkwmVwWBBVQV3KXscvyXHC'
    )
    
  eng.owner = netmon
  
  assert id(eng) == id(netmon.epoch_manager)  

  has_data = netmon.network_load_status(FN_NETWORK)
  
  if has_data:    
    l.P("Current time epoch is: {} ({})".format(eng.get_time_epoch(), eng.epoch_to_date()))
    
    
    nodes = {
      x: netmon.network_node_address(x) for x in netmon.all_nodes
    }
        
    dct_hb = {}
    
    # now check the nodes for some usable data
    current_epoch = eng.get_time_epoch()
    for node_name in nodes:
      hbs = netmon.get_box_heartbeats(node_name)
      idx = -1
      done = False
      good_hbs = defaultdict(list)
      for hb in hbs:
        ep = eng.get_epoch_id(hb[ct.PAYLOAD_DATA.EE_TIMESTAMP])
        if ep >= current_epoch:
          good_hbs[ep].append(hb)
      if len(good_hbs) > 0:
        dct_hb[node_name] = good_hbs
    
    l.P("Data available for epochs:\n{}".format(
      "\n".join(["{}: {}".format(x, list(dct_hb[x].keys())) for x in dct_hb]) 
    ))
    
    
    for step in range(5):
      current_date = DATES[step]
      eng._set_dbg_date(current_date)
      epoch = eng.get_epoch_id(current_date)
      l.P("Running step {} - epoch {} / {}".format(
        step, epoch, current_date
      ))
      if eng.maybe_close_epoch():
        for node_name in dct_hb:
          node_addr = eng.owner.network_node_address(node_name)
          l.P("Node {} (first ep: {}) @ epoch {} has avail: {} ({})".format(
            node_name, eng.get_node_first_epoch(node_addr), epoch - 1, 
            eng.get_node_previous_epoch(node_addr), eng.get_node_previous_epoch(node_addr, as_percentage=True)
          ))
      for node_name in dct_hb:
        node_addr = eng.owner.network_node_address(node_name)
        for hb in dct_hb[node_name][epoch]:
          eng.register_data(node_addr, hb)
    
    l.show_timers()