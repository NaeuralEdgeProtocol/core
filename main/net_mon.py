"""


"""
import json
import os
import numpy as np

from time import time, sleep
from copy import deepcopy
from collections import deque, OrderedDict
from datetime import datetime as dt
from core import DecentrAIObject
from core import constants as ct

from .epochs_manager import EpochsManager

def exponential_score(left, right, val, right_is_better=False, normed=False):
  num = 50
  interval = np.linspace(left, right, num=num)
  scores = np.linspace(100, 1, num=num) # TODO try with np.geomspace
  if right_is_better:
    rng = range(len(interval)-1, -1, -1)
    sgn = '>='
  else:
    rng = range(len(interval))
    sgn = '<='

  for s,i in enumerate(rng):
    str_eval = '{}{}{}'.format(val, sgn, interval[i])
    res = eval(str_eval)
    if res:
      if normed:
        return scores[s] / 100.1
      else:
        return scores[s]

  return 0
#enddef


NETMON_MUTEX = 'NETMON_MUTEX'

NETMON_DB = 'db.pkl'
NETMON_DB_SUBFOLDER = 'network_monitor'

class NetworkMonitor(DecentrAIObject):
  
  HB_HISTORY = 30 * 60 // 10 # 30 minutes of history with 10 seconds intervals
  


  def __init__(self, log, node_name, node_addr, epoch_manager=None, **kwargs):
    self.node_name = node_name
    self.node_addr = node_addr
    self.__network_heartbeats = {}
    self.network_hashinfo = {}
    self.__epoch_manager = epoch_manager
    super(NetworkMonitor, self).__init__(log=log, prefix_log='[NMON]', **kwargs)    
    return


  @property
  def all_heartbeats(self):
    result = self.__network_heartbeats
    return result


  @property
  def all_nodes(self):
    return list(self.__network_heartbeats.keys())


  @property
  def epoch_manager(self):
    return self.__epoch_manager


  def startup(self):
    if self.__epoch_manager is None:
      self.__epoch_manager = EpochsManager(log=self.log, owner=self)
    return

  
  def _set_network_heartbeats(self, network_heartbeats):
    if isinstance(network_heartbeats, dict):
      for eeid in network_heartbeats:
        # make sure that the heartbeats are deques
        network_heartbeats[eeid] = deque(network_heartbeats[eeid], maxlen=self.HB_HISTORY)
      self.__network_heartbeats = network_heartbeats
    else:
      self.P("Error setting network heartbeats. Invalid type: {}".format(type(network_heartbeats)), color='r')
    return 
  
  
  def __register_heartbeat(self, eeid, data):
    if eeid not in self.__network_heartbeats:
      self.P("Box {} is alive in the network".format(eeid), color='y')
      self.__network_heartbeats[eeid] = deque(maxlen=self.HB_HISTORY)
    #endif
    if ct.HB.ENCODED_DATA in data:
      str_data = data.pop(ct.HB.ENCODED_DATA)
      dct_hb = json.loads(self.log.decompress_text(str_data))
      data = {
        **data,
        **dct_hb,
      }
    #endif encoded data
    # begin mutexed section
    self.log.lock_resource(NETMON_MUTEX)
    self.__network_heartbeats[eeid].append(data)
    self.log.unlock_resource(NETMON_MUTEX)
    # end mutexed section
    return
    
  
  def get_box_heartbeats(self, eeid):
    box_heartbeats = deque(self.all_heartbeats[eeid], maxlen=self.HB_HISTORY)
    return box_heartbeats


  # Helper protected methods section
  if True:
    def __network_nodes_list(self):
      if self.all_heartbeats is None:
        return []
      return list(self.all_heartbeats.keys())

    def __network_node_past_hearbeats_by_number(self, eeid, nr=1, reverse_order=True):
      if eeid not in self.__network_nodes_list():
        self.P("`_network_node_past_hearbeats_by_number`: EE_ID '{}' not available".format(eeid))
        return
      
      box_heartbeats = self.get_box_heartbeats(eeid)
      if reverse_order:
        lst_heartbeats = list(reversed(box_heartbeats))[:nr]
      else:
        lst_heartbeats = box_heartbeats[-nr:]
      return lst_heartbeats

    def __network_node_past_heartbeats_by_interval(self, eeid, minutes=60, dt_now=None, reverse_order=True):
      if eeid not in self.__network_nodes_list():
        self.P("`_network_node_past_heartbeats_by_interval`: EE_ID '{}' not available".format(eeid))
        return
      
      if dt_now is None:
        dt_now = dt.now()
        
      lst_heartbeats = []
      box_heartbeats = self.get_box_heartbeats(eeid)
      for heartbeat in reversed(box_heartbeats):
        remote_time = heartbeat[ct.HB.CURRENT_TIME]
        remote_tz = heartbeat.get(ct.PAYLOAD_DATA.EE_TIMEZONE)
        ts = self.log.utc_to_local(remote_time, remote_utc=remote_tz, fmt=ct.HB.TIMESTAMP_FORMAT)
        passed_minutes = (dt_now - ts).total_seconds() / 60.0
        if passed_minutes < 0 or passed_minutes > minutes:
          break
        lst_heartbeats.append(heartbeat)
      #endfor
      if not reverse_order:
        lst_heartbeats = list(reversed(lst_heartbeats))
      return lst_heartbeats

    def __network_node_last_heartbeat(self, eeid, return_empty_dict=False):
      if eeid not in self.__network_nodes_list():
        msg = "`_network_node_last_heartbeat`: EE_ID '{}' not available".format(eeid)
        if not return_empty_dict:
          raise ValueError(msg)
        else:
          self.P(msg, color='r')
          return {}
        #endif raise or return
      return self.all_heartbeats[eeid][-1]

    def __network_node_last_valid_heartbeat(self, eeid, minutes=3):
      past_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes, )
      if len(past_heartbeats) == 0:
        return

      return past_heartbeats[0]


  # "MACHINE_MEMORY" section (protected methods)
  if True:
    def __network_node_machine_memory(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.MACHINE_MEMORY]
  #endif


  # "AVAILABLE_MEMORY" section (protected methods)
  if True:
    def __network_node_past_available_memory_by_number(self, eeid, nr=1, norm=True):
      machine_mem = self.__network_node_machine_memory(eeid=eeid)
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.AVAILABLE_MEMORY] / machine_mem if norm else h[ct.HB.AVAILABLE_MEMORY] for h in lst_heartbeats]
      return lst

    def __network_node_past_available_memory_by_interval(self, eeid, minutes=60, norm=True, reverse_order=True):
      machine_mem = self.__network_node_machine_memory(eeid=eeid)
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes, reverse_order=reverse_order)
      lst = [h[ct.HB.AVAILABLE_MEMORY] / machine_mem if norm else h[ct.HB.AVAILABLE_MEMORY] for h in lst_heartbeats]
      return lst

    def __network_node_last_available_memory(self, eeid, norm=True):
      machine_mem = self.__network_node_machine_memory(eeid=eeid)
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.AVAILABLE_MEMORY] / machine_mem if norm else hearbeat[ct.HB.AVAILABLE_MEMORY]
  #endif

  # "PROCESS_MEMORY" section (protected methods)
  if True:
    def __network_node_past_process_memory_by_number(self, eeid, nr=1, norm=True):
      machine_mem = self.__network_node_machine_memory(eeid=eeid)
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.PROCESS_MEMORY] / machine_mem if norm else h[ct.HB.PROCESS_MEMORY] for h in lst_heartbeats]
      return lst

    def __network_node_past_process_memory_by_interval(self, eeid, minutes=60, norm=True):
      machine_mem = self.__network_node_machine_memory(eeid=eeid)
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes)
      lst = [100*h[ct.HB.PROCESS_MEMORY] / machine_mem if norm else h[ct.HB.PROCESS_MEMORY] for h in lst_heartbeats]
      return lst

    def __network_node_last_process_memory(self, eeid, norm=True):
      machine_mem = self.__network_node_machine_memory(eeid=eeid)
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.PROCESS_MEMORY] / machine_mem if norm else hearbeat[ct.HB.PROCESS_MEMORY]
  #endif

  # "CPU_USED" section (protected methods)
  if True:
    def __network_node_past_cpu_used_by_number(self, eeid, nr=1):
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.CPU_USED] for h in lst_heartbeats]
      return lst
    
    def __get_timestamps(self, lst_heartbeats):
      timestamps = [h[ct.HB.CURRENT_TIME].split('.')[0] for h in lst_heartbeats]
      return timestamps

    def __network_node_past_cpu_used_by_interval(
      self, eeid, minutes=60, dt_now=None, 
      return_timestamps=False, reverse_order=True
    ):
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(
        eeid=eeid, minutes=minutes, dt_now=dt_now, reverse_order=reverse_order
      )
      lst = [h[ct.HB.CPU_USED] for h in lst_heartbeats]
      if return_timestamps:
        timestamps = self.__get_timestamps(lst_heartbeats)
        return lst, timestamps
      else:
        return lst

    def __network_node_last_cpu_used(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.CPU_USED]
  #endif

  # "GPUS" section (protected methods)
  if True:
    def __network_node_past_gpus_by_number(self, eeid, nr=1):
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.GPUS] for h in lst_heartbeats]
      for i in range(len(lst)):
        if isinstance(lst[i], str):
          lst[i] = {}
      return lst

    def __network_node_past_gpus_by_interval(
      self, eeid, minutes=60, dt_now=None, 
      return_timestamps=False, reverse_order=True
    ):
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(
        eeid=eeid, minutes=minutes, dt_now=dt_now, reverse_order=reverse_order,
      )
      lst = [h[ct.HB.GPUS] for h in lst_heartbeats]
      for i in range(len(lst)):
        if isinstance(lst[i], str):
          lst[i] = {}
      if return_timestamps:
        timestamps = self.__get_timestamps(lst_heartbeats)
        return lst, timestamps
      else:
        return lst

    def __network_node_last_gpus(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      gpus = hearbeat[ct.HB.GPUS]
      if isinstance(gpus, str):
        gpus = [{}]
      return gpus
  #endif

  # "UPTIME" section (protected methods)
  if True:
    def __network_node_uptime(self, eeid, as_minutes=True):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      result = hearbeat[ct.HB.UPTIME]
      if as_minutes:
        result = result / 60
      return result
  #endif

  # "DEVICE_STATUS" section (protected methods)
  if True:
    def __network_node_past_device_status_by_number(self, eeid, nr=1):
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.DEVICE_STATUS] for h in lst_heartbeats]
      return lst

    def __network_node_past_device_status_by_interval(self, eeid, minutes=60):
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes)
      lst = [h[ct.HB.DEVICE_STATUS] for h in lst_heartbeats]
      return lst

    def __network_node_last_device_status(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.DEVICE_STATUS]
  #endif

  # "ACTIVE_PLUGINS" section (protected methods)
  if True:
    def __network_node_past_active_plugins_by_number(self, eeid, nr=1):
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.ACTIVE_PLUGINS] for h in lst_heartbeats]
      return lst

    def __network_node_past_active_plugins_by_interval(self, eeid, minutes=60):
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes)
      lst = [h[ct.HB.ACTIVE_PLUGINS] for h in lst_heartbeats]
      return lst

    def __network_node_last_active_plugins(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.ACTIVE_PLUGINS]
  #endif

  # "TOTAL_DISK" section (protected methods)
  if True:
    def __network_node_total_disk(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.TOTAL_DISK]
  #endif

  # "AVAILABLE_DISK" section (protected methods)
  if True:
    def __network_node_past_available_disk_by_number(self, eeid, nr=1, norm=True):
      total_disk = self.__network_node_total_disk(eeid=eeid)
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.AVAILABLE_DISK] / total_disk if norm else h[ct.HB.AVAILABLE_DISK] for h in lst_heartbeats]
      return lst

    def __network_node_past_available_disk_by_interval(self, eeid, minutes=60, norm=True):
      total_disk = self.__network_node_total_disk(eeid=eeid)
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes)
      lst = [h[ct.HB.AVAILABLE_DISK] / total_disk if norm else h[ct.HB.AVAILABLE_DISK] for h in lst_heartbeats]
      return lst

    def __network_node_last_available_disk(self, eeid, norm=True):
      total_disk = self.__network_node_total_disk(eeid=eeid)
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.AVAILABLE_DISK] / total_disk if norm else hearbeat[ct.HB.AVAILABLE_DISK]
  #endif

  # "SERVING_PIDS" section (protected methods)
  if True:
    def __network_node_past_serving_pids_by_number(self, eeid, nr=1):
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.SERVING_PIDS] for h in lst_heartbeats]
      return lst

    def __network_node_past_serving_pids_by_interval(self, eeid, minutes=60):
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes)
      lst = [h[ct.HB.SERVING_PIDS] for h in lst_heartbeats]
      return lst

    def __network_node_last_serving_pids(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.SERVING_PIDS]
  #endif

  # "LOOPS_TIMINGS" section (protected methods)
  if True:
    def __network_node_past_loops_timings_by_number(self, eeid, nr=1):
      lst_heartbeats = self.__network_node_past_hearbeats_by_number(eeid=eeid, nr=nr)
      lst = [h[ct.HB.LOOPS_TIMINGS] for h in lst_heartbeats]
      return lst

    def __network_node_past_loops_timings_by_interval(self, eeid, minutes=60):
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(eeid=eeid, minutes=minutes)
      lst = [h[ct.HB.LOOPS_TIMINGS] for h in lst_heartbeats]
      return lst

    def __network_node_last_loops_timings(self, eeid):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      return hearbeat[ct.HB.LOOPS_TIMINGS]
  #endif

  # "DEFAULT_CUDA" section (protected methods)
  if True:
    def __network_node_default_cuda(self, eeid, as_int=True):
      hearbeat = self.__network_node_last_heartbeat(eeid=eeid)
      default_cuda = hearbeat[ct.HB.DEFAULT_CUDA]
      if as_int:
        if ':' not in default_cuda:
          return
        default_cuda = int(default_cuda.split(':')[1])

      return default_cuda
  #endif
  
    

  # PUBLIC METHODS SECTION
  if True:    
    
    def network_node_last_comm_info(self, eeid):
      """
        "COMMS" : {
          "IN_KB": value /-1 if not available
          "OUT_KB" : value /-1 if not available
          "HB" : {
              "ERR" : "Never" / "2020-01-01 00:00:00"
              "MSG" : "OK" / "Connection timeout ..."
              "IN_KB" : value /-1 if not available
              "OUT_KB" : value /-1 if not available
              "FAILS" : value - 0 is best :)
            }
          "PAYL" ...
          "CMD"  ...
          "NTIF" ...
        }
      """
      dct_comms = {}    
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      dct_comms[ct.HB.COMM_INFO.IN_KB] = round(hb.get(ct.HB.COMM_INFO.IN_KB, -1), 3)
      dct_comms[ct.HB.COMM_INFO.OUT_KB] = round(hb.get(ct.HB.COMM_INFO.OUT_KB, -1), 3)
      dct_stats = hb.get(ct.HB.COMM_STATS, {})
      mapping = {
        ct.COMMS.COMMUNICATION_HEARTBEATS : "HB",
        ct.COMMS.COMMUNICATION_DEFAULT : "PAYL",
        ct.COMMS.COMMUNICATION_COMMAND_AND_CONTROL : "CMD",
        ct.COMMS.COMMUNICATION_NOTIFICATIONS : "NTIF",
      }
      for k,v in dct_stats.items():
        report = mapping[k]
        remote_error_time = v.get("ERRTM", None)
        local_error_time = remote_error_time
        if isinstance(remote_error_time, str):
          remote_tz = hb.get(ct.PAYLOAD_DATA.EE_TIMEZONE)
          local_error_time = self.log.utc_to_local(
            remote_error_time, 
            remote_utc=remote_tz, 
            fmt=ct.HB.TIMESTAMP_FORMAT_SHORT,
            as_string=True,
          )
        dct_comms[report] = {
          "ERR"     : local_error_time or "Never",
          "MSG"     : str(v.get("ERROR", None) or "OK"),
          "FAILS"   : v.get("FAILS", 0),
          ct.HB.COMM_INFO.IN_KB : v.get(ct.HB.COMM_INFO.IN_KB, -1),
          ct.HB.COMM_INFO.OUT_KB : v.get(ct.HB.COMM_INFO.OUT_KB, -1),
        }
      return dct_comms
      
    
    def network_node_info_available(self, eeid):
      return eeid in self.__network_nodes_list()
    
    
    def network_node_last_heartbeat(self, eeid):
      return self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
    
    def register_heartbeat(self, eeid, data):
      self.__register_heartbeat(eeid, data)
      return
        
    def network_nodes_status(self):
      dct_results = {}
      nodes = self.__network_nodes_list()
      for eeid in nodes:
        dct_res = self.network_node_status(eeid=eeid)
        dct_results[eeid] = dct_res
      return dct_results
    
    
    def network_node_main_loop(self, eeid):
      try:
        dct_timings = self.__network_node_last_loops_timings(eeid=eeid)
      except:
        return 1e10
      return round(dct_timings['main_loop_avg_time'],4)
      
          
    def network_node_is_ok_loops_timings(self, eeid, max_main_loop_timing=1):
      return self.network_node_main_loop(eeid) <= max_main_loop_timing


    def network_node_is_ok_uptime(self, eeid, min_uptime=60):
      uptime = self.__network_node_uptime(eeid=eeid)
      return uptime >= min_uptime


    def network_node_uptime(self, eeid, as_str=True):
      uptime_sec = self.__network_node_uptime(eeid=eeid, as_minutes=False)
      if as_str:
        result = self.log.elapsed_to_str(uptime_sec)
      else:
        result = uptime_sec
      return result


    def network_node_available_disk(self, eeid):
      available_disk = self.__network_node_last_available_disk(eeid=eeid, norm=False)
      return available_disk

    def network_node_available_disk_prc(self, eeid):
      prc_available_disk = self.__network_node_last_available_disk(eeid=eeid, norm=True)
      return prc_available_disk

    def network_node_is_ok_available_disk_prc(self, eeid, min_prc_available=0.15):
      # can create other heuristics based on what happened on the last x minutes interval (using _network_node_past_available_disk_by_interval)
      prc_available_disk = self.network_node_available_disk_prc(eeid=eeid)
      return prc_available_disk >= min_prc_available

    def network_node_is_ok_available_disk_size(self, eeid, min_gb_available=50):
      # can create other heuristics based on what happened on the last x minutes interval (using _network_node_past_available_disk_by_interval)
      available_disk = self.network_node_available_disk(eeid=eeid)
      return available_disk >= min_gb_available


    def network_node_available_memory(self, eeid):
      available_mem = self.__network_node_last_available_memory(eeid=eeid, norm=False)
      return available_mem

    def network_node_available_memory_prc(self, eeid):
      prc_available_mem = self.__network_node_last_available_memory(eeid=eeid, norm=True)
      return prc_available_mem

    def network_node_is_ok_available_memory_prc(self, eeid, min_prc_available=0.20):
      # can create other heuristics based on what happened on the last x minutes interval (using _network_node_past_available_memory_by_interval)
      prc_available_mem = self.network_node_available_memory_prc(eeid=eeid)
      return prc_available_mem >= min_prc_available

    def network_node_is_ok_available_memory_size(self, eeid, min_gb_available=2):
      # can create other heuristics based on what happened on the last x minutes interval (using _network_node_past_available_memory_by_interval)
      available_mem = self.network_node_available_memory(eeid=eeid)
      return available_mem >= min_gb_available


    def network_node_is_ok_device_status(self, eeid, dt_now=None):
      if self.__network_node_last_device_status(eeid=eeid) != ct.DEVICE_STATUS_ONLINE:
        return False

      if ct.DEVICE_STATUS_EXCEPTION in self.__network_node_past_device_status_by_interval(eeid=eeid, minutes=60):
        return False
      
      if self.network_node_last_seen(eeid=eeid, as_sec=True, dt_now=dt_now) > 60:
        return False

      return True

    def network_node_simple_status(self, eeid, dt_now=None):


      if ct.DEVICE_STATUS_EXCEPTION in self.__network_node_past_device_status_by_interval(eeid=eeid, minutes=60):
        return "PAST-EXCEPTION"
      
      if self.network_node_last_seen(eeid=eeid, as_sec=True, dt_now=dt_now) > 60:
        return "LOST STATUS"

      last_status = self.__network_node_last_device_status(eeid=eeid)
      return last_status
    
    
    def network_node_py_ver(self, eeid):
      result = None
      hb = self.__network_node_last_heartbeat(eeid)
      if isinstance(hb, dict):
        result = hb.get(ct.HB.PY_VER)
      return result      
    
    
    def network_node_version(self, eeid):
      result = None
      hb = self.__network_node_last_heartbeat(eeid)
      if isinstance(hb, dict):
        result = hb.get(ct.HB.VERSION)
      return result

    
    def network_node_is_recent(self, eeid, dt_now=None, max_recent_minutes=15):
      elapsed_seconds = self.network_node_last_seen(eeid=eeid, as_sec=True, dt_now=dt_now)
      mins = elapsed_seconds / 60
      recent = mins <= max_recent_minutes
      return recent


    def network_node_is_ok_cpu_used(self, eeid, max_cpu_used=50):
      # can create other heuristics based on what happened on the last x minutes interval (using _network_node_past_cpu_used_by_interval)
      return self.__network_node_last_cpu_used(eeid=eeid) <= max_cpu_used


    def network_node_is_available(self, eeid):
      ok_loops_timings = self.network_node_is_ok_loops_timings(eeid=eeid, max_main_loop_timing=5)
      ok_avail_disk = self.network_node_is_ok_available_disk_size(eeid=eeid, min_gb_available=5)
      ok_avail_mem = self.network_node_is_ok_available_memory_size(eeid=eeid)
      ok_cpu_used = self.network_node_is_ok_cpu_used(eeid=eeid, max_cpu_used=50)
      ok_device_status = self.network_node_is_ok_device_status(eeid=eeid)
      ok_uptime = self.network_node_is_ok_uptime(eeid=eeid, min_uptime=60)

      ok_node = ok_loops_timings and ok_avail_disk and ok_avail_mem and ok_cpu_used and ok_device_status and ok_uptime
      return ok_node
    
    def network_node_gpu_capability(self, eeid, device_id,
                                    min_gpu_used=20, max_gpu_used=90,
                                    min_prc_allocated_mem=20, max_prc_allocated_mem=90,
                                    min_gpu_mem_gb=4, max_gpu_mem_gb=30):

      gpus = self.__network_node_last_gpus(eeid=eeid)
      dct_ret = {
        'WEIGHTED_CAPABILITY'       : 0, 
        'INDIVIDUAL_CAPABILITIES'   : {}, 
        'DEVICE_ID'                 : device_id,
        'NAME'                      : None,
      }

      if not isinstance(device_id, int):
        if device_id is not None:
          self.P("Requested device_id '{}' in `_network_node_gpu_capability` is not integer for e2:{}".format(device_id, eeid), color='r')
        return dct_ret

      if device_id >= len(gpus):
        if device_id is not None:
          self.P("Requested device_id '{}' in `_network_node_gpu_capability` not available e2:{}".format(device_id, eeid), color='r')
        return dct_ret

      dct_g = gpus[device_id]
      dct_ret['NAME'] = dct_g.get('NAME')

      # these default values for `get`s are meant to generate a 0 score for each monitorized parameter if they are not returned in the dictionary
      allocated_mem = dct_g.get('ALLOCATED_MEM', 1)
      total_mem = dct_g.get('TOTAL_MEM', 1)
      gpu_used = dct_g.get('GPU_USED', 100)
      prc_allocated_mem = 100 * allocated_mem / total_mem

      capabilities = {
        'GPU_USED'      : {'SCORE': None, 'WEIGHT': 0.4, 'STR_VAL' : "{:.2f}%".format(gpu_used)},
        'GPU_MEM'       : {'SCORE': None, 'WEIGHT': 0.2, 'STR_VAL' : "{:.2f}GB".format(total_mem)},
        'ALLOCATED_MEM' : {'SCORE': None, 'WEIGHT': 0.4, 'STR_VAL' : "{:.2f}%".format(prc_allocated_mem)},
      }

      capabilities['ALLOCATED_MEM']['SCORE'] = round(exponential_score(
        left=min_prc_allocated_mem, right=max_prc_allocated_mem,
        val=prc_allocated_mem, right_is_better=False
      ), 2)
      capabilities['GPU_USED']['SCORE'] = round(exponential_score(
        left=min_gpu_used, right=max_gpu_used,
        val=gpu_used, right_is_better=False
      ), 2)
      capabilities['GPU_MEM']['SCORE'] = round(exponential_score(
        left=min_gpu_mem_gb, right=max_gpu_mem_gb,
        val=total_mem, right_is_better=True
      ), 2)
      dct_ret['INDIVIDUAL_CAPABILITIES'] = capabilities

      weighted_capability = 0
      for sw in capabilities.values():
        score = sw['SCORE']
        weight = sw['WEIGHT']

        if score == 0:
          # if any parameter has score 0, then it means something is wrong with the gpu, thus it can't be used
          return dct_ret

        weighted_capability += score * weight
      #endfor

      dct_ret['WEIGHTED_CAPABILITY'] = round(weighted_capability, 2)
      return dct_ret

    def network_node_default_gpu_capability(self, eeid,
                                            min_gpu_used=20, max_gpu_used=90,
                                            min_prc_allocated_mem=20, max_prc_allocated_mem=90,
                                            min_gpu_mem_gb=4, max_gpu_mem_gb=30):
      default_cuda = self.__network_node_default_cuda(eeid=eeid, as_int=True)
      dct_gpu_capability = self.network_node_gpu_capability(
        eeid=eeid, device_id=default_cuda,
        min_gpu_used=min_gpu_used,
        max_gpu_used=max_gpu_used,
        min_prc_allocated_mem=min_prc_allocated_mem,
        max_prc_allocated_mem=max_prc_allocated_mem,
        min_gpu_mem_gb=min_gpu_mem_gb,
        max_gpu_mem_gb=max_gpu_mem_gb,
      )

      return dct_gpu_capability
    

    def network_node_gpus_capabilities(self, eeid,
                                       min_gpu_used=20, max_gpu_used=90,
                                       min_prc_allocated_mem=20, max_prc_allocated_mem=90,
                                       min_gpu_mem_gb=8, max_gpu_mem_gb=30):
      capabilities = []
      for device_id in range(len(self.__network_node_last_gpus(eeid=eeid))):
        dct_gpu_capability = self.network_node_gpu_capability(
          eeid=eeid, device_id=device_id,
          min_gpu_used=min_gpu_used,
          max_gpu_used=max_gpu_used,
          min_prc_allocated_mem=min_prc_allocated_mem,
          max_prc_allocated_mem=max_prc_allocated_mem,
          min_gpu_mem_gb=min_gpu_mem_gb,
          max_gpu_mem_gb=max_gpu_mem_gb,
        )

        capabilities.append(dct_gpu_capability)
      # endfor

      return capabilities


    def network_top_n_avail_nodes(self, n, min_gpu_capability=10, verbose=1, permit_less=False):
      ### TODO continous process (1 iter/s), `network_top_n_aval_nodes` just have to return the map, not to compute it every time.
      ### TODO nmon_reader (shared object) that reads and returns the map. Effective nmon will be thread (continous process)

      def log_nodes_details():
        self.P("Top {} available nodes search finished. There are {}/{} available nodes in the network.".format(
          n, len(lst_available_nodes), len(lst_nodes)
        ))

        if len(lst_tuples) > 0:
          str_log = "There are {}/{} nodes with GPU capabilities: (GPU_USED: {}->{} / ALLOCATED_MEM: {}->{} / GPU_MEM: {}->{})".format(
            len(lst_tuples), len(lst_available_nodes),
            min_gpu_used, max_gpu_used,
            min_prc_allocated_mem, max_prc_allocated_mem,
            min_gpu_mem_gb, max_gpu_mem_gb,
          )
          for name, score in lst_tuples:
            individual_capabilities = dct_individual_capabilities[name]
            details = " | ".join(["{}: {}".format(_k,_v) for _k,_v in individual_capabilities.items()])
            str_log += "\n * {}: {} (device_id: {}) (details: {})".format(name, score, dct_device_id[name], details)
          #endfor
          self.P(str_log)
        else:
          self.P("No available node with GPU capabilities.", color='y')
        return
      #enddef

      lst_nodes = self.__network_nodes_list()
      lst_available_nodes = list(filter(
        lambda _eeid: self.network_node_is_available(eeid=_eeid),
        lst_nodes
      ))

      lst_capabilities = []
      dct_individual_capabilities = {}
      dct_device_id = {}
      min_gpu_used, max_gpu_used = 20, 90
      min_prc_allocated_mem, max_prc_allocated_mem = 20, 90
      min_gpu_mem_gb, max_gpu_mem_gb = 4, 40
      for _eeid in lst_available_nodes:
        dct_gpu_capability = self.network_node_default_gpu_capability(
          eeid=_eeid,
          min_gpu_used=min_gpu_used, max_gpu_used=max_gpu_used,
          min_prc_allocated_mem=min_prc_allocated_mem, max_prc_allocated_mem=max_prc_allocated_mem,
          min_gpu_mem_gb=min_gpu_mem_gb, max_gpu_mem_gb=max_gpu_mem_gb,
        )

        lst_capabilities.append(dct_gpu_capability['WEIGHTED_CAPABILITY'])
        dct_individual_capabilities[_eeid] = dct_gpu_capability['INDIVIDUAL_CAPABILITIES']
        dct_device_id[_eeid] = dct_gpu_capability['DEVICE_ID']
      #endfor

      np_capabilities_ranking = np.argsort(lst_capabilities)[::-1]

      np_nodes_sorted = np.array(lst_available_nodes)[np_capabilities_ranking]
      np_capabilities_sorted = np.array(lst_capabilities)[np_capabilities_ranking]

      good_indexes = np.where(np_capabilities_sorted >= min_gpu_capability)[0]

      good_nodes_sorted = list(map(lambda elem: str(elem), np_nodes_sorted[good_indexes][:n]))
      good_capabilities_sorted = list(map(lambda elem: float(elem), np_capabilities_sorted[good_indexes][:n]))
      lst_tuples = list(zip(good_nodes_sorted, good_capabilities_sorted))
      nr_found_workers = len(lst_tuples)

      if verbose >= 1:
        log_nodes_details()

      if nr_found_workers == n:
        final_log = "Successfully found {} workers: {}".format(nr_found_workers, good_nodes_sorted)
        color = 'g'
        ret = good_nodes_sorted
      else:
        final_log = "Unsuccessful search - only {}/{} workers qualify: {}".format(nr_found_workers, n, good_nodes_sorted)
        color = 'y'
        ret = [] if not permit_less else good_nodes_sorted
      #endif

      if verbose >= 1:
        self.P(final_log, color=color)

      return ret
    
    
    def network_save_status(self):
      self.P("Saving network map status...")
      # begin mutexed section
      self.log.lock_resource(NETMON_MUTEX)
      try:
        start_copy = time()
        _data_copy = deepcopy(self.__network_heartbeats)
        elapsed = time() - start_copy
        self.P("  Copy done in {:.2f}s".format(elapsed))
      except:
        pass
      self.log.unlock_resource(NETMON_MUTEX)
      # end mutexed section
      self.log.save_pickle_to_data(
        data=_data_copy, 
        fn='db.pkl',
        subfolder_path='network_monitor'
      )
      return
    
    
    def network_load_status(self, external_db=None):
      """
      Load network map status from previous session.
      """
      result = False
      
      if external_db is None:
        _fn = os.path.join(NETMON_DB_SUBFOLDER, NETMON_DB)
        db_file = self.log.get_data_file(_fn)
      else:
        db_file = external_db if os.path.isfile(external_db) else None
      #endif external_db is not None

      if db_file is not None:
        self.P("Previous nodes states found. Loading network map status...")
        __network_heartbeats = self.log.load_pickle(db_file)
        if __network_heartbeats is not None:
          # update the current network info with the loaded info
          # this means that all heartbeats received until this point
          # will be appended after the loaded ones 
          current_heartbeats = self.__network_heartbeats # save current heartbeats maybe already received
          self._set_network_heartbeats(__network_heartbeats)
          nr_loaded = len(__network_heartbeats)
          nr_received = len(current_heartbeats)
          previous_keys = set(__network_heartbeats.keys())
          current_keys = set(current_heartbeats.keys())
          not_present_keys = previous_keys - current_keys
          self.P("Current network of {} nodes inited with {} previous nodes".format(
            nr_received, nr_loaded), boxed=True
          )
          self.P("Nodes not present in current network: {}".format(not_present_keys), color='r')
          # lock the NETMON_MUTEX
          for eeid in current_heartbeats:
            for data in current_heartbeats[eeid]:
              # TODO: replace register_heartbeat with something simpler
              self.__register_heartbeat(eeid, data)
          # unlock the NETMON_MUTEX
          # end for
          result = True
        else:
          msg = "Error loading network map status"
          msg += "\n  File: {}".format(db_file)
          msg += "\n  Size: {}".format(os.path.getsize(db_file))
          self.P(msg, color='r')
        #endif __network_heartbeats loaded ok
      else:
        self.P("No previous network map status found.", color='r')
      #endif db_file is not None
      return result
    
      
    def network_node_last_seen(self, eeid, as_sec=True, dt_now=None):
      """
      Returns the `datetime` in local time when a particular remote node has last been seen
      according to its heart-beats.

      Parameters
      ----------
      eeid : str
        the node id.
        
      as_sec: bool (optional)
        will returns seconds delta instead of actual date
        
      dt_now: datetime (optional)
        replace now datetime with given one. default uses current datetime

      Returns
      -------
      dt_remote_to_local : datetime
        the local datetime when node was last seen.
      
        OR
      
      elapsed: float
        number of seconds last seen
      

      """
      hb = self.__network_node_last_heartbeat(eeid=eeid)
      ts = hb[ct.PAYLOAD_DATA.EE_TIMESTAMP]
      tz = hb.get(ct.PAYLOAD_DATA.EE_TIMEZONE)
      dt_remote_to_local = self.log.utc_to_local(ts, tz, fmt=ct.HB.TIMESTAMP_FORMAT)
      if dt_now is None:
        dt_now = dt.now()

      elapsed = dt_now.timestamp() - dt_remote_to_local.timestamp()
      
      if as_sec:
        return elapsed
      else:
        return dt_remote_to_local
    
      
    def network_node_default_gpu_history(
      self, eeid, minutes=60, dt_now=None, 
      reverse_order=True, return_timestamps=False
    ):
      device_id = self.__network_node_default_cuda(eeid=eeid)
      lst_statuses, timestamps = [], []
      if device_id is not None:
        result = self.__network_node_past_gpus_by_interval(
          eeid=eeid, minutes=minutes, dt_now=dt_now, 
          reverse_order=reverse_order, return_timestamps=return_timestamps,
        )
        if return_timestamps:
          lst_all_statuses, timestamps = result
        else:
          lst_all_statuses = result
          
        try:
          # TODO: fix this bug !
          lst_statuses = [x[device_id] for x in lst_all_statuses]
        except:
          pass
      
      if return_timestamps:
        return lst_statuses, timestamps
      return lst_statuses
      
    
    def network_node_default_gpu_average_avail_mem(self, eeid, minutes=60, dt_now=None):
      result = None
      lst_statuses = self.network_node_default_gpu_history(eeid=eeid, minutes=minutes, dt_now=dt_now)
      mem = [x['FREE_MEM'] for x in lst_statuses]
      try:
        val = np.mean(mem)
        result = round(val, 1) if not np.isnan(val) else None
      except:
        pass
      return result
    
    
    def network_node_default_gpu_average_load(self, eeid, minutes=60, dt_now=None):
      result = None
      lst_statuses = self.network_node_default_gpu_history(eeid=eeid, minutes=minutes, dt_now=dt_now)      
      try:
        gpuload = [x['GPU_USED'] for x in lst_statuses]
        val = np.mean(gpuload)
        result = round(val, 1) if not np.isnan(val) else None
      except:
        pass
      return result
    
    
    def network_node_remote_time(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.CURRENT_TIME)
    
    
    def network_node_deploy_type(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.GIT_BRANCH)      
    

    def network_node_is_supervisor(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      res = hb.get(ct.HB.EE_IS_SUPER, False)
      if res is None:
        res = False
      return res

    def network_node_address(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.EE_ADDR)
    
    def network_node_eeid(self, address):
      nodes = self.__network_nodes_list()
      for eeid in nodes:
        if self.network_node_address(eeid) == address:
          return eeid
      return None
    
    def network_node_whitelist(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.EE_WHITELIST)
    
    def network_node_local_tz(self, eeid, as_zone=True):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      if as_zone:
        return hb.get(ct.PAYLOAD_DATA.EE_TZ)
      else:
        return hb.get(ct.PAYLOAD_DATA.EE_TIMEZONE)
      
    def network_node_today_heartbeats(self, eeid, dt_now=None):
      """
      Returns the today (overridable via dt_now) heartbeats of a particular remote node.

      Parameters
      ----------
      eeid : str
          id of the node
      dt_now : datetime, optional
          override the now-time, by default None
      """
      if dt_now is None:
        dt_now = dt.now()
      dt_now = dt_now.replace(hour=0, minute=0, second=0, microsecond=0)
      hbs = self.__network_heartbeats[eeid]
      for hb in hbs:
        ts = hb[ct.PAYLOAD_DATA.EE_TIMESTAMP]
        dt_ts = self.log.utc_to_local(ts, fmt=ct.HB.TIMESTAMP_FORMAT)
        if dt_ts >= dt_now:
          yield hb
      
            
    def network_node_is_secured(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.SECURED, False) 
    
    
    def network_node_pipelines(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.CONFIG_STREAMS)
    
    
    def network_node_hb_interval(self, eeid):
      hb = self.__network_node_last_heartbeat(eeid=eeid, return_empty_dict=True)
      return hb.get(ct.HB.EE_HB_TIME)
  
    
    def network_node_status(self, eeid, min_uptime=60, dt_now=None):
      avail_disk = self.__network_node_last_available_disk(eeid=eeid, norm=False)
      avail_disk_prc = round(self.__network_node_last_available_disk(eeid=eeid, norm=True),3)

      avail_mem = self.__network_node_last_available_memory(eeid=eeid, norm=False)
      avail_mem_prc = round(self.__network_node_last_available_memory(eeid=eeid, norm=True), 3)

      is_alert_disk = avail_disk_prc < 0.15
      is_alert_ram = avail_mem_prc < 0.15

      #comms
      dct_comms = self.network_node_last_comm_info(eeid=eeid)
      #end comms

      dct_gpu_capability = self.network_node_default_gpu_capability(eeid=eeid)
      gpu_name = dct_gpu_capability['NAME']
      
      score=dct_gpu_capability['WEIGHTED_CAPABILITY']
      
      uptime_sec = round(self.__network_node_uptime(eeid=eeid, as_minutes=False),2)      
      uptime_min = uptime_sec / 60
      ok_uptime = uptime_sec >= (min_uptime * 60)
      
      working_status = self.network_node_simple_status(eeid=eeid, dt_now=dt_now)
      is_online = working_status == ct.DEVICE_STATUS_ONLINE
      
      recent = self.network_node_is_recent(eeid=eeid, dt_now=dt_now)

      trusted = recent and is_online and ok_uptime
      trust_val = exponential_score(left=0, right=min_uptime * 4, val=uptime_min, normed=True, right_is_better=True)
      trust = 0 if not is_online else round(trust_val,3)
      trust = trusted * trust
      
      score = round(score * trust_val,2)
      if trusted and score == 0:
        score = 10
        
      cpu_past1h = round(np.mean(self.__network_node_past_cpu_used_by_interval(eeid=eeid, minutes=60, dt_now=dt_now)),2)
      cpu_past1h = cpu_past1h if not np.isnan(cpu_past1h) else None
      main_loop_time = self.network_node_main_loop(eeid) 
      main_loop_freq = 0 if main_loop_time == 0 else 1 / (main_loop_time + 1e-14)
      address=self.network_node_address(eeid)
      
      trust_info = 'NORMAL_EVAL'
      
      if address is None:
        trusted = False
        trust = 0
        trust_info = 'NO_ADDRESS'
      
      is_secured = self.network_node_is_secured(eeid)
      trusted = is_secured and trusted
      
      dct_result = dict(
        address=address,        
        trusted=trusted,
        trust=trust,
        secured=is_secured,
        whitelist=self.network_node_whitelist(eeid),
        trust_info=trust_info,
        is_supervisor=self.network_node_is_supervisor(eeid),
        working=working_status,
        recent=recent,
        deployment=self.network_node_deploy_type(eeid) or "Unknown",
        version=self.network_node_version(eeid),
        py_ver=self.network_node_py_ver(eeid),
        last_remote_time=self.network_node_remote_time(eeid),
        node_tz=self.network_node_local_tz(eeid),
        node_utc=self.network_node_local_tz(eeid, as_zone=False),
        main_loop_avg_time=main_loop_time,
        main_loop_freq=round(main_loop_freq, 2),
        
        # main_loop_cap
        uptime=self.log.elapsed_to_str(uptime_sec),
        last_seen_sec=round(self.network_node_last_seen(eeid, as_sec=True, dt_now=dt_now),2),
        
        avail_disk=avail_disk,
        avail_disk_prc=avail_disk_prc,
        is_alert_disk=is_alert_disk,

        avail_mem=avail_mem,  
        avail_mem_prc=avail_mem_prc,  
        is_alert_ram=is_alert_ram,    
        
        cpu_past1h=cpu_past1h,        
        gpu_load_past1h=self.network_node_default_gpu_average_load(eeid=eeid, minutes=60, dt_now=dt_now),
        gpu_mem_past1h=self.network_node_default_gpu_average_avail_mem(eeid=eeid, minutes=60, dt_now=dt_now),
        gpu_name=gpu_name,
        SCORE=score,        
        #comms:
        comms=dct_comms,
        #end comms
      )
      return dct_result    
    
    
    def network_node_history(self, eeid, minutes=8*60, dt_now=None, reverse_order=True, hb_step=4):
      # TODO: set HIST_DEBUG to False
      HIST_DEBUG = True
      lst_heartbeats = self.__network_node_past_heartbeats_by_interval(
        eeid=eeid, minutes=minutes, dt_now=dt_now,
        reverse_order=True,
      )
      last_hb = lst_heartbeats[0]
        
      timestamps = self.__get_timestamps(lst_heartbeats)
      hb = self.__network_node_last_heartbeat(eeid)
      
      cpu_hist = self.__network_node_past_cpu_used_by_interval(
        eeid=eeid, minutes=minutes,  dt_now=dt_now, return_timestamps=HIST_DEBUG,
        reverse_order=True,
      )
            
      mem_avail_hist = self.__network_node_past_available_memory_by_interval(
        eeid=eeid, minutes=minutes, 
        reverse_order=True,
        # dt_now=dt_now, # must implement
      )

      gpu_hist = self.network_node_default_gpu_history(
        eeid=eeid, minutes=minutes, dt_now=dt_now, return_timestamps=HIST_DEBUG,        
        reverse_order=True, 
      )      
      
      if HIST_DEBUG: # debug / sanity-checks
        cpu_hist, cpu_timestamps = cpu_hist
        gpu_hist, gpu_timestamps = gpu_hist
        assert hb == last_hb
        assert timestamps == cpu_timestamps      
        assert timestamps == gpu_timestamps
      # endif debug

      gpu_load_hist = [x['GPU_USED'] for x in gpu_hist]
      gpu_mem_avail_hist = [x['FREE_MEM'] for x in gpu_hist]
      
      total_disk=hb[ct.HB.TOTAL_DISK]
      total_mem=hb[ct.HB.MACHINE_MEMORY]
      gpu_mem_total = gpu_hist[0]['TOTAL_MEM'] if len(gpu_hist) > 0 else None
      
      # we assume data is from oldest to last newest (reversed order)
      timestamps = timestamps[::hb_step]
      cpu_hist = cpu_hist[::hb_step]
      mem_avail_hist = mem_avail_hist[::hb_step]
      gpu_load_hist = gpu_load_hist[::hb_step]
      gpu_mem_avail_hist = gpu_mem_avail_hist[::hb_step]
      
      
      if not reverse_order:
        timestamps = list(reversed(timestamps))
        cpu_hist = list(reversed(cpu_hist))
        mem_avail_hist = list(reversed(mem_avail_hist))
        gpu_load_hist = list(reversed(gpu_load_hist))
        gpu_mem_avail_hist = list(reversed(gpu_mem_avail_hist))
      
      dct_result = OrderedDict(dict(
        total_disk=total_disk,
        total_mem=total_mem,
        mem_avail_hist=mem_avail_hist,
        cpu_hist=cpu_hist,
        gpu_mem_total=gpu_mem_total,
        gpu_load_hist=gpu_load_hist,
        gpu_mem_avail_hist=gpu_mem_avail_hist,
        timestamps=timestamps,
      ))
      
      return dct_result
    
  #endif


if __name__ == '__main__':
  from core import Logger
  l = Logger(lib_name='tstn', base_folder='.', app_folder='_local_cache')
  network_heartbeats = l.load_pickle_from_output('test_network_heartbeats.pkl')
  
  str_dt = '2023-05-25 18:13:00'
  dt_now = None # l.str_to_date(str_dt)
  eeids = list(network_heartbeats.keys())
  eeid = 'gts-test2' # eeids[0]
  
  nmon = NetworkMonitor(log=l)
  nmon._set_network_heartbeats(network_heartbeats)
  res = {}
  for eeid in eeids:
    if eeid == 'gts-test2':
      print()
    res[eeid] = nmon.network_node_status(eeid=eeid, min_uptime=120, dt_now=dt_now)
  l.P("Results:\n{}".format(
    json.dumps(res, indent=4), 
  ))
  l.P(nmon.network_node_history(eeid))
  