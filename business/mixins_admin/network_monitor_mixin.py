
# TODO: move all string to PyE2
class NMonConst:
  NMON_CMD_HISTORY = 'history'
  NMON_CMD_LAST_CONFIG = 'last_config'
  NMON_CMD_E2 = 'e2'
  NMON_CMD_REQUEST = 'request'
  NMON_RES_CURRENT_SERVER = 'current_server'
  NMON_RES_E2_TARGET_ID = 'e2_target_id'
  NMON_RES_E2_TARGET_ADDR = 'e2_target_addr'
  NMON_RES_NODE_HISTORY = 'node_history'
  NMON_RES_PIPELINE_INFO = 'e2_pipelines'

class _NetworkMonitorMixin:
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__history = self.deque(maxlen=180)
    self.__active_nodes = set()
    self.__lost_nodes = self.defaultdict(int)
    return
  
  
  @property
  def active_nodes(self):
    return self.__active_nodes
  
  @property
  def all_nodes(self):
    return self.netmon.all_nodes

  
  def _add_to_history(self):
    nodes = self.netmon.network_nodes_status()
    new_nodes = []
    self.__history.append(nodes) 
    current_nodes = list(nodes.keys())
    for eeid in current_nodes:
      if eeid not in self.__active_nodes:
        if nodes[eeid]['last_seen_sec'] < self.cfg_supervisor_alert_time:
          if self.cfg_log_info:
            self.P("New node '{}':\n{}".format(eeid, self.json.dumps(nodes[eeid], indent=4)))
          new_nodes.append(eeid)
    self.__active_nodes.update(new_nodes)
    return nodes, new_nodes
  
  
  def _supervisor_check(self):
    is_alert, nodes = False, {}
    for eeid in list(self.__active_nodes):
      last_seen_ago = self.netmon.network_node_last_seen(eeid=eeid, as_sec=True)
      if last_seen_ago > self.cfg_supervisor_alert_time:
        if self.cfg_log_info:
          hb = self.netmon.network_node_last_heartbeat(eeid=eeid)              
          hb_t = hb.get(self.const.HB.CURRENT_TIME)
          ts = hb[self.const.PAYLOAD_DATA.EE_TIMESTAMP]
          tz = hb.get(self.const.PAYLOAD_DATA.EE_TIMEZONE)
          dt_local = self.log.utc_to_local(ts, tz, fmt=self.const.HB.TIMESTAMP_FORMAT)
          dt_now = self.datetime.now()
          
          elapsed = dt_now.timestamp() - dt_local.timestamp()
          
          self.P("Found issue with {}:\n\nLast seen: {}\nStatus: {}\nHB: {}\n\n".format(
            eeid, last_seen_ago, self.json.dumps(self.__history[-1][eeid], indent=4),
            self.json.dumps(dict(hb_t=hb_t, ts=ts, tz=tz, elapsed=elapsed),indent=4)
          ))
        #endif debug
        uptime_sec = self.netmon.network_node_uptime(eeid=eeid, as_str=True)
        is_alert = True
        nodes[eeid] = {
            'last_seen_sec' : round(last_seen_ago, 1),
            'uptime_sec' : uptime_sec,
        }        
        self.__lost_nodes[eeid] += 1
      else:
        if eeid in self.__lost_nodes:
          del self.__lost_nodes[eeid]
      #endif is active or not
    #endfor check if any "active" is not active anymore
    
    for lost_eeid in self.__lost_nodes:
      if self.__lost_nodes[lost_eeid] > 10 and lost_eeid in self.__active_nodes:
        self.__active_nodes.remove(lost_eeid)
        self.P("Removed '{}' from active nodes after {} fails. Ongoing issues: {}".format(
          lost_eeid, self.__lost_nodes[lost_eeid], {k:v for k,v in self.__lost_nodes.items()},
        ))
      #endif
    #endfor clean lost nodes after a while
    return is_alert, nodes    
    
  
  def _get_rankings(self):
    nodes = self.__history[-1]
    if self.cfg_exclude_self:
      nodes = {k:v for k,v in nodes.items() if k != self.eeid}
    ranking = [(eeid, nodes[eeid]['SCORE'], nodes[eeid]['last_seen_sec']) for eeid in nodes]
    ranking = sorted(ranking, key=lambda x:x[1], reverse=True)
    return ranking  
  
  
  def _exec_netmon_request(self,  target_id, target_addr, request_type, request_options={}):
    if not isinstance(request_options, dict):
      request_options = {}
    payload_params = {}
    elapsed_t = 0
    if target_id not in self.all_nodes:
      self.P("Received `{}` request for missing node '{}'".format(
        request_type, target_id), color='r'
      )
      return
    
    if request_type == NMonConst.NMON_CMD_HISTORY:
      step = request_options.get('step', 20)
      time_window_hours = request_options.get('time_window_hours', 24)
      if not isinstance(time_window_hours, int):
        time_window_hours = 24
      if not isinstance(step, int):
        step = 4
      minutes = time_window_hours * 60
      self.P("Received edge node history request for '{}', step={}, hours={}".format(
        target_id, step, minutes // 60
      ))
      start_t = self.time()
      info = self.netmon.network_node_history(
        eeid=target_id, hb_step=step, minutes=minutes,
        reverse_order=False
      )
      elapsed_t = round(self.time() - start_t, 5)
      addr = self.netmon.network_node_address(eeid=target_id)
      payload_params[NMonConst.NMON_RES_NODE_HISTORY] = info
    elif request_type == NMonConst.NMON_CMD_LAST_CONFIG:
      self.P("Received edge node status request for '{}'".format(target_id))
      info = self.netmon.network_node_pipelines(eeid=target_id)    
      addr = self.netmon.network_node_address(eeid=target_id)
      payload_params[NMonConst.NMON_RES_PIPELINE_INFO] = info
    else:
      self.P("Network monitor on `{}` received invalid request type `{}` for target:address <{}:{}>".format(
        self.eeid, request_type, target_id, target_addr
        ), color='r'
      )
      return
    # construct the payload
    payload_params[NMonConst.NMON_CMD_REQUEST] = request_type
    payload_params[NMonConst.NMON_RES_CURRENT_SERVER] = self.e2_addr
    payload_params[NMonConst.NMON_RES_E2_TARGET_ID] = target_id
    payload_params[NMonConst.NMON_RES_E2_TARGET_ADDR] = addr 
    self.P("  Network monitor sending <{}> response to <{}>".format(request_type, target_id))
    self.add_payload_by_fields(
      call_history_time=elapsed_t,
      **payload_params,
    )        
    return