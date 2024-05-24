from core.business.base import BasePluginExecutor
from core.business.mixins_admin.network_monitor_mixin import _NetworkMonitorMixin, NMonConst

__VER__ = '1.0.1'

_CONFIG = {
  **BasePluginExecutor.CONFIG,

  'ALLOW_EMPTY_INPUTS'            : True,
  
  "PROCESS_DELAY"                 : 10,
  
  "SUPERVISOR"                    : False, 
  "SUPERVISOR_ALERT_TIME"         : 30,
  "SUPERVISOR_LOG_TIME"           : 60,
  "SEND_IF_NOT_SUPERVISOR"        : False,
  
  "ALERT_RAISE_CONFIRMATION_TIME" : 1,
  "ALERT_LOWER_CONFIRMATION_TIME" : 2,
  "ALERT_DATA_COUNT"              : 3,
  "ALERT_RAISE_VALUE"             : 0.5,
  "ALERT_LOWER_VALUE"             : 0.4,
  "ALERT_MODE"                    : 'mean',
  "ALERT_MODE_LOWER"              : 'max',
  
  
  
  # debug stuff
  "LOG_INFO"            : False,
  "LOG_FULL_INFO"       : False,
  "EXCLUDE_SELF"        : False,
  "SAVE_NMON_EACH"      : 5, # this saves each SAVE_NMON_EACH * PROCESS_DELAY seconds

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES'],
  },
}


class NetMon01Plugin(
  BasePluginExecutor,
  _NetworkMonitorMixin,
  ):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(NetMon01Plugin, self).__init__(**kwargs)
    self._nmon_counter = 0
    self.__supervisor_log_time = 0
    self.__state_loaded = False
    return

  def startup(self):
    super().startup()
    return
  
  def on_init(self):
    is_supervisor = self.cfg_supervisor
    if isinstance(is_supervisor, str):
      self.P("Found string value for SUPERVISOR: {}. Converting to bool".format(is_supervisor))
      self.config_data['SUPERVISOR'] = is_supervisor.lower() == 'true'
    #endif is string
    return
  
  def _maybe_load_state(self):
    if self.__state_loaded:
      return
    self.__state_loaded = True
    self.netmon.network_load_status()
    return
  
  def _on_command(self, data, **kwargs):
    super(NetMon01Plugin, self)._on_command(data)
    request_type = 'history' #default to history
    request_options = {}
    if isinstance(data, dict):
      dct_cmd = {k.lower() : v for k,v in data.items()} # lower case instance command keys
      target_node = dct_cmd.get('node', None)
      target_addr = dct_cmd.get('addr', None)
      request_type = dct_cmd.get('request', 'history')
      request_options = dct_cmd.get('options', {})

    if target_node is not None:
      target_addr = self.netmon.network_node_addr(target_node) or target_addr

    if target_addr is not None:
      self.P("Network monitor on {} ({}) received request: {}".format(self.e2_addr, self.eeid, data))
      self._exec_netmon_request(
        target_addr=target_addr,
        request_type=request_type,
        request_options=request_options,
      )    
    else:
      self.P("Network monitor on {} ({}) received invalid request: {}".format(self.e2_addr, self.eeid, data), color='r')
    return
  

  def _process(self):
    payload = None
    self._nmon_counter += 1      
    self._maybe_load_state()
    
    # TODO: change to addresses later
    current_nodes, new_nodes = self._add_to_history()       
    ranking = self._get_rankings()    
    
    str_ranking = ", ".join(["{}:{:.0f}:{:.1f}s".format(a,b,c) for a,b,c in ranking])
    
    
    is_anomaly, alerted_nodes = False, None
    
    current_network = None
    current_alerted = None
    is_supervisor = False
    current_ranking = ranking
    current_new = new_nodes

    if self.cfg_supervisor:
      # save status
      save_nmon_each = int(min(self.cfg_save_nmon_each, 300))
      if (self._nmon_counter % save_nmon_each) == 0: 
        self.P("Saving netmon status for {} nodes".format(len(current_nodes)))
        self.netmon.network_save_status()
      #endif save status

      is_anomaly, alerted_nodes = self._supervisor_check()
      self.alerter_add_observation(int(is_anomaly))        
      current_network = current_nodes
      current_alerted = alerted_nodes
      is_supervisor = True  
      
      if (self.time() - self.__supervisor_log_time) > self.cfg_supervisor_log_time:
        self.P("***** Supervisor node sending status for network of {} nodes *****".format(len(current_network)))
        self.__supervisor_log_time = self.time()
      #endif supervisor log time       
    #endif supervisor or not
    
    if self.cfg_supervisor or self.cfg_send_if_not_supervisor:
      message="" if len(current_alerted) == 0 else "Missing/lost processing nodes: {}".format(list(current_alerted.keys()))
      # for this plugin only ALERTS should be used in UI/BE
      payload = self._create_payload(
        current_server=self.e2_addr,
        current_network=current_network,
        current_alerted=current_alerted,
        message=message,
        status=message,
        current_ranking=current_ranking,
        current_new=current_new,
        is_supervisor=is_supervisor,
      )        

    if self.cfg_log_full_info:
      self.P("Full info:\n{}".format(self.json.dumps(current_nodes, indent=4)))

    if self.cfg_log_info:
      self.P("Anomaly: {}, IsAlert: {}, IsNewAlert: {}, Alerted: {}, Ranking: {}".format(
        is_anomaly, self.alerter_is_alert(), self.alerter_is_new_alert(),
        list(alerted_nodes.keys()), str_ranking
      ))
      self.P("Alerter status: {}".format(self.get_alerter_status()))
    if self.alerter_is_new_alert():
      self.P("NetMon anomaly reported:\n********************\nAnomaly reported for: {}\n********************".format(alerted_nodes))
    #endif show alerts
    return payload