"""
IMPORTANT:
  The commands that the `CommunicationManager` responds to are defined directly in the Orchestrator `cmd_handlers_...` methods defined in `core/main/command_handlers.py`
"""

import json
from core import Logger
from core.local_libraries import _ConfigHandlerMixin
from core.manager import Manager
from time import time
from core import constants as ct

from core.bc import VerifyMessage, DefaultBlockEngine

import numpy as np


class CommunicationManager(Manager, _ConfigHandlerMixin):
  def __init__(self,
               log: Logger,
               config,
               shmem,
               device_version,
               environment_variables=None,
               avail_commands=[],
               **kwargs):
    self.shmem = shmem
    # TODO: add Shared Memory Manager instance to each plugin under ct.SHMEM.COMM
    self.config = config
    self._device_version = device_version

    self._environment_variables = environment_variables
    self._dct_comm_plugins = None
    self._last_print_info = time()
    self._command_queues = None
    self.__lst_commands_from_self = []
    self._predefined_commands = avail_commands

    self.avg_comm_loop_timings = 0

    super(CommunicationManager, self).__init__(log=log, prefix_log='[COMM]', **kwargs)
    return

  def startup(self):
    super().startup()
    self.config_data = self.config
    self._dct_comm_plugins = self._dct_subalterns
    self._command_queues = self._empty_commands()
    return
  
  @property
  def blockchain_manager(self) -> DefaultBlockEngine:
    return self.shmem[ct.BLOCKCHAIN_MANAGER]
  
  @property
  def is_secured(self):
    return self.log.config_data.get(ct.CONFIG_STARTUP_v2.K_SECURED, False)

  @property
  def _device_id(self):
    id = self.log.config_data.get(ct.CONFIG_STARTUP_v2.K_EE_ID, '')
    return id  


  @property
  def has_failed_comms(self):
    for comm in self._dct_comm_plugins.values():
      if comm.comm_failed_after_retries:
        self.P("Detected total communication failure on comm {}. This may generate shutdown/restart.".format(comm.__class__.__name__), color='error')
        return True
    return False  


  def _verify_command_signature(self, cmd) -> VerifyMessage:
    result = None
    if self.blockchain_manager is not None:
      result = self.blockchain_manager.verify(cmd, return_full_info=True, verify_allowed=True)
      if not result.valid:
        self.P("Command received from sender addr <{}> verification failed: {}".format(
          result.sender, result.message
          ),color='r'
        )
      else:
        self.P("Command received from sender addr <{}> verification OK.".format(result.sender))
    else:
      raise ValueError("Blockchain Manager is unavailable for verifying the incoming command data")
    return result
    

  def _empty_commands(self):
    return {
      x: [] for x in self._predefined_commands
    }


  def get_received_commands(self):
    # When someone from outside wants to get the received commands, they are automatically refreshed
    # in order to 'make space' for new commands

    # TODO: refactor with deque - this implementation is bad
    ret_received_commands = []
    for k, v in self._command_queues.items(): # for each command type
      for c in v: # for each command of that type
        # each command should be a tuple `payload, sender, session`
        ret_received_commands.append((k, c))
      # endfor each command of that type
    # endfor each command type
    self._command_queues = self._empty_commands()
    return ret_received_commands

  def _get_plugin_class(self, name):
    _module_name, _class_name, _class_def, _class_config = self._get_module_name_and_class(
      locations=ct.PLUGIN_SEARCH.LOC_COMM_PLUGINS,
      name=name,
      suffix=ct.PLUGIN_SEARCH.SUFFIX_COMM_PLUGINS,
      safety_check=True,  # perform safety check
      safe_locations=ct.PLUGIN_SEARCH.SAFE_LOC_COMM_PLUGINS,
    )

    if _class_def is None:
      msg = "Error loading communication plugin '{}'".format(name)
      self.P(msg, color='r')
      self._create_notification(
        notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
        msg=msg,
        info="No code/script defined for communication plugin '{}' in {}".format(
          name, ct.PLUGIN_SEARCH.LOC_COMM_PLUGINS)
      )
      raise ValueError(msg)
    # endif

    return _class_def, _class_config

  def start_communication(self):
    plugin_name = self.config["TYPE"]
    config_instance = self.config["PARAMS"]
    config_instance[ct.EE_ID] = self._device_id

    _class_def, _default_config = self._get_plugin_class(plugin_name)
    id_comm = 0
    for comm_type, paths in self.config["INSTANCES"].items():
      id_comm += 1

      comm_type = comm_type.upper()
      send_channel_name = paths.get("SEND_TO", None)
      recv_channel_name = paths.get("RECV_FROM", None)
      if send_channel_name is not None:
        send_channel_name = send_channel_name.upper()
      if recv_channel_name is not None:
        recv_channel_name = recv_channel_name.upper()

      try:
        comm_plugin = _class_def(
          log=self.log,
          shmem=self.shmem,
          signature=plugin_name,
          comm_type=comm_type,
          default_config=_default_config,
          upstream_config=config_instance,
          environment_variables=self._environment_variables,
          send_channel_name=send_channel_name,
          recv_channel_name=recv_channel_name,
          timers_section=str(id_comm) + '_' + comm_type,
        )
        comm_plugin.validate(raise_if_error=True)
        comm_plugin.start()
        self.add_subaltern(comm_type, comm_plugin)
      except Exception as exc:
        msg = "Exception '{}' when initializing communication plugin {}".format(exc, plugin_name)
        self.P(msg, color='r')
        self._create_notification(
          notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
          msg=msg,
          autocomplete_info=True
        )
        raise exc
      # end try-except
    # endfor

    return

  def send(self, data, event_type='PAYLOAD'):
    if event_type == 'COMMAND':
      communicator = self._dct_comm_plugins[ct.COMMS.COMMUNICATION_COMMAND_AND_CONTROL]

      receiver_id, receiver_addr, command_type, command_content = data

      should_bypass_commands_to_self = self._environment_variables.get("LOCAL_COMMAND_BYPASS", True)
      is_command_to_self = receiver_addr == self.blockchain_manager.address

      if should_bypass_commands_to_self and is_command_to_self:
        # if we are the receiver, we bypass sending and receiving the command to the network
        self.P("Bypassing network communication for {} local command to self".format(command_type), color='y')
        self.__lst_commands_from_self.append((command_type, command_content))
      else:
        command = {
          ct.EE_ID: receiver_id,
          ct.COMMS.COMM_SEND_MESSAGE.K_ACTION: command_type,
          ct.COMMS.COMM_SEND_MESSAGE.K_PAYLOAD: command_content,
          ct.COMMS.COMM_SEND_MESSAGE.K_INITIATOR_ID: self._device_id,
          ct.COMMS.COMM_SEND_MESSAGE.K_SESSION_ID: self.log.session_id,
          ct.COMMS.COMM_SEND_MESSAGE.K_TIME: self.log.now_str()
        }
        communicator.send((receiver_id, receiver_addr, command))
    else:
      message = {
        'EE_ID': self._device_id,
        'EE_EVENT_TYPE': event_type,
        'EE_VERSION': self._device_version,
        **data
      }

      if event_type == 'PAYLOAD':
        communicator = self._dct_comm_plugins[ct.COMMS.COMMUNICATION_DEFAULT]
        communicator.send(message)
        if len(communicator.loop_timings) > 0:
          self.avg_comm_loop_timings = np.mean(communicator.loop_timings)
      elif event_type == 'NOTIFICATION':
        communicator = self._dct_comm_plugins[ct.COMMS.COMMUNICATION_NOTIFICATIONS]
        communicator.send(message)
      elif event_type == 'HEARTBEAT':
        communicator = self._dct_comm_plugins[ct.COMMS.COMMUNICATION_HEARTBEATS]
        communicator.send(message)
      # endif
    # endif
    return

  def close(self):
    self.P("Clossing all comm plugins", color='y')
    for comm in self._dct_comm_plugins.values():
      comm.stop()
    self.P("Done closing all comm plugins.", color='y')
    return

  def maybe_process_incoming(self):
    communicator_for_config = self._dct_comm_plugins[ct.COMMS.COMMUNICATION_HEARTBEATS]
    incoming_commands = communicator_for_config.get_messages()
    for json_msg in incoming_commands:
      self.process_command_message(json_msg)
    self.process_commands_from_self()
    return self.get_received_commands()

  def get_total_bandwidth(self):
    inkB, outkB = 0, 0
    keys = list(self._dct_comm_plugins.keys())
    for name in keys:
      comm = self._dct_comm_plugins.get(name)
      if comm is not None:
        inkB += comm.get_incoming_bandwidth()
        outkB += comm.get_outgoing_bandwidth()
    return inkB, outkB

  def get_comms_status(self):
    dct_stats = {}
    keys = list(self._dct_comm_plugins.keys())
    for name in keys:
      comm = self._dct_comm_plugins.get(name)
      if comm is not None:
        errors, times = comm.get_error_report()
        dct_stats[name] = {
          'SVR': comm.has_server_conn,
          'RCV': comm.has_recv_conn,
          'SND': comm.has_send_conn,
          'ACT': comm.last_activity_time,
          'ADDR': comm.server_address,
          'FAILS': len(errors),
          'ERROR': errors[-1] if len(errors) > 0 else None,
          'ERRTM': times[-1] if len(times) > 0 else None,
          ct.HB.COMM_INFO.IN_KB: comm.get_incoming_bandwidth(),
          ct.HB.COMM_INFO.OUT_KB: comm.get_outgoing_bandwidth(),
        }
    return dct_stats

  def maybe_show_info(self):
    now = time()
    if (now - self._last_print_info) >= ct.COMMS.COMM_SECS_SHOW_INFO:
      communicator = self._dct_comm_plugins[ct.COMMS.COMMUNICATION_DEFAULT]
      dct_stats = self.get_comms_status()
      keys = list(dct_stats.keys())
      ml = max([len(k) for k in keys])
      lines = []
      inkB, outkB = 0, 0
      for n in dct_stats:
        inkB += dct_stats[n][ct.HB.COMM_INFO.IN_KB]
        outkB += dct_stats[n][ct.HB.COMM_INFO.OUT_KB]
        active = dct_stats[n]['ACT']
        line = '    {}: live {}, conn:{}, rcv:{}, snd:{}, fails:{}, err: {}, {}'.format(
          n + ' ' * (ml - len(n)),
          self.log.time_to_str(active),
          int(dct_stats[n]['SVR']),
          int(dct_stats[n]['RCV']),
          int(dct_stats[n]['SND']),
          dct_stats[n]['FAILS'],
          dct_stats[n]['ERRTM'],
          dct_stats[n]['ADDR'],
        )
        lines.append(line)

      self.P("Showing comms statistics (In/Out/Total {:.2f} kB / {:.2f} kB / {:.2f} kB):\n{}".format(
        inkB, outkB, inkB + outkB,
        "\n".join(lines)),
        color=ct.COLORS.COMM
      )
      self._last_print_info = now
      statistics_payloads_trip = communicator.statistics_payloads_trip
      if len(statistics_payloads_trip):
        self.P(statistics_payloads_trip)
    # endif
    return

  def _save_input_command(self, payload):
    fn = '{}.txt'.format(self.log.now_str())
    self.log.save_output_json(
      data_json=payload,
      fname=fn,
      subfolder_path='input_commands',
      verbose=False
    )
    return

  def process_commands_from_self(self):
    for action, payload in self.__lst_commands_from_self:
      # we populate the fields with our data because the message is supposed to be from us
      self.process_decrypted_command(
        action=action,
        payload=payload,
        sender_addr=self.blockchain_manager.address,
        initiator_id=self._device_id,
        session_id=self.log.session_id,
        validated_command=True,
      )
    self.__lst_commands_from_self = []
    return

  def process_decrypted_command(self, action, payload, sender_addr=None, initiator_id=None, session_id=None, validated_command=False, json_msg=None):
    """
    This method is called to process a command that has been decrypted.
    We assume the command has been validated at this point, so we can proceed with processing it.
    We can discard a command if action is None or if the action is not recognized.

    action: str
      The action to be performed

    payload: dict
      The payload of the command

    sender_addr: str
      The address of the sender

    initiator_id: str
      The ID of the initiator

    session_id: str
      The ID of the session

    validated_command: bool
      Whether the command has been validated or not

    json_msg: dict
      The original JSON message
    """
    if action is not None:
      action = action.upper()
      if payload is None:
        self.P("  Message with action '{}' does not contain payload".format(
          action), color='y'
        )
        payload = {}  # initialize payload
      # endif no payload

      if isinstance(payload, dict):
        # we add the sender address to the payload
        payload[ct.COMMS.COMM_RECV_MESSAGE.K_SENDER_ADDR] = sender_addr
        # we add or modify payload session & initiator for downstream tasks
        if payload.get(ct.COMMS.COMM_RECV_MESSAGE.K_INITIATOR_ID) is None or initiator_id is not None:
          payload[ct.COMMS.COMM_RECV_MESSAGE.K_INITIATOR_ID] = initiator_id
        if payload.get(ct.COMMS.COMM_RECV_MESSAGE.K_SESSION_ID) is None or session_id is not None:
          payload[ct.COMMS.COMM_RECV_MESSAGE.K_SESSION_ID] = session_id
        # we send the message that this command was validated or not
        payload[ct.COMMS.COMM_RECV_MESSAGE.K_VALIDATED] = validated_command
      # endif

      if action not in self._command_queues.keys():
        self.P("  '{}' - command unknown".format(action), color='y')
      else:
        # each command is a tuple as below
        self._command_queues[action].append((payload, sender_addr, initiator_id, session_id))
        # self._save_input_command(payload)
      # endif
    else:
      self.P('  Message does not contain action. Nothing to process...', color='y')
      self.P('  Message received: \n{}'.format(json_msg), color='y')
    return

  def process_command_message(self, json_msg):
    device_id = json_msg.get(ct.EE_ID, None)
    action = json_msg.get(ct.COMMS.COMM_RECV_MESSAGE.K_ACTION, None)
    payload = json_msg.get(ct.COMMS.COMM_RECV_MESSAGE.K_PAYLOAD, None)
    initiator_id = json_msg.get(ct.COMMS.COMM_RECV_MESSAGE.K_INITIATOR_ID, None)
    session_id = json_msg.get(ct.COMMS.COMM_RECV_MESSAGE.K_SESSION_ID, None)
    sender_addr = json_msg.get(ct.COMMS.COMM_RECV_MESSAGE.K_SENDER_ADDR, None)

    self.P("Received message with action '{}' from <{}:{}>".format(
      action, initiator_id, session_id),
      color='y'
    )
    failed = False
    
    if device_id != self._device_id:
      self.P('  Message is not for the current device {} != {}'.format(device_id, self._device_id), color='y')
      failed = True
      
    ### signature verification
    verify_msg = self._verify_command_signature(json_msg)
    if not verify_msg.valid:
      if self.is_secured:
        msg = "Received invalid command from {}({}):{} due to '{}'. Command will be DROPPED.".format(
          initiator_id, verify_msg.sender, json_msg, verify_msg.message
        )
        failed = True
      else:
        msg = "Received invalid command from {}({}):{} due to '{}'. Command is accepted due to UNSECURED node.".format(
          initiator_id, verify_msg.sender, json_msg, verify_msg.message
        )
        failed = False
      self.P(msg, color='error')
      self._create_notification(
        notif=ct.STATUS_TYPE.STATUS_ABNORMAL_FUNCTIONING,
        msg=msg,
        displayed=True,
      )
    else:
      msg = "* * * *  Command from {}({}) is VALIDATED.".format(
        initiator_id, verify_msg.sender
      )
      self.P(msg)
    ### end signature verification
    validated_command = verify_msg.valid 


    if failed:
      self.P("  Message dropped.", color='r')      
    else:
      is_encrypted = json_msg.get(ct.COMMS.COMM_RECV_MESSAGE.K_EE_IS_ENCRYPTED, False)
      if is_encrypted:
        encrypted_data = json_msg.pop(ct.COMMS.COMM_RECV_MESSAGE.K_EE_ENCRYPTED_DATA, None)
        str_data = self.blockchain_manager.decrypt(encrypted_data, sender_addr)
        if str_data is None:
          self.P("  Decryption failed. Message dropped.", color='r')
        else:
          try:
            dict_data = json.loads(str_data)
            action = dict_data.get(ct.COMMS.COMM_RECV_MESSAGE.K_ACTION, None)
            payload = dict_data.get(ct.COMMS.COMM_RECV_MESSAGE.K_PAYLOAD, None)
          except Exception as e:
            self.P("Error while decrypting message: {}\n{}".format(str_data, e), color='r')
      # endif is_encrypted

      # TODO: change this value to False when all participants send encrypted messages
      if not is_encrypted and not self._environment_variables.get("ACCEPT_UNENCRYPTED_COMMANDS", True):
        self.P("  Message is not encrypted. Message dropped because `ACCEPT_UNENCRYPTED_COMMANDS=False`.", color='r')
      else:
        self.process_decrypted_command(
          action=action,
          payload=payload,
          sender_addr=sender_addr,
          initiator_id=initiator_id,
          session_id=session_id,
          validated_command=validated_command,
          json_msg=json_msg
        )
    return

  def validate_macro(self):
    communication = self.config_data
    if communication is None:
      msg = "'COMMUNICATION' is not configured in `config_app.txt`"
      self.add_error(msg)
      return

    plugin_name = communication.get("TYPE", None)
    params = communication.get("PARAMS", None)
    dct_instances = communication.get("INSTANCES", None)

    if plugin_name is None:
      msg = "Parameter 'TYPE' is not configured for 'COMMUNICATION' in `config_app.txt`"
      self.add_error(msg)

    if params is None:
      msg = "Parameter 'PARAMS' is not configured for 'COMMUNICATION' in `config_app.txt`"
      self.add_error(msg)
    else:
      found_channels = []
      for k in params:
        if k in ct.COMMS.COMMUNICATION_VALID_CHANNELS:
          found_channels.append(k)

      if len(set(ct.COMMS.COMMUNICATION_VALID_CHANNELS) - set(found_channels)) != 0:
        self.add_error("Make sure that all communication channels {} are configured as 'PARAMS' for 'COMMUNICATION' in `config_app.txt`".format(
          ct.COMMS.COMMUNICATION_VALID_CHANNELS))
      port = params.get("PORT", None)
      if port is not None:
        if not isinstance(port, int):
          self.add_warning("Parameter 'PORT' is not an integer for 'COMMUNICATION' in `config_app.txt` - casting to int")
          params["PORT"] = int(port)
        # endif not int
      # endif port
          
    #endif params

    if dct_instances is None:
      msg = "Parameter 'INSTANCES' is not configured for 'COMMUNICATION' in `config_app.txt`"
      self.add_error(msg)
    else:
      found_instances = []
      for comm_type, paths in dct_instances.items():
        comm_type = comm_type.upper()
        send_channel_name = paths.get("SEND_TO", None)
        recv_channel_name = paths.get("RECV_FROM", None)
        if send_channel_name is not None:
          send_channel_name = send_channel_name.upper()
        if recv_channel_name is not None:
          recv_channel_name = recv_channel_name.upper()
        if comm_type not in ct.COMMS.COMMUNICATION_VALID_TYPES:
          msg = "Parameter 'INSTANCE' is misconfigured for 'COMMUNICATION' in `config_app.txt` - unknown instance {}; please try one of these: {}".format(
            comm_type, ct.COMMS.COMMUNICATION_VALID_TYPES)
          self.add_error(msg)

        if send_channel_name is not None and send_channel_name not in ct.COMMS.COMMUNICATION_VALID_CHANNELS:
          msg = "Parameter 'INSTANCE' is misconfigured for 'COMMUNICATION' in `config_app.txt` - for instance {} 'SEND_TO' is not valid; please try one of these: {}".format(
            comm_type, ct.COMMS.COMMUNICATION_VALID_CHANNELS)
          self.add_error(msg)

        if recv_channel_name is not None and recv_channel_name not in ct.COMMS.COMMUNICATION_VALID_CHANNELS:
          msg = "Parameter 'INSTANCE' is misconfigured for 'COMMUNICATION' in `config_app.txt` - for instance {} 'RECV_FROM' is not valid; please try one of these: {}".format(
            comm_type, ct.COMMS.COMMUNICATION_VALID_CHANNELS)
          self.add_error(msg)

        found_instances.append(comm_type)
      # endfor
      if len(set(ct.COMMS.COMMUNICATION_VALID_TYPES) - set(found_instances)) != 0:
        self.add_error("Make sure that all communication instances {} are configured as 'INSTANCES' for 'COMMUNICATION' in `config_app.txt`".format(
          ct.COMMS.COMMUNICATION_VALID_TYPES))

    # endif

    return
