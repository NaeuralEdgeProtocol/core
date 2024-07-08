import subprocess

from core.business.base import BasePluginExecutor

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePluginExecutor.CONFIG,

  'ALLOW_EMPTY_INPUTS'  : True,

  'PROCESS_DELAY'       : 180,
  'KERNEL_LOG_LEVEL'    : 'emerg,alert,crit,err',
  'MAX_TEMPERATURE'     : None,

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES'],
  },
}

class MonCT:
  HOOK_PREFIX = '_system_health_monitor'
  # The following constants should really be defined externally
  # since they are part of plugin_base_utils
  TEMP_DATA_TEMP = 'temperatures'
  TEMP_DATA_MSG = 'message'
  TEMP_DATA_CURRENT = 'current'
  TEMP_DATA_HIGH = 'high'
  TEMP_DATA_CRITICAL = 'critical'

class SystemHealthMonitor01Plugin(BasePluginExecutor):
  """
  Monitors kernel logs for errors via dmesg.
  """

  CONFIG = _CONFIG

  def on_init(self):
    # Initialize the set of monitor hooks. Each monitor hook is a method
    # starting with _system_health_monitor, should have only self as the argument as
    # should return a string.
    self.monitor_hooks = []
    predicate = self.inspect.ismethod
    for name, method in self.inspect.getmembers(self, predicate=predicate):
      if name.startswith(MonCT.HOOK_PREFIX):
        self.P(f"Added {name} as a monitor hook!")
        self.monitor_hooks.append(method)
    #endfor all methods

    self.last_exec_time = self.time()
    return

  def _get_kernel_errors(self, minutes : float, level : str) -> str:
    """
    Return a string containing all errors from the OS kernel logs.

    Parameters
    ----------
    minutes: float, the number of minutes in the past for which the errors
      will pe captured.
    level: str, log level to be used when retrieving errors from the
      kernel logs.

    Returns
    -------
    str, a string containing all the kernel errors

    Note this can throw an exception if we don't have enough privileges
    to read from the kernel logs.
    """

    dmesg_args = [
      'dmesg',
      '--level', level,
      '-T',
      '--since', f"-{minutes}min"
    ]
    dmesg_process = subprocess.Popen(
      dmesg_args,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL
    )
    out, _ = dmesg_process.communicate()
    if dmesg_process.returncode != 0:
      raise RuntimeError("could not run dmesg")

    out = out.decode().strip()
    return out

  def _system_health_monitor_kernel_logs(self) -> str:
    """
    Get the last error log lines since the last run.

    Parameters
    ----------
    None

    Returns
    -------
    str: all kernel errors since the last run
    """
    current_time = self.time()
    eps = 0.05
    minutes = round((current_time - self.last_exec_time) / 60 + eps, 3)

    msg = ""
    try:
      level = self.cfg_kernel_log_level.lower()
      out = self._get_kernel_errors(minutes=minutes, level=level)
      if len(out) > 0:
        msg = f"Found the following kernel errors:\n{out}\n"
    except Exception as E:
      self.P(f"Could not retrieve kernel errors, {E}", color="red")

    return msg

  def _system_health_monitor_temperatures(self) -> str:
    """
    Get a string listing all temperature issues found on the device.

    Parameters
    ----------
    None

    Returns
    -------
    str: all kernel errors since the last run
    """
    temperature_info = self.get_temperature_sensors(as_dict=True)
    temps = temperature_info[MonCT.TEMP_DATA_TEMP]
    if temps in [None, {}]:
      self.P(temperature_info[MonCT.TEMP_DATA_MSG], color='r')
      return ""
    #endif no sensor data

    msg = ""
    for sensor, status in temps.items():
      current = status[MonCT.TEMP_DATA_CURRENT]
      temp_threshold = status[MonCT.TEMP_DATA_HIGH]
      if self.cfg_max_temperature is not None:
        temp_threshold = self.cfg_max_temperature
      if status[MonCT.TEMP_DATA_CURRENT] >= temp_threshold:
        sensor_str = f"{sensor} at {current}°C CRITICAL!"
        msg += f"High temperature detected: {sensor_str}\n"
      #endif high temp
    #endfor values

    if len(msg) > 0:
      msg = f"Found the following temperature issues:\n{msg}"

    return msg

  def process(self):
    current_time = self.time()

    # Run all monitoring hooks. We concatenate any interesting messages
    # and raise an alert if there are any such messages.
    msg = ""
    for hook in self.monitor_hooks:
      msg += hook()
    #endfor all hooks
    msg = msg.rstrip()

    if len(msg) > 0:
      self.P(msg, color='red')
      self.add_payload_by_fields(is_alert=True, status=msg)
    #endif signal found errors

    # Finally update the last run time.
    self.last_exec_time = current_time
    return
