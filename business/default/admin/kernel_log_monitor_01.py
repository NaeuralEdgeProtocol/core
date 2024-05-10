import subprocess

from core.business.base import BasePluginExecutor

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePluginExecutor.CONFIG,

  'ALLOW_EMPTY_INPUTS'  : True,

  'PROCESS_DELAY'       : 180,
  'KERNEL_LOG_LEVEL'    : 'emerg,alert,crit,err',

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES'],
  },
}

class KernelLogMonitor01Plugin(BasePluginExecutor):
  """
  Monitors kernel logs for errors via dmesg.
  """

  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self.last_exec_time = None
    super(KernelLogMonitor01Plugin, self).__init__(**kwargs)
    return

  def startup(self):
    super().startup()
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

  def process(self):
    payload_params = {}
    current_time = self.time()
    eps = 0.05
    minutes = round((current_time - self.last_exec_time) / 60 + eps, 3)

    try:
      level = self.cfg_kernel_log_level.lower()
      out = self._get_kernel_errors(minutes=minutes, level=level)
      if len(out) > 0:
        self.P(f"Found the following kernel errors:\n{out}", color='red')
        # Fill out the payload
        payload_params['IS_ALERT'] = True
        self.add_payload_by_fields(**payload_params)
    except Exception as E:
      self.P(f"Could not retrieve kernel errors, {E}", color="red")

    self.last_exec_time = current_time

    return