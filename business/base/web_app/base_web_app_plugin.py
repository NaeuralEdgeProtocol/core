import os
import shutil
import subprocess
import tempfile

from jinja2 import Environment, FileSystemLoader

from core.business.base import BasePluginExecutor
from core.business.mixins_libs.ngrok_mixin import _NgrokMixinPlugin

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePluginExecutor.CONFIG,
  'ALLOW_EMPTY_INPUTS': True,
  'RUN_WITHOUT_IMAGE': True,
  'PROCESS_DELAY': 5,

  'USE_NGROK': False,
  'NGROK_ENABLED': False,
  'NGROK_DOMAIN': None,
  'NGROK_EDGE_LABEL': None,

  'ASSETS': None,

  'SETUP_COMMANDS': [],
  'START_COMMANDS': [],

  'PORT': None,

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES']
  },
}


class BaseWebAppPlugin(_NgrokMixinPlugin, BasePluginExecutor):
  """
  A base plugin which will handle the lifecycle of a web application.
  Through this plugin, you can expose your business logic as a web application,
  using some implementation of a web server.

  You can also deploy your web application to the internet using ngrok.
  To do this, set the `USE_NGROK` flag to True in config and set the necessary
  environment variables.

  TODO: add ngrok necessary data in the config (after securing the configs)
  """

  CONFIG = _CONFIG

  def __check_port_valid(self):
    # Check the config as we're going to use it to start processes.
    if not isinstance(self.cfg_port, int):
      raise ValueError("Port not an int")
    if self.cfg_port < 0 or self.cfg_port > 65535:
      raise ValueError("Invalid port value {}".format(self.cfg_port))
    return

  def __allocate_port(self):
    self.lock_resource('USED_PORTS')
    if 'USED_PORTS' not in self.plugins_shmem:
      self.plugins_shmem['USED_PORTS'] = {}
    dct_shmem_ports = self.plugins_shmem['USED_PORTS']
    used_ports = dct_shmem_ports.values()

    if self.cfg_port is not None:
      self.__check_port_valid()

      if self.cfg_port in used_ports:
        raise Exception("Port {} is already in use.".format(self.cfg_port))
      else:
        dct_shmem_ports[self.str_unique_identification] = self.cfg_port
    else:
      port = self.np.random.randint(49152, 65535)
      while port in used_ports:
        port = self.np.random.randint(49152, 65535)
      # endwhile
      dct_shmem_ports[self.str_unique_identification] = port
    # endif port
    self.unlock_resource('USED_PORTS')
    return

  def __prepare_env(self):
    # pop all `EE_` keys
    prepared_env = dict(self.os_environ)
    to_pop_keys = []
    for key in prepared_env:
      if key.startswith('EE_'):
        to_pop_keys.append(key)
    # endfor all keys

    for key in to_pop_keys:
      prepared_env.pop(key)

    # add mandatory keys
    prepared_env["PWD"] = self.script_temp_dir
    prepared_env["PORT"] = str(self.port)

    # add optional keys, found in `.env` file from assets
    env_file_path = self.os_path.join(self.cfg_assets, '.env')
    if self.os_path.exists(env_file_path):
      with open(env_file_path, 'r') as f:
        for line in f:
          if line.startswith('#') or len(line.strip()) == 0:
            continue
          key, value = line.strip().split('=', 1)
          prepared_env[key] = value

    # write .env file in the target directory
    # environment variables are passed in subprocess.Popen, so this is not needed
    # but it's useful for debugging
    with open(self.os_path.join(self.script_temp_dir, '.env_used'), 'w') as f:
      for key, value in prepared_env.items():
        f.write(f"{key}={value}\n")

    return prepared_env

  @property
  def port(self):
    if 'USED_PORTS' not in self.plugins_shmem:
      return None
    if self.str_unique_identification not in self.plugins_shmem['USED_PORTS']:
      return None
    port = self.plugins_shmem['USED_PORTS'][self.str_unique_identification]
    return port

  def _on_init(self):
    self.__allocate_port()
    self.script_temp_dir = tempfile.mkdtemp()
    self.prepared_env = self.__prepare_env()

    self.assets_initialized = False

    self.setup_commands_started = [False] * len(self.get_setup_commands())
    self.setup_commands_finished = [False] * len(self.get_setup_commands())
    self.setup_commands_processes = [None] * len(self.get_setup_commands())
    self.setup_commands_start_time = [None] * len(self.get_setup_commands())

    self.start_commands_started = [False] * len(self.get_start_commands())
    self.start_commands_finished = [False] * len(self.get_start_commands())
    self.start_commands_processes = [None] * len(self.get_start_commands())
    self.start_commands_start_time = [None] * len(self.get_start_commands())

    self.logs = self.deque(maxlen=1000)
    self.dct_logs_reader = {}

    self.err_logs = self.deque(maxlen=1000)
    self.dct_err_logs_reader = {}

    self.P(f"Port: {self.port}")
    self.P(f"Setup commands: {self.get_setup_commands()}")
    self.P(f"Start commands: {self.get_start_commands()}")

    super(BaseWebAppPlugin, self)._on_init()
    return

  def get_setup_commands(self):
    try:
      return super(BaseWebAppPlugin, self).get_setup_commands() + self.cfg_setup_commands
    except AttributeError:
      return self.cfg_setup_commands

  def get_start_commands(self):
    try:
      return super(BaseWebAppPlugin, self).get_start_commands() + self.cfg_start_commands
    except AttributeError:
      return self.cfg_start_commands

  @property
  def jinja_args(self):
    return {}

  def _on_close(self):
    self.lock_resource('USED_PORTS')
    if 'USED_PORTS' in self.plugins_shmem:
      self.plugins_shmem['USED_PORTS'].pop(self.str_unique_identification, None)
    self.unlock_resource('USED_PORTS')

    super(BaseWebAppPlugin, self)._on_close()
    return

  # Exposed methods
  def __run_command(self, command, read_stdout=True, read_stderr=True):
    if isinstance(command, list):
      command_args = command
    elif isinstance(command, str):
      command_args = command.split(' ')
    else:
      raise ValueError("Command must be a string or a list of strings")

    process = subprocess.Popen(
      command_args,
      env=self.prepared_env,
      cwd=self.script_temp_dir,
      stdout=subprocess.PIPE if read_stdout else None,
      stderr=subprocess.PIPE if read_stderr else None,
    )
    logs_reader = self.LogReader(process.stdout) if read_stdout else None
    err_logs_reader = self.LogReader(process.stderr) if read_stderr else None
    return process, logs_reader, err_logs_reader

  def __wait_for_command(self, process, timeout, lst_log_reader=None):
    if lst_log_reader is None:
      lst_log_reader = []
    process_finished = False
    failed = False
    try:
      process.wait(timeout)
      for log_reader in lst_log_reader:
        if log_reader is not None:
          log_reader.stop()
      # endfor
      failed = process.returncode != 0
      process_finished = True
    except subprocess.TimeoutExpired:
      pass

    return process_finished, failed

  def initialize_assets(self, src_dir, dst_dir, jinja_args):
    """
    Initialize and copy fastapi assets, expanding any jinja templates.
    All files from the source directory are copied copied to the
    destination directory with the following exceptions:
      - are symbolic links are ignored
      - files named ending with .jinja are expanded as jinja templates,
        .jinja is removed from the filename and the result copied to
        the destination folder.
    This maintains the directory structure of the source folder.

    Parameters
    ----------
    src_dir: str, path to the source directory
    dst_dir: str, path to the destination directory
    jinja_args: dict, jinja keys to use while expanding the templates

    Returns
    -------
    None
    """
    self.P(f'Copying assets from {src_dir} to {dst_dir} with keys {jinja_args}')

    env = Environment(loader=FileSystemLoader('.'))
    # Walk through the source directory.
    for root, _, files in os.walk(src_dir):
      for file in files:
        src_file_path = self.os_path.join(root, file)
        dst_file_path = self.os_path.join(
          dst_dir,
          self.os_path.relpath(src_file_path, src_dir)
        )

        # If we have a symlink don't do anything.
        if self.os_path.islink(src_file_path):
          continue

        # Make sure the destination directory exists.
        os.makedirs(self.os_path.dirname(dst_file_path), exist_ok=True)

        # If this file is a jinja template render it to a file with the
        # .jinja suffix removed.
        if src_file_path.endswith(('.jinja')):
          dst_file_path = dst_file_path[:-len('.jinja')]
          template = env.get_template(src_file_path)
          rendered_content = template.render(jinja_args)
          with open(dst_file_path, 'w') as f:
            f.write(rendered_content)
          continue
        # endif jinja template

        # This is not a jinja template, just copy it do the destination
        # folder as is.
        shutil.copy2(src_file_path, dst_file_path)
      # endfor all files
    # endfor os.walk

    self.P("Assets copied successfully")

    return

  def __maybe_print_key_logs(self, key):
    logs_reader = self.dct_logs_reader.get(key)
    if logs_reader is not None:
      logs = logs_reader.get_next_characters()
      if len(logs) > 0:
        self.P(f"[{key}]: {logs}")
        self.logs.append(f"[{key}]: {logs}")

    err_logs_reader = self.dct_err_logs_reader.get(key)
    if err_logs_reader is not None:
      err_logs = err_logs_reader.get_next_characters()
      if len(err_logs) > 0:
        self.P(f"[{key}]: {err_logs}")
        self.logs.append(f"[{key}]: {err_logs}")
    return

  def __maybe_print_all_logs(self):
    for key, logs_reader in self.dct_logs_reader.items():
      if logs_reader is not None:
        logs = logs_reader.get_next_characters()
        if len(logs) > 0:
          self.P(f"[{key}]: {logs}")
          self.logs.append(f"[{key}]: {logs}")

    for key, err_logs_reader in self.dct_err_logs_reader.items():
      if err_logs_reader is not None:
        err_logs = err_logs_reader.get_next_characters()
        if len(err_logs) > 0:
          self.P(f"[{key}]: {err_logs}")
          self.logs.append(f"[{key}]: {err_logs}")
    return

  def __maybe_read_and_stop_key_log_readers(self, key):
    logs_reader = self.dct_logs_reader.get(key)
    if logs_reader is not None:
      logs_reader.stop()

    err_logs_reader = self.dct_err_logs_reader.get(key)
    if err_logs_reader is not None:
      err_logs_reader.stop()

    self.__maybe_print_key_logs(key)

    self.dct_logs_reader.pop(key, None)
    self.dct_err_logs_reader.pop(key, None)
    return

  def __maybe_read_and_stop_all_log_readers(self):
    # TODO: refactor this to stop all log readers
    if self.log_reader is not None:
      self.log_reader.stop()
    if self.error_log_reader is not None:
      self.error_log_reader.stop()

    self.__maybe_print_logs()

    self.log_reader = None
    self.error_log_reader = None
    return

  def __maybe_init_assets(self):
    if self.assets_initialized:
      return

    self.initialize_assets(
      src_dir=self.cfg_assets,
      dst_dir=self.script_temp_dir,
      jinja_args=self.jinja_args
    )

    self.assets_initialized = True
    return

  def __maybe_run_nth_setup_command(self, idx, timeout=None):
    if idx > 0 and not self.setup_commands_finished[idx - 1]:
      # Previous setup command has not finished yet. Skip this one.
      return

    if not self.setup_commands_started[idx]:
      self.P(f"Running setup command nr {idx}: {self.get_setup_commands()[idx]}")
      proc, logs_reader, err_logs_reader = self.__run_command(self.get_setup_commands()[idx])
      self.setup_commands_processes[idx] = proc
      self.dct_logs_reader[f"setup_{idx}"] = logs_reader
      self.dct_err_logs_reader[f"setup_{idx}"] = err_logs_reader

      self.setup_commands_started[idx] = True
      self.setup_commands_start_time[idx] = self.time()
    # endif setup command started

    if not self.setup_commands_finished[idx]:
      finished, failed = self.__wait_for_command(
        process=self.setup_commands_processes[idx],
        timeout=0.1,
        lst_log_reader=[
          self.dct_logs_reader[f"setup_{idx}"],
          self.dct_err_logs_reader[f"setup_{idx}"]
        ]
      )

      self.setup_commands_finished[idx] = finished

      if finished and not failed:
        self.__maybe_read_and_stop_key_log_readers(f"setup_{idx}")
        self.add_payload_by_fields(
          command_type="setup",
          command_idx=idx,
          command_str=self.get_setup_commands()[idx],
          command_status="success"
        )
        self.P(f"Setup command nr {idx} finished successfully")
      elif finished and failed:
        self.__maybe_read_and_stop_key_log_readers(f"setup_{idx}")
        self.add_payload_by_fields(
          command_type="setup",
          command_idx=idx,
          command_str=self.get_setup_commands()[idx],
          command_status="failed"
        )
        self.P(f"ERROR: Setup command nr {idx} finished with exit code {self.setup_commands_processes[idx].returncode}")
        # TODO: stop all setup if error occurs
      elif not finished and timeout is not None and timeout > 0:
        if self.time() - self.setup_commands_start_time[idx] > timeout:
          self.setup_commands_processes[idx].kill()
          self.__maybe_read_and_stop_key_log_readers(f"setup_{idx}")
          self.P(f"ERROR: Setup command nr {idx} timed out")
          self.add_payload_by_fields(
            command_type="setup",
            command_idx=idx,
            command_str=self.get_setup_commands()[idx],
            command_status="timeout"
          )
          # TODO: stop all setup if error occurs
    # endif setup command finished
    return

  def __has_finished_setup_commands(self):
    return all(self.setup_commands_finished)

  def __maybe_run_all_setup_commands(self):
    if self.__has_finished_setup_commands():
      return

    for idx in range(len(self.get_setup_commands())):
      self.__maybe_run_nth_setup_command(idx)
    return

  def __has_finished_start_commands(self):
    return all(self.start_commands_finished)

  def __maybe_run_nth_start_command(self, idx, timeout=5):
    if idx > 0 and not self.start_commands_finished[idx - 1]:
      # Previous start command has not finished yet. Skip this one.
      return

    if not self.start_commands_started[idx]:
      self.P(f"Running start command nr {idx}: {self.get_start_commands()[idx]}")
      proc, logs_reader, err_logs_reader = self.__run_command(self.get_start_commands()[idx])
      self.start_commands_processes[idx] = proc
      self.dct_logs_reader[f"start_{idx}"] = logs_reader
      self.dct_err_logs_reader[f"start_{idx}"] = err_logs_reader

      self.start_commands_started[idx] = True
      self.start_commands_start_time[idx] = self.time()
    # endif start command started

    if not self.start_commands_finished[idx]:
      finished, _ = self.__wait_for_command(
        process=self.start_commands_processes[idx],
        timeout=0.1,
        lst_log_reader=[
          self.dct_logs_reader[f"start_{idx}"],
          self.dct_err_logs_reader[f"start_{idx}"]
        ]
      )

      self.start_commands_finished[idx] = finished

      if finished:
        self.__maybe_read_and_stop_key_log_readers(f"start_{idx}")
        self.add_payload_by_fields(
          command_type="start",
          command_idx=idx,
          command_str=self.get_start_commands()[idx],
          command_status="failed"
        )
        self.P(f"Start command nr {idx} finished unexpectedly. Please check the logs.")
      elif self.time() - self.start_commands_start_time[idx] > timeout:
        self.start_commands_finished[idx] = True
        self.add_payload_by_fields(
          command_type="start",
          command_idx=idx,
          command_str=self.get_start_commands()[idx],
          command_status="success"
        )
        self.P(f"Start command nr {idx} is running")
        # TODO: stop all start if error occurs
    # endif setup command finished
    return

  def __maybe_run_all_start_commands(self):
    if self.__has_finished_start_commands():
      return

    if not self.__has_finished_setup_commands():
      return

    for idx in range(len(self.get_start_commands())):
      self.__maybe_run_nth_start_command(idx)
    return

  def _process(self):
    self.__maybe_init_assets()

    self.__maybe_run_all_setup_commands()

    self.__maybe_run_all_start_commands()

    self.__maybe_print_all_logs()
    return

  def on_close(self):
    if not any(self.start_commands_started):
      self.P("Server was never started. Skipping teardown.")
      return

    # Teardown server
    for idx, process in enumerate(self.start_commands_processes):
      try:
        if process is not None:
          self.P(f"Forcefully killing start command nr {idx}")
          process.kill()
          self.__wait_for_command(
            process=process,
            timeout=3,
            lst_log_reader=[
              self.dct_logs_reader[f"start_{idx}"],
              self.dct_err_logs_reader[f"start_{idx}"]
            ]
          )
          self.__maybe_print_key_logs(f"start_{idx}")
          self.P(f"Killed start command nr {idx}")
      except Exception as _:
        self.P(f'Could not kill start command nr {idx}')

    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.
    return

  def __get_delta_logs(self):
    logs = list(self.logs)
    logs = "".join(logs)
    self.logs.clear()

    err_logs = list(self.err_logs)
    err_logs = "".join(err_logs)
    self.err_logs.clear()

    return logs, err_logs

  def on_command(self, data, delta_logs=None, full_logs=None, **kwargs):
    if (isinstance(data, str) and data.upper() == 'DELTA_LOGS') or delta_logs:
      logs, err_logs = self.__get_delta_logs()
      self.add_payload_by_fields(
        on_command_request=data,
        logs=logs,
        err_logs=err_logs,
      )
    if (isinstance(data, str) and data.upper() == 'FULL_LOGS') or full_logs:
      # TODO: Implement full logs
      self.add_payload_by_fields(
        on_command_request=data,
        logs=[]
      )

    return
