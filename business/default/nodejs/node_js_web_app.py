import os
import shutil
import subprocess
import tempfile

from core.business.base.web_app.base_web_app_plugin import BaseWebAppPlugin as BasePlugin

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePlugin.CONFIG,
  'PROCESS_DELAY': 5,

  'USE_NGROK': True,
  'NGROK_ENABLED': True,
  'NGROK_DOMAIN': None,
  'NGROK_EDGE_LABEL': None,

  'PORT': None,
  'ASSETS': None,
  'SETUP_COMMANDS': [],

  'RUN_COMMAND': "",

  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES']
  },
}


class NodeJsWebAppPlugin(BasePlugin):
  """
  A plugin which handles a NodeJS web app.

  Assets must be a path to a directory containing the NodeJS app.
  """

  CONFIG = _CONFIG

  def on_init(self):
    super(NodeJsWebAppPlugin, self).on_init()
    self._script_temp_dir = tempfile.mkdtemp()

    self.prepared_env = self.__prepare_env()
    self.prepared_env["PWD"] = self._script_temp_dir
    self.prepared_env["PORT"] = str(self.port)

    if self.os_path.exists(self.os_path.join(self.cfg_assets, '.env')):
      with open(self.os_path.join(self.cfg_assets, '.env'), 'r') as f:
        for line in f:
          if line.startswith('#') or len(line.strip()) == 0:
            continue
          key, value = line.strip().split('=', 1)
          self.prepared_env[key] = value

    self.assets_initialized = False

    self.setup_commands_started = [False] * len(self.cfg_setup_commands)
    self.setup_commands_finished = [False] * len(self.cfg_setup_commands)
    self.setup_commands_processes = [None] * len(self.cfg_setup_commands)

    self.run_command_started = False
    self.run_command_process = None

    self.logs = self.deque(maxlen=1000)
    self.logs_reader = None
    self.err_logs = self.deque(maxlen=1000)
    self.err_logs_reader = None
    self.last_log_read_timestamp = 0
    return

  def __prepare_env(self):
    prepared_env = dict(self.os_environ)
    to_pop_keys = []
    for key in prepared_env:
      if key.startswith('EE_'):
        to_pop_keys.append(key)
    # endfor all keys

    for key in to_pop_keys:
      prepared_env.pop(key)
    return prepared_env

  def __run_command(self, command):
    command_args = command.split(' ')
    process = subprocess.Popen(
      command_args,
      env=self.prepared_env,
      cwd=self._script_temp_dir,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    self.logs_reader = self.LogReader(process.stdout)
    self.err_logs_reader = self.LogReader(process.stderr)
    return process

  def __wait_for_command(self, process, timeout):
    process_finished = False
    failed = False
    try:
      process.wait(timeout)
      self.sleep(0.1)
      self.logs_reader.stop()
      self.err_logs_reader.stop()

      self.read_current_logs()

      failed = process.returncode != 0
      process_finished = True
    except subprocess.TimeoutExpired:
      pass

    return process_finished, failed

  def get_current_logs(self):
    logs = []
    L = len(self.logs)
    for _ in range(L):
      logs.append(self.logs.popleft())
    logs = "".join(logs)

    err_logs = []
    E = len(self.err_logs)
    for _ in range(E):
      err_logs.append(self.err_logs.popleft())
    err_logs = "".join(err_logs)

    return logs, err_logs

  def _maybe_run_setup_command(self, idx):
    if idx > 0 and not self.setup_commands_finished[idx - 1]:
      # Previous setup command has not finished yet. Skip this one.
      return

    if not self.setup_commands_started[idx]:
      self.P(f"Running setup command nr {idx}: {self.cfg_setup_commands[idx]}")
      self.setup_commands_processes[idx] = self.__run_command(self.cfg_setup_commands[idx])
      self.setup_commands_started[idx] = True
    # endif setup command started

    if not self.setup_commands_finished[idx]:
      timeout = 3
      finished, failed = self.__wait_for_command(self.setup_commands_processes[idx], timeout)
      self.setup_commands_finished[idx] = finished

      if finished and not failed:
        self.add_payload_by_fields(
          command_type="setup",
          command_idx=idx,
          command_str=self.cfg_setup_commands[idx],
          command_status="success"
        )
        self.P(f"Setup command nr {idx} finished successfully")
      elif finished and failed:
        self.add_payload_by_fields(
          command_type="setup",
          command_idx=idx,
          command_str=self.cfg_setup_commands[idx],
          command_status="failed"
        )
        self.P(f"ERROR: Setup command nr {idx} finished with exit code {self.setup_commands_processes[idx].returncode}")

    # endif setup command finished
    return

  def has_finished_setup_commands(self):
    return all(self.setup_commands_finished)

  def maybe_init_assets(self):
    if self.assets_initialized:
      return

    self.P('Starting Nodejs app...')

    src_dir = self.cfg_assets
    dst_dir = self._script_temp_dir

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

        shutil.copy2(src_file_path, dst_file_path)
      # endfor all files
    # endfor os.walk

    # write .env file in the target directory
    # environment variables are passed in subprocess.Popen, so this is not needed
    # but it's useful for debugging
    with open(self.os_path.join(dst_dir, '.env'), 'w') as f:
      for key, value in self.prepared_env.items():
        f.write(f"{key}={value}\n")

    self.assets_initialized = True

    return

  def maybe_run_all_setup_commands(self):
    if self.has_finished_setup_commands():
      return

    for idx in range(len(self.cfg_setup_commands)):
      self._maybe_run_setup_command(idx)
    return

  def maybe_run_start_command(self):
    if not self.has_finished_setup_commands():
      # Waiting for setup commands to finish before running the run command.
      return

    if not self.run_command_started:
      self.P(f"Running run command: {self.cfg_run_command}")
      self.run_command_process = self.__run_command(self.cfg_run_command)

      timeout = 3
      self.sleep(timeout)

      if self.run_command_process.poll() is None:
        self.P("Server started successfully")
        self.add_payload_by_fields(
          command_type="run",
          command_str=self.cfg_run_command,
          command_status="success"
        )
        self.run_command_started = True
      else:
        self.P(f"ERROR: Run command finished with exit code {self.run_command_process.returncode}")
        self.add_payload_by_fields(
          command_type="run",
          command_str=self.cfg_run_command,
          command_status="failed"
        )
      # endif check server started
    # endif run command started

    return

  def read_current_logs(self):
    logs = self.logs_reader.get_next_characters()
    if len(logs) > 0:
      self.P(logs)
      self.logs.append(logs)

    err_logs = self.err_logs_reader.get_next_characters()
    if len(err_logs) > 0:
      self.P(err_logs, color='r')
      self.err_logs.append(err_logs)
    return

  def on_close(self):
    if not self.run_command_started:
      self.P("Nodejs server was never started. Skipping teardown.")
      return

    # Teardown nodejs
    try:
      self.P("Forcefully killing nodejs server")
      self.run_command_process.kill()
      self.__wait_for_command(self.run_command_process, 3)
      self.P("Killed nodejs server")
    except Exception as _:
      self.P('Could not kill nodejs server')

    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.

    super(NodeJsWebAppPlugin, self).on_close()
    return

  def on_command(self, data, delta_logs=None, full_logs=None, **kwargs):
    if (isinstance(data, str) and data.upper() == 'DELTA_LOGS') or delta_logs:
      # TODO: Implement delta logs
      self.add_payload_by_fields(
        on_command_request=data,
        logs=[]
      )
      pass
    if (isinstance(data, str) and data.upper() == 'FULL_LOGS') or full_logs:
      logs, err_logs = self.get_current_logs()
      self.add_payload_by_fields(
        on_command_request=data,
        logs=logs,
        err_logs=err_logs,
      )

    return

  def process(self):
    self.maybe_init_assets()

    self.maybe_run_all_setup_commands()

    self.maybe_run_start_command()

    self.read_current_logs()
    return
