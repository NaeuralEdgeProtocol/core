import os
import shutil

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

  def __get_delta_logs(self):
    logs = list(self.logs)
    logs = "".join(logs)
    self.logs.clear()

    err_logs = list(self.err_logs)
    err_logs = "".join(err_logs)
    self.err_logs.clear()

    return logs, err_logs

  def __maybe_print_ngrok_logs(self):
    if self.logs_reader is not None:
      logs = self.logs_reader.get_next_characters()
      if len(logs) > 0:
        self.P(logs)
        self.logs.append(logs)

    if self.err_logs_reader is not None:
      err_logs = self.err_logs_reader.get_next_characters()
      if len(err_logs) > 0:
        self.P(err_logs, color='r')
        self.err_logs.append(err_logs)
    return

  def __maybe_init_assets(self):
    if self.assets_initialized:
      return

    self.P('Starting Nodejs app...')

    src_dir = self.cfg_assets
    dst_dir = self.script_temp_dir

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
    self.assets_initialized = True

    return

  def __maybe_run_nth_setup_command(self, idx):
    if idx > 0 and not self.setup_commands_finished[idx - 1]:
      # Previous setup command has not finished yet. Skip this one.
      return

    if not self.setup_commands_started[idx]:
      self.P(f"Running setup command nr {idx}: {self.cfg_setup_commands[idx]}")
      self.setup_commands_processes[idx], self.logs_reader, self.err_logs_reader = self.run_command(self.cfg_setup_commands[idx])
      self.setup_commands_started[idx] = True
    # endif setup command started

    if not self.setup_commands_finished[idx]:
      timeout = 3
      finished, failed = self.wait_for_command(self.setup_commands_processes[idx], timeout, [self.logs_reader, self.err_logs_reader])
      self.__maybe_print_ngrok_logs()
      self.logs_reader = None
      self.err_logs_reader = None

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

  def __has_finished_setup_commands(self):
    return all(self.setup_commands_finished)

  def __maybe_run_all_setup_commands(self):
    if self.__has_finished_setup_commands():
      return

    for idx in range(len(self.cfg_setup_commands)):
      self.__maybe_run_nth_setup_command(idx)
    return

  def __maybe_run_start_command(self):
    if not self.__has_finished_setup_commands():
      # Waiting for setup commands to finish before running the run command.
      return

    if not self.run_command_started:
      self.P(f"Running run command: {self.cfg_run_command}")
      self.run_command_process, self.logs_reader, self.err_logs_reader = self.run_command(self.cfg_run_command)

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

  def on_init(self):
    super(NodeJsWebAppPlugin, self).on_init()
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
    return

  def on_close(self):
    if not self.run_command_started:
      self.P("Nodejs server was never started. Skipping teardown.")
      return

    # Teardown nodejs
    try:
      self.P("Forcefully killing nodejs server")
      self.run_command_process.kill()
      timeout = 3
      self.wait_for_command(self.run_command_process, timeout, [self.logs_reader, self.err_logs_reader])
      self.__maybe_print_ngrok_logs()
      self.logs_reader = None
      self.err_logs_reader = None

      self.P("Killed nodejs server")
    except Exception as _:
      self.P('Could not kill nodejs server')

    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.
    return

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

  def process(self):
    self.__maybe_init_assets()

    self.__maybe_run_all_setup_commands()

    self.__maybe_run_start_command()

    self.__maybe_print_ngrok_logs()
    return
