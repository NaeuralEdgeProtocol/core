import os
import shutil
import subprocess
import tempfile

from core.business.base.web_app.base_web_app_plugin import BaseWebAppPlugin as BasePlugin

__VER__ = '0.0.0.0'

_CONFIG = {
  **BasePlugin.CONFIG,
  'USE_NGROK': False,
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

    self.prepared_env = self.deepcopy(os.environ)
    self.prepared_env["PWD"] = self._script_temp_dir
    self.prepared_env["PORT"] = str(self.port)

    self.assets_initialized = False

    self.setup_commands_started = [False] * len(self.cfg_setup_commands)
    self.setup_commands_finished = [False] * len(self.cfg_setup_commands)
    self.setup_commands_processes = [None] * len(self.cfg_setup_commands)

    self.run_command_started = False
    self.run_command_process = None
    return

  def __run_command(self, command):
    command_args = command.split(' ')
    process = subprocess.Popen(
      command_args,
      env=self.prepared_env,
      cwd=self._script_temp_dir
    )
    return process

  def __wait_for_command(self, process, timeout):
    process_finished = False
    failed = False
    try:
      process.wait(timeout)
      failed = process.returncode != 0
      process_finished = True
    except subprocess.TimeoutExpired:
      pass

    return process_finished, failed

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

  def process(self):
    self.maybe_init_assets()

    self.maybe_run_all_setup_commands()

    self.maybe_run_start_command()

    return

  def on_close(self):
    if not self.run_command_started:
      self.P("Nodejs server was never started. Skipping teardown.")
      return

    # Teardown nodejs
    try:
      self.P("Forcefully killing nodejs server")
      self.run_command_process.kill()
      self.P("Killed nodejs server")
    except Exception as _:
      self.P('Could not kill nodejs server')

    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.

    super(NodeJsWebAppPlugin, self).on_close()
    return
