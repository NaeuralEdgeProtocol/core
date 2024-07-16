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

  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES']
  },
}


class NodeJsWebAppPlugin(BasePlugin):
  """
  A plugin which handles a NodeJS web app.
  """

  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self.nodejs_process = None
    super(NodeJsWebAppPlugin, self).__init__(**kwargs)
    return

  def get_web_server_path(self):
    return self._script_temp_dir

  def _initialize_assets(self, src_dir, dst_dir):
    """
    Initialize and copy nodejs assets.
    All files from the source directory are copied copied to the
    destination directory with the following exceptions:
      - are symbolic links are ignored
    This maintains the directory structure of the source folder.

    Parameters
    ----------
    src_dir: str, path to the source directory
    dst_dir: str, path to the destination directory

    Returns
    -------
    None
    """
    self.P(f'Copying nodejs assets from {src_dir} to {dst_dir}')

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

        # This is not a jinja template, just copy it do the destination
        # folder as is.
        shutil.copy2(src_file_path, dst_file_path)
      # endfor all files
    # endfor os.walk

    # make sure assets folder exists
    os.makedirs(self.os_path.join(dst_dir, 'assets'), exist_ok=True)
    self.assets_initialized = True
    return

  def on_init(self):
    super(NodeJsWebAppPlugin, self).on_init()
    self._script_temp_dir = tempfile.mkdtemp()

    self.prepared_env = self.deepcopy(os.environ)
    self.prepared_env["PWD"] = self._script_temp_dir
    self.prepared_env["PORT"] = str(self.port)

    self.assets_initialized = False

    self.npm_init_started = False
    self.npm_init_finished = False

    self.npm_install_started = False
    self.npm_install_finished = False

    self.npm_started = False
    return

  def process(self):
    if not self.assets_initialized:
      # Start the Nodejs app
      self.P('Starting Nodejs app...')
      script_path = self.os_path.join(self._script_temp_dir, 'main.py')
      self.P("Using script at {}".format(script_path))

      src_dir = self.os_path.join('plugins', 'business', 'nodejs', self.cfg_assets)

      self._initialize_assets(src_dir, self._script_temp_dir)
    # endif not self.assets_initialized

    if not self.npm_init_started:
      # Init the nodejs server
      npm_init_args = [
        'npm',
        'init',
        '-y'
      ]

      npm_init_process = subprocess.Popen(
        npm_init_args,
        env=self.prepared_env,
        cwd=self._script_temp_dir
      )
      self.npm_init_started = True
    # endif not self.npm_init_started

    if not self.npm_init_finished:
      timeout = 10

      try:
        npm_init_process.wait(timeout)
        self.npm_init_finished = True
      except subprocess.TimeoutExpired:
        self.P("WARNING: npm init timed out. Continuing anyway.")
    # endif not self.npm_init_finished

    if not self.npm_install_started:
      # Install the nodejs server
      npm_install_args = [
        'npm',
        'install'
      ]

      npm_install_process = subprocess.Popen(
        npm_install_args,
        env=self.prepared_env,
        cwd=self._script_temp_dir
      )
      self.npm_install_started = True
    # endif not self.npm_install_started

    if not self.npm_install_finished:
      timeout = 10

      try:
        npm_install_process.wait(timeout)
        self.npm_install_finished = True
      except subprocess.TimeoutExpired:
        self.P("WARNING: npm install timed out. Continuing anyway.")
    # endif not self.npm_install_finished

    if not self.npm_started:
      # Set PWD and cwd to the folder containing the nodejs script and assets
      nodejs_args = [
        'npm',
        'start'
      ]
      self.nodejs_process = subprocess.Popen(
        nodejs_args,
        env=self.prepared_env,
        cwd=self._script_temp_dir
      )
      self.npm_started = True

    return

  def on_close(self):
    if not self.npm_started:
      self.P("Nodejs server was never started. Skipping teardown.")
      return

    # Teardown nodejs
    try:
      # FIXME: there must be a clean way to do this.
      self.P("Forcefully killing nodejs server")
      self.nodejs_process.kill()
      self.P("Killed nodejs server")
    except Exception as _:
      self.P('Could not kill nodejs server')

    # TODO remove temporary folder. For now it's useful
    # to keep around for debugging purposes.

    super(NodeJsWebAppPlugin, self).on_close()
    return
