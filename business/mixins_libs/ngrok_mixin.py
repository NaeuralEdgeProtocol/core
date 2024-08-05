import subprocess

__VER__ = '0.0.0.0'


class _NgrokMixinPlugin(object):
  class NgrokCT:
    NG_TOKEN = 'EE_NGROK_AUTH_TOKEN'
    # HTTP_GET = 'get'
    # HTTP_PUT = 'put'
    # HTTP_POST = 'post'

  """
  A plugin which exposes all of its methods marked with @endpoint through
  fastapi as http endpoints, and further tunnels traffic to this interface
  via ngrok.

  The @endpoint methods can be triggered via http requests on the web server
  and will be processed as part of the business plugin loop.
  """

  @property
  def __ng_token(self):
    return self.os_environ.get(_NgrokMixinPlugin.NgrokCT.NG_TOKEN, None)

  def __maybe_print_ngrok_logs(self):
    if self.__ngrok_log_reader is not None:
      logs = self.__ngrok_log_reader.get_next_characters()
      if len(logs) > 0:
        self.P(logs)

    if self.__ngrok_error_log_reader is not None:
      err_logs = self.__ngrok_error_log_reader.get_next_characters()
      if len(err_logs) > 0:
        self.P(err_logs, color='r')
    return

  def __maybe_read_and_stop_ngrok_log_readers(self):
    if self.__ngrok_log_reader is not None:
      self.__ngrok_log_reader.stop()
    if self.__ngrok_error_log_reader is not None:
      self.__ngrok_error_log_reader.stop()

    self.__maybe_print_ngrok_logs()

    self.__ngrok_log_reader = None
    self.__ngrok_error_log_reader = None
    return

  def __maybe_authenticate_ngrok(self):
    if self.__ngrok_authentication_finished:
      return

    if not self.__ngrok_authentication_started:
      self.__ngrok_authentication_started = True
      ngrok_auth_args = ['ngrok', 'authtoken', self.__ng_token]
      self.P('Trying to authenticate ngrok')
      self.__ngrok_auth_process, self.__ngrok_log_reader, self.__ngrok_error_log_reader = self.run_command(
        ngrok_auth_args)
      self.__ngrok_authentication_started_time = self.time()

    timeout = 0.1
    finished, failed = self.wait_for_command(
      self.__ngrok_auth_process,
      timeout,
      lst_log_reader=[self.__ngrok_log_reader, self.__ngrok_error_log_reader]
    )

    if finished and not failed:
      self.__maybe_read_and_stop_ngrok_log_readers()
      self.__ngrok_authentication_finished = True
      self.P('Successfully authenticated ngrok.', color='g')

    elif finished and failed:
      self.__maybe_read_and_stop_ngrok_log_readers()
      self.__ngrok_authentication_started = False
      raise RuntimeError('Could not authenticate ngrok. Please check your ngrok authentication token.')
    elif not finished:
      if self.time() - self.__ngrok_authentication_started_time > 3:
        self.__ngrok_auth_process.kill()
        self.__maybe_read_and_stop_ngrok_log_readers()
        self.__ngrok_authentication_started = False
        raise RuntimeError('Could not authenticate ngrok. Authentication process timed out.')
    return

  def __maybe_start_ngrok_tunnel(self):
    if self.__ngrok_tunnel_checked:
      # Tunnel is up and ready
      return

    if not self.__ngrok_authentication_finished:
      # Wait for authentication to finish before starting the tunnel.
      return

    if not self.__ngrok_tunnel_started:
      self.__ngrok_tunnel_started = True

      # Start ngrok in the background.
      self.P('Starting ngrok on port {} with domain {} and edge {}'.format(
        self.port, self.cfg_ngrok_domain, self.cfg_ngrok_edge_label))
      if self.cfg_ngrok_edge_label is not None:
        ngrok_start_args = [
          'ngrok',
          'tunnel',
          str(self.port),
          '--label',
          f'edge={self.cfg_ngrok_edge_label}'
        ]
      elif self.cfg_ngrok_domain is not None:
        ngrok_start_args = [
          'ngrok',
          'http',
          str(self.port),
          f'--domain={self.cfg_ngrok_domain}'
        ]
      else:
        self.__ngrok_tunnel_started = False
        raise RuntimeError("No domain/edge specified. Please check your configuration.")
      # endif

      self.__ngrok_tunnel_process, self.__ngrok_log_reader, self.__ngrok_error_log_reader = self.run_command(ngrok_start_args)
      self.__ngrok_tunnel_started_time = self.time()
      self.P('Starting ngrok process with PID {}'.format(self.__ngrok_tunnel_process.pid))
    # endif

    timeout = 0.1
    finished, _ = self.wait_for_command(
      self.__ngrok_tunnel_process,
      timeout,
      lst_log_reader=[self.__ngrok_log_reader, self.__ngrok_error_log_reader]
    )

    if finished:
      self.__maybe_read_and_stop_ngrok_log_readers()
      self.__ngrok_tunnel_started = False
      raise RuntimeError('Ngrok process finished unexpectedly. Please check the logs.')
    elif self.time() - self.__ngrok_tunnel_started_time > 3:
      self.P('ngrok started successfully.')
      self.__ngrok_tunnel_checked = True

    return

  def __maybe_close_ngrok_tunnel(self):
    if not self.__ngrok_tunnel_started:
      return

    try:
      self.P('Killing ngrok..')
      self.__ngrok_tunnel_process.kill()
      self.__maybe_read_and_stop_ngrok_log_readers()
      self.P('Killed ngrok')
    except Exception as _:
      self.P('Could not kill ngrok server')

    # FIXME: we assume that the process is killed, even when an exception is raised.
    self.__ngrok_tunnel_started = False
    return

  # Public API
  def on_init(self):
    super(_NgrokMixinPlugin, self).on_init()
    self.__cached_use_ngrok = self.cfg_use_ngrok or self.cfg_ngrok_enabled
    self.__ngrok_tunnel_process = None

    self.__ngrok_auth_process = None
    self.__ngrok_authentication_started = False
    self.__ngrok_authentication_started_time = 0
    self.__ngrok_authentication_finished = False

    self.__ngrok_tunnel_process = None
    self.__ngrok_tunnel_started = False
    self.__ngrok_tunnel_started_time = 0
    self.__ngrok_tunnel_checked = False

    self.__ngrok_log_reader = None
    self.__ngrok_error_log_reader = None
    return

  def _process(self):
    super(_NgrokMixinPlugin, self)._process()

    if self.cfg_use_ngrok or self.cfg_ngrok_enabled:
      self.__maybe_authenticate_ngrok()
      self.__maybe_start_ngrok_tunnel()
      self.__maybe_print_ngrok_logs()
    # endif
    return

  def _on_config(self):
    super(_NgrokMixinPlugin, self)._on_config()
    if self.__cached_use_ngrok and not (self.cfg_use_ngrok or self.cfg_ngrok_enabled):
      self.P("User disabled ngrok. Closing tunnel.")
      self.__maybe_close_ngrok_tunnel()
    # endif
    return

  def _on_close(self):
    super(_NgrokMixinPlugin, self)._on_close()
    self.__maybe_close_ngrok_tunnel()
    return
