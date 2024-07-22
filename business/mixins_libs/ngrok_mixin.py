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

  def on_init(self):
    self.cached_use_ngrok = self.cfg_use_ngrok
    self.ngrok_process = None

    self.ngrok_authentication_done = False
    self.ngrok_tunnel_started = False
    return

  @property
  def ng_token(self):
    return self.os_environ.get(_NgrokMixinPlugin.NgrokCT.NG_TOKEN, None)

  def _maybe_authenticate_ngrok(self):
    if self.ngrok_authentication_done:
      return

    ngrok_auth_args = ['ngrok', 'authtoken', self.ng_token]
    self.P('Trying to authenticate ngrok')
    auth_process = subprocess.Popen(ngrok_auth_args, stdout=subprocess.DEVNULL)

    # Wait for the process to finish and get the exit code
    try:
      if auth_process.wait(timeout=10) != 0:
        raise RuntimeError('Could not authenticate ngrok. Please check your ngrok authentication token.')
    except subprocess.TimeoutExpired:
      auth_process.kill()
      raise RuntimeError('Could not authenticate ngrok. Authentication process timed out.')

    self.ngrok_authentication_done = True
    self.P('Successfully authenticated ngrok.', color='g')
    return

  def _maybe_start_ngrok_tunnel(self):
    if self.ngrok_tunnel_started:
      return

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
      raise RuntimeError("No domain/edge specified. Please check your configuration.")
    # endif

    self.ngrok_process = subprocess.Popen(ngrok_start_args, stdout=subprocess.DEVNULL)
    # FIXME: this returns the pid, but what happens in case of error?

    self.P('Starting ngrok process with PID {}'.format(self.ngrok_process.pid))

    # Wait until ngrok has started.
    self.sleep(10)

    # FIXME: does this assumes linux (even if we run with docker)?
    if not self.os_path.exists('/proc/' + str(self.ngrok_process.pid)):
      raise RuntimeError('Failed to start ngrok')
    self.ngrok_tunnel_started = True
    self.P('ngrok started successfully.')

    return

  def _maybe_close_ngrok_tunnel(self):
    if not self.ngrok_tunnel_started:
      return

    try:
      self.P('Killing ngrok..')
      self.ngrok_process.kill()
      self.P('Killed ngrok')
    except Exception as _:
      self.P('Could not kill ngrok server')

    # FIXME: we assume that the process is killed, even when an exception is raised.
    self.ngrok_tunnel_started = False
    return

  def _process(self):
    super(_NgrokMixinPlugin, self)._process()

    if self.cfg_use_ngrok:
      self._maybe_authenticate_ngrok()
      self._maybe_start_ngrok_tunnel()
    # endif
    return

  def on_config(self):
    super(_NgrokMixinPlugin, self).on_config()
    if self.cached_use_ngrok and not self.cfg_use_ngrok:
      self.P("User disabled ngrok. Closing tunnel.")
      self._maybe_close_ngrok_tunnel()
    # endif
    return

  def on_close(self):
    super(_NgrokMixinPlugin, self).on_close()
    self._maybe_close_ngrok_tunnel()
    return
