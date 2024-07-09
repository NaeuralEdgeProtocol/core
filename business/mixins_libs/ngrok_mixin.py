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

  def __init__(self, **kwargs):
    self.ngrok_process = None
    super(_NgrokMixinPlugin, self).__init__(**kwargs)
    return

  @property
  def ng_token(self):
    return self.os_environ.get(_NgrokMixinPlugin.NgrokCT.NG_TOKEN, None)

  def _check_port_valid(self):
    # Check the config as we're going to use it to start processes.
    if not isinstance(self.cfg_port, int):
      raise ValueError("Ngrok port not an int")
    if self.cfg_port < 0 or self.cfg_port > 65535:
      raise ValueError("Invalid port value {}".format(self.cfg_port))
    return

  def _authenticate_ngrok(self):
    ngrok_auth_args = ['ngrok', 'authtoken', self.ng_token]
    self.P('Trying to authenticate ngrok')
    auth_process = subprocess.Popen(ngrok_auth_args, stdout=subprocess.DEVNULL)

    # Wait for the process to finish and get the exit code
    if auth_process.wait(timeout=10) != 0:
      raise RuntimeError('Could not authenticate ngrok')
    self.P('Successfully authenticated ngrok.', color='g')
    return

  def _start_ngrok_tunnel(self):
    # Start ngrok in the background.
    self.P('Starting ngrok on port {} with domain {}'.format(self.cfg_port, self.cfg_ngrok_domain))
    if self.cfg_ngrok_edge_label is not None:
      ngrok_start_args = [
        'ngrok',
        'tunnel',
        str(self.cfg_port),
        '--label',
        f'edge={self.cfg_ngrok_edge_label}'
      ]
    elif self.cfg_ngrok_domain is not None:
      ngrok_start_args = [
        'ngrok',
        'http',
        str(self.cfg_port),
        f'--domain={self.cfg_ngrok_domain}'
      ]
    else:
      raise RuntimeError("No domain/edge specified")
    #endif

    self.ngrok_process = subprocess.Popen(ngrok_start_args, stdout=subprocess.DEVNULL)
    #FIXME: this returns the pid, but what happens in case of error?
    self.P('ngrok started with PID {}'.format(self.ngrok_process.pid), color='g')
    return

  def _check_ngrok_started_successfully(self):
    # Wait until ngrok has started.
    self.P('Waiting for ngrok to start...', color='b')
    self.sleep(10)

    # FIXME: does this assumes linux (even if we run with docker)?
    if not self.os_path.exists('/proc/' + str(self.ngrok_process.pid)):
      raise RuntimeError('Failed to start ngrok')
    self.P('ngrok started successfully."', color='g')

    return

  def on_init(self):
    super(_NgrokMixinPlugin, self).on_init()

    if not self.cfg_use_ngrok:
      return

    self._check_port_valid()

    self._authenticate_ngrok()

    self._start_ngrok_tunnel()

    self._check_ngrok_started_successfully()

    self.P('Starting app...')
    return

  def on_close(self):
    super(_NgrokMixinPlugin, self).on_close()
    if not self.cfg_use_ngrok:
      return

    # Teardown ngrok
    try:
      self.P('Killing ngrok..')
      self.ngrok_process.kill()
      self.P('Killed ngrok')
    except Exception as _:
      self.P('Could not kill ngrok server')

    return
