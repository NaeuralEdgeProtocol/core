import subprocess

__VER__ = '0.0.0.0'

class _NgrokMixinPlugin(object):
  class NgrokCT:
    NG_TOKEN = 'EE_NGROK_AUTH_TOKEN'
    NG_DOMAIN = 'EE_NGROK_DOMAIN'
    NG_EDGE_LABEL = 'EE_NGROK_EDGE_LABEL'
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

  @property
  def ng_domain(self):
    return self.os_environ.get(_NgrokMixinPlugin.NgrokCT.NG_DOMAIN, None)

  @property
  def ng_edge(self):
    return self.os_environ.get(_NgrokMixinPlugin.NgrokCT.NG_EDGE_LABEL, None)

  def on_init(self):
    super(_NgrokMixinPlugin, self).on_init()
    if self.cfg_use_ngrok:
      port = self.cfg_port

      # Check the config as we're going to use it to start processes.
      if not isinstance(port, int):
        raise ValueError("Ngrok port not an int")
      if port < 0 or port > 65535:
        raise ValueError("Invalid port value {}".format(port))

      ngrok_auth_args = ['ngrok', 'authtoken', self.ng_token]
      self.P('Trying to authenticate ngrok')
      auth_process = subprocess.Popen(ngrok_auth_args, stdout=subprocess.DEVNULL)

      # Wait for the process to finish and get the exit code
      if auth_process.wait(timeout=10) != 0:
        raise RuntimeError('Could not authenticate ngrok')
      self.P('Successfully authenticated ngrok.', color='g')

      # Start ngrok in the background.
      self.P('Starting ngrok on port {} with domain {}'.format(port, self.ng_domain))
      if self.ng_edge is not None:
        # A domain was specified in the env, just start on that domain.
        edge_label = self.ng_edge
        ngrok_start_args = [
          'ngrok',
          'tunnel',
          str(port),
          '--label',
          f'edge={edge_label}'
        ]
      elif self.ng_domain is not None:
        ngrok_start_args = [
          'ngrok',
          'http',
          str(port),
          '--domain=' + self.ng_domain
        ]
      else:
        raise RuntimeError("No domain/edge specified")
      #endif

      self.ngrok_process = subprocess.Popen(ngrok_start_args, stdout=subprocess.DEVNULL)
      #FIXME: this returns the pid, but what happens in case of error?
      self.P('ngrok started with PID {}'.format(self.ngrok_process.pid), color='g')

      # Wait until ngrok has started.
      self.P('Waiting for ngrok to start...', color='b')
      self.sleep(10)

      # FIXME: does this assumes linux (even if we run with docker)?
      if not self.os_path.exists('/proc/' + str(self.ngrok_process.pid)):
        raise RuntimeError('Failed to start ngrok')
      self.P('ngrok started successfully."', color='g')

      # Start the app
      self.P('Starting app...')
      # TODO: add method to implement in child classes

    return

  def on_close(self):
    super(_NgrokMixinPlugin, self).on_close()
    if self.cfg_use_ngrok:
      # Teardown ngrok
      try:
        self.P('Killing ngrok..')
        self.ngrok_process.kill()
        self.P('Killed ngrok')
      except Exception as _:
        self.P('Could not kill ngrok server')

    return
