#global dependencies
from datetime import datetime as dt

#local dependencies
from core import constants as ct
from core.business.base import BasePluginExecutor
from minio import Minio

__VER__ = '0.2.1'

_CONFIG = {
  **BasePluginExecutor.CONFIG,
  'MINIO_HOST'          : None,
  'MINIO_ACCESS_KEY'    : None,
  'MINIO_SECRET_KEY'    : None,
  'MINIO_SECURE'        : None,
  
  'ENV_HOST'            : 'EE_MINIO_ENDPOINT',
  'ENV_ACCESS_KEY'      : 'EE_MINIO_ACCESS_KEY',
  'ENV_SECRET_KEY'      : 'EE_MINIO_SECRET_KEY',
  'ENV_SECURE'          : 'EE_MINIO_SECURE',

  'ALLOW_EMPTY_INPUTS'  : True,
  
  'PROCESS_DELAY'       : 5,
  
  'MAX_FILES_PER_ITER'  : 2_000,

  'ALERT_DATA_COUNT'    : 1,
  'ALERT_RAISE_VALUE'   : 0.8,
  'ALERT_LOWER_VALUE'   : 0.75,
  'ALERT_MODE'          : 'min',

  'MAX_SERVER_QUOTA'    : 20,
  
  'QUOTA_UNIT'          : ct.FILE_SIZE_UNIT.GB,
  
  'DEBUG_MODE'          : False,
  
  'MIN_TIME_BETWEEN_PAYLOADS' : 60 * 5, # 5 minutes

  'VALIDATION_RULES': {
    **BasePluginExecutor.CONFIG['VALIDATION_RULES'],
  },
}

class MinioMonit01Plugin(BasePluginExecutor):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(MinioMonit01Plugin, self).__init__(**kwargs)
    return

  
  def on_init(self):
    if self.is_supervisor_node:
      self.P("MinioMonit01Plugin initializing on SUPERVISOR node...")
    else:
      self.P("MinioMonit01Plugin initializing on simple worker node...")
    self.__global_iter = 0
    self.__last_display = 0
    self.__default_host = None
    self.__default_access_key = None
    self.__default_secret_key = None    
    self.__default_secure = None  
    self.__minio_client = None
    self.__buckets_list = []
    self.__current_bucket_objects_generator = None
    self.__last_minio_payload_time = 0
    self.__reset_size()
    return

  
  def __reset_size(self):
    self.__server_size = 0
    self.__bucket_size = {}
    self.__current_bucket_no = 0
    return
  
  
  def __maybe_get_env_config(self):
    if self.is_supervisor_node and self.__default_host is None:
      self.__default_host = self.os_environ.get(self.cfg_env_host)
      self.__default_access_key = self.os_environ.get(self.cfg_env_access_key)
      self.__default_secret_key = self.os_environ.get(self.cfg_env_secret_key)
      self.__default_secure = self.json_loads(self.os_environ.get(self.cfg_env_secure))
      self.P("Detected supervisor node, using environment variables for Minio connection...")
      self.P("  MINIO_HOST: {}, SECURE: {}".format(self.__default_host, self.__default_secure))
    #endif
    return
  
  
  def __get_connection_config(self):
    self.__maybe_get_env_config()
    host = self.cfg_minio_host or self.__default_host
    access_key = self.cfg_minio_access_key or self.__default_access_key
    secret_key = self.cfg_minio_secret_key or self.__default_secret_key
    secured = self.cfg_minio_secure or self.__default_secure
    return host, access_key, secret_key, secured
  
      

  def __get_next_bucket(self):
    if len(self.__buckets_list) == 0:
      # reset size and get new buckets
      self.__reset_size()
      self.__buckets_list = self.__minio_client.list_buckets()  
      for bucket in self.__buckets_list:
        self.__bucket_size[bucket.name] = {
          'size': 0,
          'objects': 0,
        }        
      str_buckets = ", ".join(map(lambda x: x.name, self.__buckets_list))    
      DISPLAY_EVERY = 15 * 60 
      if (self.time() - self.__last_display) > DISPLAY_EVERY:
        self.__last_display = self.time()
        self.P("Analysing {} (secured: {}). Iterating through {} buckets: {}".format(
          self.__host, self.__secured,
          len(self.__buckets_list), 
          str_buckets,
          )
        )
    self.__current_bucket_no += 1
    return self.__buckets_list.pop()


  def __maybe_get_new_objects_generator(self):
    if self.__current_bucket_objects_generator is None:
      self.__current_bucket = self.__get_next_bucket()
      self.__current_bucket_objects_generator = self.__minio_client.list_objects(
        bucket_name=self.__current_bucket.name, 
        recursive=True,
      )
      if self.cfg_debug_mode:
        self.P("Iteration {}/{} through bucket '{}'".format(
          self.__current_bucket_no, len(self.__bucket_size),
          self.__current_bucket.name, 
          )
        )
    return


  def __maybe_create_client_connection(self):
    host, access_key, secret_key, secured = self.__get_connection_config()    
    self.__host = host
    self.__secured = secured
    if (
      self.__minio_client is None and
      host is not None and
      access_key is not None and
      secret_key is not None and
      secured is not None
      ):
        self.P("Creating Minio client connection at iteration {} to {}...".format(self.__global_iter, self.cfg_minio_host))
        self.__minio_client = Minio(
          endpoint=host,
          access_key=access_key,
          secret_key=secret_key,
          secure=secured,
        )
    else:
      if self.__minio_client is None and self.__global_iter < 2:
        self.P(f"Missing Minio connection parameters (is_supervisor_node: {self.is_supervisor_node}, host: {host}, access_key: {access_key}, secret_key: {secret_key}, secure: {secured})", color='r')


  def __process_iter_files(self):
    start_time = self.time()
    local_count = 0
    for _ in range(self.cfg_max_files_per_iter):
      try:
        obj = next(self.__current_bucket_objects_generator)
        self.__server_size += obj.size
        local_count += 1
        self.__bucket_size[self.__current_bucket.name]['size'] += obj.size
        self.__bucket_size[self.__current_bucket.name]['objects'] += 1
      except StopIteration:
        self.__current_bucket_objects_generator = None
        break
    elapsed_time = self.time() - start_time + 1e-10
    if self.cfg_debug_mode and local_count > 0:
      self.P("Processed {} objects in {:.1f}s, {:.0f} files/s".format(
        local_count, elapsed_time, local_count / elapsed_time)
      )
      n_o = self.__bucket_size[self.__current_bucket.name]['objects']
      cut = self.cfg_max_files_per_iter * 3
      if n_o > 0 and (n_o % cut) == 0:
        self.P("  Bucket '{}' size: {:.2f} {} (so far for {} objs)".format(
          self.__current_bucket.name,
          self.convert_size(self.__bucket_size[self.__current_bucket.name]['size'], self.cfg_quota_unit),
          self.cfg_quota_unit,
          self.__bucket_size[self.__current_bucket.name]['objects'],
          )
        ) 
    return


  def _process(self):
    self.__global_iter += 1
    payload = None

    self.__maybe_create_client_connection()
    if self.__minio_client is None:
      return

    self.__maybe_get_new_objects_generator()

    self.__process_iter_files()

    # when the plugin iterated through all buckets send payload
    if self.__current_bucket_objects_generator is None and len(self.__buckets_list) == 0:
      converted_size = self.convert_size(self.__server_size, self.cfg_quota_unit)
      percentage_used = converted_size / self.cfg_max_server_quota

      self.alerter_add_observation(percentage_used)
      
      for b in self.__bucket_size:
        self.__bucket_size[b]['size_h'] = "{:,.2f} {}".format(
          self.convert_size(self.__bucket_size[b]['size'], self.cfg_quota_unit),
          self.cfg_quota_unit,
        )
      
      
      if (self.time() - self.__last_minio_payload_time) > self.cfg_min_time_between_payloads:        
        self.__last_minio_payload_time = self.time()
        msg = "Server size: {:.1f} {} of configured quota {:.1f} {} ({:.1f} %)".format(
          converted_size, self.cfg_quota_unit, self.cfg_max_server_quota, self.cfg_quota_unit,
          percentage_used * 100,
        )
        color = 'r' if self.alerter_is_alert() else None
        # only log-show detailed info in debug mode
        if self.cfg_debug_mode:
          self.P("{}:\n{}".format(
            msg, self.json_dumps(self.__bucket_size, indent=2),
            ), color=color
          )
        else:
          self.P(msg, color=color)
        # alerts are handled automatically by above code: adding used percentage
        payload = self._create_payload(
          server_size=self.__server_size,
          buckets=self.__bucket_size,
          status=msg,
        )
      #endif time check
    return payload