import json
import numpy as np
import pandas as pd
import cv2
import PIL
import requests
import uuid
import os
import sys
import traceback
import inspect
import re
import base64
import zlib


from collections import OrderedDict, defaultdict, deque
from io import BytesIO
from time import sleep, time
from datetime import datetime, timedelta
from copy import deepcopy
from xml.etree import ElementTree
from urllib.parse import urlparse
from functools import partial

from core import constants as ct

from core.serving.ai_engines.utils import (
  get_serving_process_given_ai_engine,
  get_ai_engine_given_serving_process,
  get_params_given_ai_engine
)

from core.utils.plugins_base.persistence_serialization_mixin import _PersistenceSerializationMixin
from core.utils.system_shared_memory import NumpySharedMemory

class NestedDotDict(dict):
  # TODO: maybe use https://github.com/mewwts/addict/blob/master/addict/addict.py
  __getattr__ = defaultdict.__getitem__
  __setattr__ = defaultdict.__setitem__
  __delattr__ = defaultdict.__delitem__

  def __init__(self, *args, **kwargs):
    super(NestedDotDict, self).__init__(*args, **kwargs)
    for key, value in self.items():
      if isinstance(value, dict):
        self[key] = NestedDotDict(value)
      elif isinstance(value, (list, tuple)):
        self[key] = type(value)(
          NestedDotDict(v) if isinstance(v, dict) else v for v in value
        )        
              
  def __deepcopy__(self, memo):
    return NestedDotDict({k: deepcopy(v, memo) for k, v in self.items()})
                
  def __reduce__(self):
    return (self.__class__, (), self.__getstate__())

  def __getstate__(self, obj=None):
    state = {}
    obj = obj or self
    for key, value in obj.items():
      if isinstance(value, NestedDotDict):
        state[key] = self.__getstate__(value)
      else:
        state[key] = value
    return state

  def __setstate__(self, state):
    self.update(state)
  
class DefaultDotDict(defaultdict):
  __getattr__ = defaultdict.__getitem__
  __setattr__ = defaultdict.__setitem__
  __delattr__ = defaultdict.__delitem__
  

class NestedDefaultDotDict(defaultdict):
  """
  A dictionary-like object supporting auto-creation of nested dictionaries and default values for undefined keys.
  """
  def __init__(self, *args, **kwargs):
    super(NestedDefaultDotDict, self).__init__(NestedDefaultDotDict, *args, **kwargs)
    for key, value in dict(self).items():
      if isinstance(value, dict):
        self[key] = NestedDefaultDotDict(value)
      elif isinstance(value, (list, tuple)):
        self[key] = type(value)(
          NestedDefaultDotDict(v) if isinstance(v, dict) else v for v in value
        )

  def __getattr__(self, item):
    if item in self:
      return self[item]
    return self.__missing__(item)

  def __setattr__(self, key, value):
    if isinstance(value, dict) and not isinstance(value, NestedDefaultDotDict):
      value = NestedDefaultDotDict(value)
    defaultdict.__setitem__(self, key, value)

  def __delattr__(self, item):
    try:
      defaultdict.__delitem__(self, item)
    except KeyError as e:
      raise AttributeError(e)

  def __deepcopy__(self, memo):
    return NestedDefaultDotDict({k: deepcopy(v, memo) for k, v in self.items()})

  def __reduce__(self):
    return (self.__class__, (), None, None, iter(self.items()))


class NPJson(json.JSONEncoder):
  """
  Used to help jsonify numpy arrays or lists that contain numpy data types.
  """
  def default(self, obj):
      if isinstance(obj, np.integer):
          return int(obj)
      elif isinstance(obj, np.floating):
          return float(obj)
      elif isinstance(obj, np.ndarray):
          return obj.tolist()
      elif isinstance(obj, np.ndarray):
          return obj.tolist()
      elif isinstance(obj, datetime):
          return obj.strftime("%Y-%m-%d %H:%M:%S")
      else:
          return super(NPJson, self).default(obj)

class _UtilsBaseMixin(
  _PersistenceSerializationMixin
  ):

  def __init__(self):
    super(_UtilsBaseMixin, self).__init__()
    return

  
  def trace_info(self):
    """
    Returns a multi-line string with the last exception stacktrace (if any)

    Returns
    -------
    str.

    """
    return traceback.format_exc()
  
  
  def python_version(self):
    """
    Utilitary method for accessing the Python version.
    Returns
    -------
    Version of python
    """
    return sys.version.split()[0]
  
  def get_serving_process_given_ai_engine(self, ai_engine):
    return get_serving_process_given_ai_engine(ai_engine)
  
    

  def timedelta(self, **kwargs):
    """
    Alias of `datetime.timedelta`
    

    Parameters
    ----------
    **kwargs : 
      can contain days, seconds, microseconds, milliseconds, minutes, hours, weeks.


    Returns
    -------
    timedelta object


    Example
    -------
      ```
        diff = self.timedelta(seconds=10)
      ```
    
    """
    return timedelta(**kwargs)  
  
  
  def time(self):
    """
    Returns current timestamp

    Returns
    -------
    time : timestamp (float)
      current timestamp.
      
      
    Example
    -------
      ```
      t1 = self.time()
      ... # do some stuff
      elapsed = self.time() - t1
      ```    

    """

    return time() 

  def now_str(self, nice_print=False, short=False):
    """
    Returns current timestamp in string format
    Parameters
    ----------
    nice_print
    short

    Returns
    -------

    """
    return self.log.now_str(nice_print=nice_print, short=short)

  def get_output_folder(self):
    """
    Provides access to get_output_folder() method from .log
    Returns
    -------

    """
    return self.log.get_output_folder()

  def get_data_folder(self):
    """
    Provides access to get_data_folder() method from .log
    Returns
    -------

    """
    return self.log.get_data_folder()

  def get_logs_folder(self):
    """
    Provides access to get_logs_folder() method from .log
    Returns
    -------

    """
    return self.log.get_logs_folder()

  def get_models_folder(self):
    """
    Provides access to get_models_folder() method from .log
    Returns
    -------

    """
    return self.log.get_models_folder()

  def get_target_folder(self, target):
    """
    Provides access to get_target_folder() method from .log
    Parameters
    ----------
    target

    Returns
    -------

    """
    return self.log.get_target_folder(target)

  def sleep(self, seconds):
    """
    sleeps current job a number of seconds
    """
    sleep(seconds)
    return  


  def uuid(self, size=13):
    """
    Returns a unique id.
  

    Parameters
    ----------
    size : int, optional
      the number of chars in the uid. The default is 13.

    Returns
    -------
    str
      the uid.
      

    Example
    -------
    
      ```
        str_uid = self.uuid()
        result = {'generated' : str_uid}
      ```      

    """
    return str(uuid.uuid4())[:13].replace('-','')
  
  @property
  def json(self):
    """
    Provides access to `json` package

    Returns
    -------
    `json` package      

    """
    return json

  @property
  def re(self):
    """
    Provides access to `re` package

    Returns
    -------
    `re` package

    """
    return re
  
  @property
  def inspect(self):
    """
    Provides access to `inspect` package

    Returns
    -------
    `inspect` package      

    """
    return inspect
    
  
  @property
  def requests(self):
    """
    Provides access to `requests` package

    Returns
    -------
    `requests` package      

    """
    return requests

  @property
  def urlparse(self):
    """
    Provides access to `urlparse` method from `urllib.parse` package

    Returns
    -------
    `urlparse` method      

    """
    return urlparse
  
  @property
  def consts(self):
    """
    Provides access to E2 constants

    Returns
    -------
    ct : package
      Use `self.consts.CONST_ACME` to acces any required constant

    """
    return ct


  @property
  def const(self):
    """
    Provides access to E2 constants

    Returns
    -------
    ct : package
      Use `self.const.ct.CONST_ACME` to acces any required constant

    """
    return ct

  @property
  def ct(self):
    """
    Provides access to E2 constants

    Returns
    -------
    ct : package
      Use `self.const.ct.CONST_ACME` to acces any required constant

    """
    return ct

  @property
  def ds_consts(self):
    """
    Alias for DatasetBuilder class from E2 constants
    Provides access to constants used in DatasetBuilderMixin
    Returns
    -------
    ct.DatasetBuilder : package
      Use `self.ds_consts.CONST_ACME` to access any required constant
    """
    return ct.DatasetBuilder

  @property
  def cv2(self):
    """
    provides access to computer vision library
    """
    return cv2

  @property
  def np(self):
    """
    Provides access to numerical processing library
    

    Returns
    -------
    np : Numpy package
      
    Example:
      ```
      np_zeros = self.np.zeros(shape=(10,10))
      ```
    """
    return np  
  
  @property
  def OrderedDict(self):
    """
    Returns the definition for `OrderedDict`

    Returns
    -------
    OrderedDict : class
      `OrderedDict` from standard python `collections` package.
      
    Example
    -------
        ```
        dct_A = self.OrderedDict({'a': 1})
        dct_A['b'] = 2
        ```

    """
    return OrderedDict  
  
  
  @property
  def defaultdict(self):
    """
    provides access to defaultdict class


    Returns
    -------
      defaultdict : class
      
    Example
    -------
      ```
        dct_integers = self.defaultdict(lambda: 0)
      ```

    """
    return defaultdict
  
  
  def DefaultDotDict(self, *args):
    """
    Returns a `DefaultDotDict` object that is a `dict` where you can use keys with dot 
    using the default initialization
    
    Inputs
    ------
    
      pass a `lambda: <type>` always
    
    Returns
    -------
      DefaultDotDict : class
     
    Example
    -------
     ```
       dct_dot = self.DefaultDotDict(lambda: str)
       dct_dot.test1 = "test"       
       print(dct_dot.test1)
       print(dct_dot.test2)
     ```

    """
    return DefaultDotDict(*args)
  
  def NestedDotDict(self, *args):
    """
    Returns a `NestedDotDict` object that is a `dict` where you can use keys with dot
    
    Returns
    -------
      defaultdict : class
     
    Example
    -------
     ```
       dct_dot = self.NestedDotDict({'test' : {'a' : 100}})
       dct_dot.test.a = "test"   
       print(dct_dot.test.a)
    """
    return NestedDotDict(*args)
  
  
  def NestedDefaultDotDict(self, *args):
    """
    Returns a `NestedDefaultDotDict` object that is a `defaultdict(dict)` where you can use keys with dot
    
    Returns
    -------
      defaultdict : class
     
    Example
    -------
     ```
      dct_dot1 = self.NestedDefaultDotDict()
      dct_dot1.test.a = "test"   
      print(dct_dot1.test.a)
       
      dct_dot2 = self.NestedDefaultDotDict({'test' : {'a' : 100, 'b' : {'c' : 200}}})
      print(dct_dot2.test.a)
      print(dct_dot2.test.b.c)
      print(dct_dot2.test.b.unk)
        
    """
    return NestedDefaultDotDict(*args)
    
    

  def path_exists(self, path):
    """
    TODO: must be reviewed
    """
    return self.os_path.exists(path)
  
  
  @property
  def deque(self):
    """
    provides access to deque class
    """
    return deque  
  
  @property
  def datetime(self):
    """
    Proxy for the `datetime.datetime`

    Returns
    -------
      datetime : datetime object
      
      
    Example
    -------
      ```
      now = self.datetime.now()
      ```

    """
    return datetime

  @property
  def deepcopy(self):
    """
    This method allows us to use the method deepcopy
    """
    return deepcopy
  
  @property
  def os_path(self):
    """
    Proxy for `os.path` package


    Returns
    -------
      package
      
      
    Example
    -------
      ```
      fn = self.diskapi_save_dataframe_to_data(df, 'test.csv')
      exists = self.os_path.exists(fn)
      ```

    """
    return os.path
  
  @property
  def os_environ(self):
    """
    Returns a copy of the current environment variables based on `os.environ`.
    Important: Changing a value in the returned dictionary does NOT change 
               the value of the actual environment variable.
    

    Returns
    -------
    _type_
        _description_
    """
    return os.environ.copy()

  @property
  def PIL(self):
    """
    provides access to PIL package
    """
    return PIL

  @property
  def BytesIO(self):
    """
    provides access to BytesIO class from io package
    """
    return BytesIO

  @property
  def ElementTree(self):
    """
    provides access to ElementTree class from xml.etree package
    """
    return ElementTree

  @property
  def pd(self):
    """
    Provides access to pandas library

    Returns
    -------
      package
      
      
    Example
    -------
      ```
      df = self.pd.DataFrame({'a' : [1,2,3], 'b':[0,0,1]})      
      ```

    """
    return pd  

  @property
  def partial(self):
    """
    Provides access to `functools.partial` method

    Returns
    -------
      method


    Example
    -------
      ```
      fn = self.partial(self.diskapi_save_dataframe_to_data, fn='test.csv')
      ```

    """
    return partial

  def safe_json_dumps(self, dct, replace_nan=False, **kwargs):
    """Safe json dumps that can handle numpy arrays and so on

    Parameters
    ----------
    dct : dict
        The dict to be dumped
        
    replace_nan : bool, optional
        Replaces nan values with None. The default is False.

    Returns
    -------
    str
        The json string
    """
    return self.log.safe_json_dumps(dct, replace_nan=replace_nan, **kwargs)

  
  def json_dumps(self, dct, replace_nan=False, **kwargs):
    """Alias for `safe_json_dumps` for backward compatibility
    """
    return self.safe_json_dumps(dct, replace_nan=replace_nan, **kwargs)
  
  def json_loads(self, json_str, **kwargs):
    """
    Parses a json string and returns the dictionary
    """
    return self.json.loads(json_str, **kwargs)
  
  
  def load_config_file(self, fn):
    """
    Loads a json/yaml config file and returns the config dictionary

    Parameters
    ----------
    fn : str
      The filename of the config file

    Returns
    -------
    dict
      The config dictionary
    """
    return self.log.load_config_file(fn=fn)
  
  
  def maybe_download(self, url, fn, target='output', **kwargs):
    """
    Enables http/htps/minio download capabilities.


    Parameters
    ----------
    url : str or list
      The URI or URIs to be used for downloads
      
    fn: str of list
      The filename or filenames to be locally used
      
    target: str
      Can be `output`, `models` or `data`. Default is `output`

    kwargs: dict
      if url starts with 'minio:' the function will retrieve minio conn
             params from **kwargs and use minio_download (if needed or forced)

    Returns
    -------
      files, messages : list, list
        all the local files and result messages from download process
      
      
    Example
    -------
    """
    res = None
    files, msgs = self.log.maybe_download(
      url=url,
      fn=fn,
      target=target,
      **kwargs,
    )
    if len(files) >= 1:
      if len(files) == 1:
        res = files[0]
      else:
        res = files
    else:
      self.P('Errors while downloading: {}'.format([str(x) for x in msgs]))
    return res
  
  
  def dict_to_str(self, dct:dict):
    """
    Transforms a dict into a pre-formatted strig without json package

    Parameters
    ----------
    dct : dict
      The given dict that will be string formatted.

    Returns
    -------
    str
      the nicely formatted.
      
      
    Example:
    -------
      ```
      dct = {
        'a' : {
          'a1' : [1,2,3]
        },
        'b' : 'abc'
      }
      
      str_nice_dict = self.dict_to_str(dct=dct)
      ```

    """
    return self.log.dict_pretty_format(dct)  
  
  def timestamp_to_str(self, ts=None, fmt='%Y-%m-%d %H:%M:%S'):
    """
    Returns the string representation of current time or of a given timestamp


    Parameters
    ----------
    ts : float, optional
      timestamp. The default is None and will generate string for current timestamp. 
    fmt : str, optional
      format. The default is '%Y-%m-%d %H:%M:%S'.


    Returns
    -------
    str
      the timestamp in string format.
      
    
    Example
    -------
        
      ```
      t1 = self.time()
      ...
      str_t1 = self.time_to_str(t1)
      result = {'T1' : str_t1}
      ```
    """
    if ts is None:
      ts = self.time()
    return self.log.time_to_str(t=ts, fmt=fmt)
  
  
  def time_to_str(self, ts=None, fmt='%Y-%m-%d %H:%M:%S'):
    """
    Alias for `timestamp_to_str`
    

    Parameters
    ----------
    ts : float, optional
      The given time. The default is None.
    fmt : str, optional
      The time format. The default is '%Y-%m-%d %H:%M:%S'.

    Returns
    -------
    str
      the string formatted time.
      
      
    Example
    -------
      ```
      t1 = self.time()
      ...
      str_t1 = self.time_to_str(t1)
      result = {'T1' : str_t1}
      ```

    """
    return self.timestamp_to_str(ts=ts, fmt=fmt)
  
  
  def datetime_to_str(self, dt=None, fmt='%Y-%m-%d %H:%M:%S'):
    """
    Returns the string representation of current datetime or of a given datetime

    Parameters
    ----------
    dt : datetime, optional
      a given datetime. The default is `None` and will generate string for current date.
    fmt : str, optional
      datetime format. The default is '%Y-%m-%d %H:%M:%S'.

    Returns
    -------
    str
      the datetime in string format.
      
    
    Example
    -------
      ```
      d1 = self.datetime()
      ...
      str_d1 = self.datetime_to_str(d1)
      result = {'D1' : str_d1}
      ```
    

    """
    if dt is None:
      dt = datetime.now()
    return datetime.strftime(dt, format=fmt)

  def time_in_interval_hours(self, ts, start, end):
    """
    Provides access to method `time_in_interval_hours` from .log
    Parameters
    ----------
    ts: datetime timestamp
    start = 'hh:mm'
    end = 'hh:mm'

    Returns
    -------

    """
    return self.log.time_in_interval_hours(ts, start, end)

  def time_in_schedule(self, ts, schedule, weekdays=None):
    """
    Check if a given timestamp `ts` is in a active schedule given the schedule data


    Parameters
    ----------
    ts : float
      the given timestamp.
      
    schedule : dict or list
      the schedule.
            
    weekdays : TYPE, optional
      list of weekdays. The default is None.


    Returns
    -------
    bool
      Returns true if time in schedule.
      

    Example
    -------
      ```
      simple_schedule = [["09:00", "12:00"], ["13:00", "17:00"]]
      is_working = self.time_in_schedule(self.time(), schedule=simple_schedule)
      ```

    """
    return self.log.time_in_schedule(
      ts=ts,
      schedule=schedule,
      weekdays=weekdays
    )
    
    
  


  def now_in_schedule(self, schedule, weekdays=None):
    """
    Check if the current time is in a active schedule given the schedule data


    Parameters
    ----------
    schedule : dict or list
      the schedule.
            
    weekdays : TYPE, optional
      list of weekdays. The default is None.


    Returns
    -------
    bool
      Returns true if time in schedule.
      

    Example
    -------
      ```
      simple_schedule = [["09:00", "12:00"], ["13:00", "17:00"]]
      is_working = self.now_in_schedule(schedule=simple_schedule)
      ```

    """
    return self.log.now_in_schedule(
      schedule=schedule,
      weekdays=weekdays
    )  
    
    
  def img_to_base64(self, img):
    """Transforms a numpy image into a base64 encoded image

    Parameters
    ----------
    img : np.ndarray
        the input image

    Returns
    -------
    str: base64 encoded image
    """
    return self.log.np_image_to_base64(img)
  
  

  def str_to_base64(self, txt, compress=False):
    """Transforms a string into a base64 encoded string

    Parameters
    ----------
    txt : str
        the input string
        
    compress : bool, optional
        if True, the string will be compressed before encoding. The default is False.

    Returns
    -------
    str: base64 encoded string
    """
    b_text = bytes(txt, 'utf-8')    
    if compress:
      b_code = zlib.compress(b_text, level=9)
    else:
      b_code = b_text
    b_encoded = base64.b64encode(b_code)
    str_encoded = b_encoded.decode('utf-8')
    return str_encoded
  
  
  def base64_to_str(self, b64, decompress=False):
    """Transforms a base64 encoded string into a normal string

    Parameters
    ----------
    b64 : str
        the base64 encoded string
        
    decompress : bool, optional
        if True, the string will be decompressed after decoding. The default is False.

    Returns
    -------
    str: the decoded string
    """
    b_encoded = b64.encode('utf-8')
    b_text = base64.b64decode(b_encoded)
    if decompress:
      b_text = zlib.decompress(b_text)
    str_text = b_text.decode('utf-8')
    return str_text
  
  


  def normalize_text(self, text):
    """
    Uses unidecode to normalize text. Requires unidecode package

    Parameters
    ----------
    text : str
      the proposed text with diacritics and so on.

    Returns
    -------
    text : str
      decoded text if unidecode was avail



    Example
    -------
      ```
      str_txt = "Ha ha ha, m\u0103 bucur c\u0103 ai \u00eentrebat!"
      str_simple = self.normalize_text(str_text)
      ```


    """
    text = text.replace('\t', '  ')
    try:
      from unidecode import unidecode
      text = unidecode(text)
    except:
      pass
    return text  
  
  
  def sanitize_name(self, name: str)->str:
    """
    Returns a sanitized name that can be used as a variable name

    Parameters
    ----------
    name : str
        the proposed name

    Returns
    -------
    str
        the sanitized name
    """
    return re.sub(r'[^\w\.-]', '_', name)
  
  def convert_size(self, size, unit):
    """
    Given a size and a unit, it returns the size in the given unit

    Parameters
    ----------
    size : int
        value to be converted
    unit : str
        one of the following: 'KB', 'MB', 'GB'

    Returns
    -------
    _type_
        _description_
    """
    new_size = size
    if unit == ct.FILE_SIZE_UNIT.KB:
      new_size = size / 1024
    elif unit == ct.FILE_SIZE_UNIT.MB:
      new_size = size / 1024**2
    elif unit == ct.FILE_SIZE_UNIT.GB:
      new_size = size / 1024**3
    return new_size  
  
  def lock_resource(self, str_res):
    """
    Locks a resource given a string. Alias to `self.log.lock_resource`

    Parameters
    ----------
    str_res : str
        the resource name
    """
    return self.log.lock_resource(str_res)

  def unlock_resource(self, str_res):
    """
    Unlocks a resource given a string. Alias to `self.log.unlock_resource`

    Parameters
    ----------
    str_res : str
        the resource name
    """
    return self.log.unlock_resource(str_res)

  def create_numpy_shared_memory_object(self, mem_name, mem_size, np_shape, np_type, create=False, is_buffer=False, **kwargs):
    """
    Create a shared memory for numpy arrays. 
    This method returns a `NumpySharedMemory` object that can be used to read/write numpy arrays from/to shared memory.
    Use this method instead of creating the object directly, as it requires the logger to be set.

    For a complete set of parameters, check the `NumpySharedMemory` class from `core.utils.system_shared_memory`

    Parameters
    ----------
    mem_name : str
        the name of the shared memory
    mem_size : int
        the size of the shared memory. can be ignored if np_shape is provided
    np_shape : tuple
        the shape of the numpy array. can be ignored if mem_size is provided
    np_type : numpy.dtype
        the type of the numpy array
    create : bool, optional
        create the shared memory if it does not exist, by default False
    is_buffer : bool, optional
        if True, the shared memory will be used as a buffer, by default False


    Returns
    -------
    NumPySharedMemory
        the shared memory object
    """
    
    return NumpySharedMemory(
      mem_name=mem_name,
      mem_size=mem_size,
      np_shape=np_shape,
      np_type=np_type,
      create=create,
      is_buffer=is_buffer,
      log=self.log,
      **kwargs
    )

if __name__ == '__main__':
  from core import Logger
  from copy import deepcopy
  
  log = Logger("UTL", base_folder='.', app_folder='_local_cache')
  
  e = _UtilsBaseMixin()
  e.log = log
  
  d1 = e.DefaultDotDict(str)
  d1.a = "test"
  print(d1.a)
  print(d1.c)

  d1 = e.DefaultDotDict(lambda: str, {'a' : 'test', 'b':'testb'})
  print(d1.a)
  print(d1.b)
  print(d1.c)
  
  d1c = deepcopy(d1)
  
  d20 = {'k0':1, 'k1': {'k11': 10, 'k12': [{'k111': 100, 'k112':200}]}}
  d2 = e.NestedDotDict(d20)
  d20c = deepcopy(d20)
  d2c = deepcopy(d2)
  
  print(d2)
  print(d2.k0)
  print(d2.k1.k12[0].k112)
  
  
  d3 = defaultdict(lambda: DefaultDotDict({'timestamp' : None, 'data' : None}))
  
  s = json.dumps(d20)
  print(s)
  
  b64 = e.str_to_base64(s)
  print("{}: {}".format(len(b64), b64[:50]))
  print(e.base64_to_str(b64))

  b64c = e.str_to_base64(s, compress=True)
  print("{}: {}".format(len(b64c), b64c[:50]))
  print(e.base64_to_str(b64c, decompress=True))
    
  config = e.load_config_file(fn='./config_startup.txt')
  

  d4 = NestedDefaultDotDict()
  
  assert d4.test == {}, "Accessing an undefined key did not return empty dict."
  
  # Test case 2: Automatically creates nested dictionaries and sets value
  d4.test2.x = 5
  assert d4.test2.x == 5, "Nested assignment failed."
  
  # Test case 3: Auto-creates both test3 and test4, where test4 has value None
  _ = d4.test3.test4  # Access to create
  assert len(d4.test3) != 0 and len(d4.test3.test4) == 0, "Nested auto-creation failed."
  
  print("All tests passed.")  
  
  