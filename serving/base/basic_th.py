import torch as th
import torchvision as tv

from core.serving.base.base_serving_process import ModelServingProcess as BaseServingProcess
from core.serving.mixins_base.th_utils import _ThUtilsMixin

_CONFIG = {
  **BaseServingProcess.CONFIG,
  
  "IMAGE_HW"                    : None,
  "WARMUP_COUNT"                : 3,
  "USE_AMP"                     : False,
  "USE_FP16"                    : True,
  "DEFAULT_DEVICE"              : "cuda:0",
  "URL"                         : None,
  "DEBUG_TIMERS"                : False,
  "MAX_BATCH_FIRST_STAGE"       : None,
  "MAX_BATCH_SECOND_STAGE"      : None,

  "MODEL_WEIGHTS_FILENAME"      : None,
  "MODEL_CLASSES_FILENAME"      : None,
  
  "SECOND_STAGE_MODEL_WEIGHTS_FILENAME" : None,
  "SECOND_STAGE_MODEL_CLASSES_FILENAME" : None,
  
  "CUDNN_BENCHMARK"             : False,

  
  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
  },
}


class BasicTorchServer(BaseServingProcess, _ThUtilsMixin):

  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self.default_input_shape = None
    self.class_names = None
    self.graph_config = {}

    super(BasicTorchServer, self).__init__(**kwargs)
    return

  @property
  def th(self):
    return th

  @property
  def tv(self):
    return tv

  @property
  def cfg_input_size(self):
    return self.cfg_image_hw

  @property
  def cfg_fp16(self):
    return self.cfg_use_fp16

  @property
  def th_dtype(self):
    ### TODO: Maybe we have uint as input
    return th.float16 if self.cfg_use_fp16 else th.float32

  @property
  def get_model_weights_filename(self):
    # This is done in order for us to be able to have more control over the weights filename
    return self.cfg_model_weights_filename

  @property
  def get_model_classes_filename(self):
    return self.cfg_model_classes_filename

  @property
  def get_url(self):
    return self.cfg_url

  def get_saved_model_fn(self):
    fn_saved_model = self.get_model_weights_filename if self.get_model_weights_filename is not None else self.server_name + '_weights.pt'
    return fn_saved_model

  def get_saved_classes_fn(self):
    fn_saved_classes = self.get_model_classes_filename if self.get_model_classes_filename is not None else self.server_name + '_classes.txt'
    return fn_saved_classes

  def has_device(self, dev=None):
    if dev is None:
      dev = th.device(self.cfg_default_device)
    elif isinstance(dev, str):
      dev = th.device(dev)
    dev_idx = 0 if dev.index is None else dev.index
    if dev.type == 'cpu' or (th.cuda.is_available() and th.cuda.device_count() > dev_idx):
      return True
    return False

  def _stop_timer(self, tmr, periodic=False):
    if self.cfg_debug_timers:
      th.cuda.synchronize()
    return super()._stop_timer(tmr, periodic=periodic)

  def _setup_model(self):
    fn_weights = None
    model_weights = None
    self.P("Model setup initiated for {}".format(self.__class__.__name__))

    if self.cfg_use_amp and self.cfg_use_fp16:
      self.P('Using both AMP and FP16 is not usually recommended.', color='r')

    str_dev = self.cfg_default_device.lower()
    dev = th.device(str_dev)
    if not self.has_device(dev):
      raise ValueError('Current machine does not have compute device {}'.format(dev))
    self.dev = dev

    # record gpu info status pre-loading
    lst_gpu_status_pre = None
    if 'cuda' in self.cfg_default_device:
      lst_gpu_status_pre = self.log.gpu_info(mb=True)

    # classes
    self.P("Prepping classes if available...")
    fn_classes = self.get_saved_classes_fn()
    url_classes = self.config_model.get(self.const.URL_CLASS_NAMES)
    if fn_classes is not None:
      self.download(
        url=url_classes, 
        fn=fn_classes,
      )
      self.class_names = self.log.load_models_json(fn_classes)

    if self.class_names is None:
      self.P("WARNING: based class names loading failed. This is not necesarely an error as classes can be loaded within models.")

    # weights preparation - only for non-dynamic models
    if self.get_url is not None:
      self.P("Prepping model weights or graph...")
      model_weights = self.get_saved_model_fn()
      self.download(
        url=self.get_url,
        fn=model_weights,
      )
      # TODO: maybe use fn_weights as the output of self.download from above
      fn_weights = self.log.get_models_file(model_weights)

    # following call should be defined in subclass and it is supposed to define
    # the model or directly load from jit file
    self.P("Loading model...")
    self.th_model = self._get_model(
      fn=model_weights
    )

    model_dev = next(self.th_model.parameters()).device
    if ( # need complex check as dev1 != dev2 will be true (if one index is None and other is 0 on same CUDA)
        (model_dev.type != self.dev.type) or  # cpu vs cuda
        ((model_dev.type == self.dev.type) and (model_dev.index != self.dev.index)) # cuda but on different onex
        ):
      self.P("Model '{}' loaded & placed on '{}' -  moving model to device:'{}'".format(
        self.__class__.__name__, model_dev, self.dev), color='y')
      self.th_model.to(self.dev)    
    self.th_model.eval()
    if self.cfg_use_fp16:
      self.th_model.half()
      
    # now load second stage model
    self._second_stage_model_load()
    
    # now warmup all models
    self._model_warmup()

    # record gpu info status post-loading
    taken, dev_nr = 0, 0
    if lst_gpu_status_pre is not None:
      dev_nr = 0 if dev.index is None else dev.index
      lst_gpu_status_post = self.log.gpu_info(mb=True)
      try:
        free_pre = lst_gpu_status_pre[dev_nr]['FREE_MEM']
        free_post = lst_gpu_status_post[dev_nr]['FREE_MEM']
        taken = free_pre - free_post
      except:
        self.P("Failed gpu check on {} PRE:{}, POST:{}\n {}\n>>>> END: GPU INFO ERROR".format(
          dev_nr, lst_gpu_status_pre, lst_gpu_status_post, 
          self.trace_info()), color='error'
        )
      
      
    curr_model_dev = next(self.th_model.parameters()).device
    
    msg = "Model {} prepared on device '{}'{}:".format(
      self.server_name, curr_model_dev,
      " using {:.0f} MB GPU GPU {}".format(taken, dev_nr) if taken > 0 else "",
      )
    msg = msg + '\n  {:<17} {}'.format('File:', fn_weights)
    for k in self.config_model:
      msg = msg + '\n  {:<17} {}'.format(k+':', self.config_model[k])
    self.P(msg, color='g')    
    return msg

  def get_input_shape(self):
    return (3, *self.cfg_input_size) if self.cfg_input_size is not None else None

  @staticmethod
  def __model_call_callback(th_inputs, model):
    """
    This is the default method for calling a model with the given inputs.
    Parameters
    ----------
    th_inputs - the inputs to be passed to the model
    model - the model to be called

    Returns
    -------
    res - the result of the model call
    """
    return model(th_inputs)

  def get_model_call_method(self, model_call_method=None):
    """
    This method returns the model call method to be used for calling the model.
    Parameters
    ----------
    model_call_method - the model call method to be used for calling the model

    Returns
    -------
    model_call_method - the model call method to be used for calling the model
    """
    if model_call_method is None:
      # If the method is not provided a custom method will be checked
      # and if it is not found the default method will be used
      default_method = getattr(self, 'model_call', None)
      return default_method if callable(default_method) else self.__model_call_callback
    return model_call_method

  def _forward_pass(self, th_inputs, model=None, model_call_method=None, debug=None, debug_str='', autocast=True):
    model = self.th_model if model is None else model
    debug = self._full_debug if debug is None else debug
    model_call_method = self.get_model_call_method(model_call_method=model_call_method)
    # endif model not provided
    if autocast:
      with th.cuda.amp.autocast(enabled=self.cfg_use_amp):
        with th.no_grad():
          if debug:
            self.P("  Forward pass {}, dev '{}' with {}:{}".format(
              debug_str, th_inputs.device, th_inputs.shape, th_inputs.dtype
            ))
          th_preds = model_call_method(model=model, th_inputs=th_inputs)
        # end no_grad
      # end autocast
    else:
      with th.no_grad():
        if debug:
          self.P("  Forward pass {}, no autocast on dev '{}' with {}:{}".format(
            debug_str, th_inputs.device, th_inputs.shape, th_inputs.dtype
          ))
        th_preds = model_call_method(model=model, th_inputs=th_inputs)
      # end no_grad
    # endif autocast or not
    return th_preds

  # TODO: maybe add support for randint?
  def model_warmup_helper(
      self, model=None, input_shape=None, warmup_count=None,
      max_batch_size=None, model_dtype=None, model_device=None,
      model_name=None, model_call_method=None
  ):
    """
    This method is used to warm up the model.
    Parameters
    ----------
    model - the model to be warmed up
    input_shape - the shape of the input to be used for the model after warmup
    warmup_count - the number of forward passes per batch size to be used for warmup
    max_batch_size - the maximum batch size to be used for the model after warmup
    model_dtype - the dtype of the input to be used for the model after warmup
    model_device - the device of the input to be used for the model after warmup
    model_name - the name of the model

    Returns
    -------
    None
    """
    model = self.th_model if model is None else model
    input_shape = self.get_input_shape() if input_shape is None else input_shape
    warmup_count = self.cfg_warmup_count if warmup_count is None else warmup_count
    max_batch_size = self.cfg_max_batch_first_stage if max_batch_size is None else max_batch_size
    model_name = self.__class__.__name__ if model_name is None else model_name

    if model is not None:
      if model_dtype is None or model_device is None:
        par0 = next(model.parameters())
        model_dtype = par0.dtype if model_dtype is None else model_dtype
        model_device = par0.device if model_device is None else model_device
      # endif model_dtype or model_device
      model_dev_type = model_device.type
      if input_shape is not None:
        self.P(f"Warming up model {model_name} with {input_shape}/{model_dtype} on device '{model_device}'"
               f" for MAX_BATCH_SIZE {max_batch_size} and WARMUP_COUNT {warmup_count}...")
        for wbs in range(1, max_batch_size + 1):
          shape = (wbs, *input_shape)
          th_warm = th.rand(
            *shape,
            device=model_device,
            dtype=model_dtype
          )
          for warmup_pass in range(1, warmup_count + 1):
            preds = self._forward_pass(
              th_warm, model=model, model_call_method=model_call_method,
              autocast='cuda' in model_dev_type and self.cfg_use_amp,
              debug=self._full_debug, debug_str=str(warmup_pass)
            )
          # endfor warmup_pass
        # endfor warmup_batch_size
      self.P("Model {} warmed up and ready for inference".format(model_name))
    else:
      self.P("ERROR: Model of {} not found".format(model_name))
    return

  def _model_warmup(self):
    shape = self.get_input_shape()
    th.backends.cudnn.benchmark = self.cfg_cudnn_benchmark

    self.model_warmup_helper(
      model=self.th_model,
      input_shape=shape,
      warmup_count=self.cfg_warmup_count,
      max_batch_size=self.cfg_max_batch_first_stage
    )

    self._second_stage_model_warmup()
    return
    
  def load_torchscript(self, fn_path, post_process_classes=False, device=None):
    """
    Generic method for loading a torchscript and returning both the model generated
    by it and its config.
    Parameters
    ----------
    fn_path - path of the specified torchscript
    include_classes - in case this is true we will try to reformat the classes names
      as a dictionary of {key:value} where the keys will be the classes indexes and
      the values will be the classes names

    Returns
    -------

    """
    extra_files = {'config.txt': ''}
    dct_config = None
    model = self.th.jit.load(
      f=fn_path,
      map_location=self.dev,
      _extra_files=extra_files,
    )
    model.eval()
    self.P("Done loading model on device {}".format(
      self.dev,
    ))
    try:
      dct_config = self.json.loads(extra_files['config.txt'].decode('utf-8'))
      if post_process_classes:
        dct_config['names'] = {int(k): v for k, v in dct_config.get('names', {}).items()}
      self.P("  Model config:\n{}".format(self.log.dict_pretty_format(dct_config)))
    except Exception as exc:
      self.P("Could not load in-model config '{}': {}. In future this will stop the model loading: {}".format(
        fn_path, exc, extra_files
      ), color='r')
    return model, dct_config

  def _class_dict_to_list(self, dct_classes):
    keys = sorted(list(dct_classes.keys()))
    result = [dct_classes[x] for x in keys]
    return result

  def ver_element_to_int(self, ver_element):
    """
    Method for getting rid of non-numeric suffixes from version elements.
    For example if our current version is '0.15.2+cpu' and the minimum version is '0.15.2' we want to
    still be able to tell that the current version is greater than or equal to the minimum version.
    Parameters
    ----------
    ver_element - the version element to be converted to int

    Returns
    -------
    int - the numeric value of the version element or 0 if the version element does not start with a digit.
    """
    ver_element_numeric = self.re.sub('\D.*', '', ver_element)
    return int(ver_element_numeric) if ver_element_numeric.isnumeric() else 0

  def ver_to_int(self, version):
    ver_elements = version.split('.')
    int_ver_elements = [self.ver_element_to_int(x) for x in ver_elements[:3]]
    weights = [1000000, 1000, 1]
    int_ver = sum(int_ver_elements[i] * weights[i] for i in range(min(len(weights), len(int_ver_elements))))

    return int_ver

  def check_version(self, min_ver, curr_ver):
    return self.ver_to_int(min_ver) <= self.ver_to_int(curr_ver)

  def valid_version(self, ver_str):
    return '.' in ver_str

  def check_versions(self, model_config, fn_path):
    err_keys = ['torch']
    env_versions = {
      'python': self.python_version(),
      'torch': self.th.__version__,
      'torchvision': self.tv.__version__
    }
    ts_versions = {key: model_config[key] if key in model_config.keys() else 'Unspecified' for key in env_versions.keys()}

    err_check = not all([
      self.check_version(min_ver=ts_versions[vkey], curr_ver=env_versions[vkey]) and self.valid_version(ts_versions[vkey])
      for vkey in err_keys
    ])
    warn_check = not all([
      self.valid_version(ts_versions[vkey]) and self.check_version(min_ver=ts_versions[vkey], curr_ver=env_versions[vkey])
      for vkey in env_versions.keys()
    ])

    if err_check:
      err_msg = f'ERROR! Torchscript graph from {fn_path} has versions above current environment versions!' \
                f'[Graph versions:{ts_versions} > Env versions: {env_versions}]'
      self.P(err_msg, color='e')
      raise Exception(err_msg)
    elif warn_check:
      warn_msg = f'WARNING! Torchscript graph from {fn_path} has versions above current environment versions or has ' \
                 f'unspecified/invalid versions! [Graph versions:{ts_versions} > Env versions: {env_versions}]'
      self.P(warn_msg, color='r')
    else:
      self.P(f'Graph version check passed! [{ts_versions} <= {env_versions}]')
    # endif version_check
    return

  def _prepare_ts_model(self, url=None, fn_model=None, post_process_classes=False, return_config=False, **kwargs):
    self.P("Preparing {} torchscript graph model {}...".format(self.server_name, self.version))
    if url is None:
      url = self.get_url
    if fn_model is None:
      fn_model = self.get_saved_model_fn()
    self.download(url, fn_model)
    fn_path = self.log.get_models_file(fn_model)
    model, config = None, None
    if fn_path is not None:
      self.P("Loading torchscript model {} ({:.03f} MB) at `{}` using map_location: {} on python v{}...".format(
          fn_model,
          self.os_path.getsize(fn_path) / 1024 / 1024,
          fn_path,
          self.dev,
          self.python_version()
        ),
        color='y'
      )
      model, config = self.load_torchscript(fn_path, post_process_classes=post_process_classes, device=self.dev)
      if self.cfg_fp16:
        model.half()
      self.check_versions(config, fn_path)
      if post_process_classes:
        class_names = self._class_dict_to_list(config['names'])
        if len(class_names) > 0:
          self.P("Loading class names from torchscript: {} classes ".format(len(class_names)))
          self.class_names = class_names
      # endif post_process_classes
    else:
      raise ValueError("Model loading failed (fn_path: {})! Please check config data: {}".format(
        fn_path, self.config_data
      ))
    return (model, config) if return_config else model
    
  def _get_model(self, fn):
    raise NotImplementedError()
    
  def _pre_process_images(self, images):
    raise NotImplementedError()
    
  # second stage methods
  def _second_stage_model_load(self):
    # this method defines default behavior for second stage model load    
    return

  def _second_stage_model_warmup(self):
    # this method defines default behavior for second stage model warmup
    return

  def _second_stage_process(self, th_preds, th_inputs=None, **kwargs):
    # this function processed post forward and should be used in subclasses
    # also here you we can add second/third stage classifiers/regressors (eg mask detector)
    return th_preds

  def _second_stage_classifier(self, first_stage_out, th_inputs):
    # override this to add second stage classifier/regressor fw pass
    # for this to be executed in model prep set `self.has_second_stage_classifier = True`
    return None

  # end second stage methods
  
  @staticmethod
  def _th_resize(self, lst_images, target_size):
    return

  ###
  ### BELOW MANDATORY (or just overwritten) FUNCTIONS:
  ###
  
  def _get_inputs_batch_size(self, inputs):
    """
    Extract the batch size of the inputs. This method is can be overwritten when inputs are dictionaries,
    because `len(inputs)` would simply be the number of keys in the dict.
    
    Example of use case: in `th_cqc`, the input is a kwarg dict with images and anchors, so the batch size
    should be the len of those inputs, not the number of inputs (which is constantly 2).

    Parameters
    ----------
    inputs : Any
        The input of the serving model

    Returns
    -------
    int
        The batch size of the input
    """
    return len(inputs) # not the greatest method but it works
  
  def _startup(self):
    msg = self._setup_model()
    return msg

  def _pre_process(self, inputs):    
    """
    This method does only the basic data extraction from upstream payload and 
    has replaced:
      ```
      def _pre_process(self, inputs):  
        prep_inputs = th.tensor(inputs, device=self.dev, dtype=self.th_dtype)
        return prep_inputs
      ```


    Parameters
    ----------
    inputs : list
      list of batched numpy images.

    Raises
    ------
    ValueError
      DESCRIPTION.

    Returns
    -------
    TYPE
      DESCRIPTION.

    """
    lst_images = None
    if isinstance(inputs, dict):
      lst_images = inputs.get('DATA', None)  
    elif isinstance(inputs, list):
      lst_images = inputs
    
    if lst_images is None or len(lst_images) == 0:
      msg = "Unknown or None `inputs` received: {}".format(inputs)
      self.P(msg, color='error')
      raise ValueError(msg)
    return self._pre_process_images(lst_images)

  def _aggregate_batch_predict(self, lst_results):
    if len(lst_results) == 1:
      return lst_results[0]
    else:
      if isinstance(lst_results[0], th.Tensor):
        return th.cat(lst_results, dim=0)
      else:
        raise NotImplementedError(
          "Please implement `_aggregate_batch_predict` method that can aggregate model output type: {}".format(
            type(lst_results[0]))
        )
    return

  def _batch_predict(self, prep_inputs, model, batch_size, aggregate_batch_predict_callback=None, **kwargs):
    lst_results = []
    start_time = self.time()

    if isinstance(prep_inputs, dict):
      last_dim = None
      for key in prep_inputs.keys():
        assert isinstance(prep_inputs[key], (list, th.Tensor, self.np.ndarray)), 'Invalid input type {}'.format(type(prep_inputs[key]))
        assert last_dim is None or last_dim == len(prep_inputs[key]), 'Inconsistent input dim: {} vs {}'.format(last_dim, prep_inputs[key])
        last_dim = len(prep_inputs[key])
      #endfor
      real_batch_size = last_dim
    else:
      real_batch_size = len(prep_inputs)
    #endif

    if batch_size is None:
      self.maybe_log_phase('_model_predict[bs=None]', start_time, done=False)
      if isinstance(prep_inputs, dict):
        return model(**prep_inputs, **kwargs)
      else:
        return model(prep_inputs, **kwargs)
      #endif
    #endif

    self.maybe_log_phase(f'_model_predict[bs={real_batch_size} mbs={batch_size} and bn={self.np.ceil(real_batch_size / batch_size).astype(int)}]', start_time, done=False)
    for i in range(self.np.ceil(real_batch_size / batch_size).astype(int)):
      if isinstance(prep_inputs, dict):
        batch = {}
        for key in prep_inputs.keys():
          batch[key] = prep_inputs[key][i * batch_size: (i+1) * batch_size]
        # endfor
        th_inferences = model(**batch, **kwargs)
      else:
        batch = prep_inputs[i * batch_size:(i+1) * batch_size]
        th_inferences = model(batch, **kwargs)
      # endif
      lst_results.append(th_inferences)
    # endfor batches
    self.maybe_log_phase(f'_model_predict[bs={real_batch_size} mbs={batch_size} and bn={self.np.ceil(real_batch_size / batch_size).astype(int)}]', start_time)
    if aggregate_batch_predict_callback is None:
      return self._aggregate_batch_predict(lst_results)
    else:
      return aggregate_batch_predict_callback(lst_results)

  def _predict(self, prep_inputs):
    ### TODO: This no longer works with multiple inputs
    if isinstance(prep_inputs, dict):
      kwargs_first_stage = prep_inputs.get('kwargs_first_stage', {})
      kwargs_second_stage = prep_inputs.get('kwargs_second_stage', {})
      inputs = prep_inputs.get('inputs', None)
      if inputs is None:
        raise ValueError('Prepocess returned dict without inputs')
    elif not isinstance(prep_inputs, (list, th.Tensor, self.np.ndarray)):
      raise ValueError('Preprocess returned invalid type {}'.format(type(prep_inputs)))
    else:
      inputs = prep_inputs
      kwargs_first_stage = {}
      kwargs_second_stage = {}
    #endif

    start_time = self.time()
    bs = self._get_inputs_batch_size(inputs)
    with th.cuda.amp.autocast(enabled=self.cfg_use_amp):
      with th.no_grad():
        str_predict_timer = 'fwd_b{}_{}'.format(str(bs), str(self.cfg_max_batch_first_stage))
        self.log.start_timer(self.predict_server_name + '_' + str_predict_timer)
        self.maybe_log_phase('_batch_process', start_time, done=False)
        th_x_stage1 = self._batch_predict(
          prep_inputs=inputs,
          model=self.th_model,
          batch_size=self.cfg_max_batch_first_stage,
          **kwargs_first_stage
        )
        self.maybe_log_phase('_batch_predict', start_time)
        self.log.stop_timer(self.predict_server_name + '_' + str_predict_timer)

        self._start_timer('stage2')
        self.maybe_log_phase('_second_stage_process', start_time, done=False)
        th_x_stage2 = self._second_stage_process(
          th_preds=th_x_stage1,
          th_inputs=inputs,
          **kwargs_second_stage
        )
        self.maybe_log_phase('_second_stage_process', start_time)

        self._stop_timer('stage2')
    return th_x_stage2

  def _post_process(self, preds):
    if isinstance(preds, (list, tuple)):
      # lets unpack
      outputs = []
      for pred in preds:
        if isinstance(pred, th.Tensor):
          outputs.append(pred.cpu().numpy())
    elif isinstance(preds, th.Tensor):
      outputs = preds.cpu().numpy()
    else:
      raise ValueError("Unknown torch model preds output of type {}".format(type(preds)))
    return outputs
  
  def _clear_model(self):
    if hasattr(self, 'th_model') and self.th_model is not None:
      self.P("Deleting current loaded torch model for {} ...".format(self.__class__.__name__))
      curr_model_dev = next(self.th_model.parameters()).device
      if 'cuda' in curr_model_dev.type.lower():
        lst_gpu_status_pre = self.log.gpu_info(mb=True)
        
      del self.th_model
      th.cuda.empty_cache()

      freed = 0
      idx = None
      if 'cuda' in curr_model_dev.type.lower():
        lst_gpu_status_post = self.log.gpu_info(mb=True)
        try:
          idx = 0 if curr_model_dev.index is None else curr_model_dev.index
          freed = lst_gpu_status_post[idx]['FREE_MEM'] - lst_gpu_status_pre[idx]['FREE_MEM']        
        except:
          self.P("Failed gpu check on {} PRE:{}, POST:{}, {}".format(
            idx, lst_gpu_status_pre[idx], lst_gpu_status_post[idx], 
            self.trace_info()), color='error'
          )



      self.P("Cleanup finished. Freed {:.0f} MB on GPU {}".format(freed, idx))
    return
  
  def _shutdown(self):
    self._clear_model()
    return

  def forward(self, inputs):
    prep_inputs = self._pre_process(inputs)
    fwd = self.th_model(prep_inputs)
    return fwd
  
  
