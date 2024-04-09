import sys
import torch as th
import torchvision as tv
import time
import json
import os
import gc
import numpy as np

from core.local_libraries.nn.th.utils import th_resize_with_pad
from PyE2 import load_dotenv


load_dotenv()


def load_model_full(
  log, model_name, model_factory, weights, hyperparams,
  weights_filename=None, hyperparams_filename=None,
  device='cpu', return_config=False
):
  """
  Method for loading a model from a given class, with given configuration
  (either as a dictionary or as a .json) hyperparameters.
  Parameters
  ----------
  log - Logger, swiss knife object used in all DecentrAI
  model_name - str, name of the model
  model_factory - class extending torch.nn.Module, class of the given model
  weights - str, local path or url to file with the weights for the model
  device - str or torch.device, device on which the input will be put
  hyperparams - str or dict
  - if dict it will be considered the configuration of the specified model
  - if str it will be considered either a local path or an url to a file containing the hyperparameters
  weights_filename - str or None
  - if weights is an url this has to be the filename to be used when saving the file locally
  hyperparams_filename - str or None
  - if hyperparams is an url this has to be the filename to be used when saving the file locally
  return_config - bool, whether to return the config as a dictionary

  Returns
  -------
  res - (torch.nn.Module, dict) if return_config else torch.nn.Module
  The instantiated model with the weights loaded and if required, the config dictionary of the model.
  """
  log.P(
    f'Attempting to load model {model_name} with config: {hyperparams} and weights at: {weights} on device {device}')
  if weights_filename is None and os.path.isfile(weights):
    weights_filename = os.path.split(weights)[-1]
  elif weights_filename is None:
    raise Exception('`weights_filename` not provided and `weights` is not a local file. Please either provide '
                    'a path to an existing local file or specify `weights_filename`!')
  if isinstance(hyperparams, str) and hyperparams_filename is None and os.path.isfile(hyperparams):
    hyperparams_filename = os.path.split(hyperparams)[-1]
  elif isinstance(hyperparams, str) and hyperparams_filename is None:
    raise Exception('`hyperparams_filename` not provided and `hyperparams` is not a local file. Please either provide '
                    'a path to an existing local file or specify `hyperparams_filename`!')

  download_kwargs = log.config_data.get('MODEL_ZOO_CONFIG', {
      "endpoint": os.environ["EE_MINIO_ENDPOINT"],
      "access_key": os.environ["EE_MINIO_ACCESS_KEY"],
      "secret_key": os.environ["EE_MINIO_SECRET_KEY"],
      "secure": os.environ["EE_MINIO_SECURE"],
      "bucket_name": "model-zoo"
    })

  saved_files, msg = log.maybe_download_model(
    url=weights,
    model_file=weights_filename,
    **download_kwargs,
  )

  hyperparams_files = [None]
  if type(hyperparams) == str:
    hyperparams_files, _ = log.maybe_download_model(
      url=hyperparams,
      model_file=hyperparams_filename,
      **download_kwargs
    )
    hyperparams = None
  model, hyperparams = log.load_model_and_weights(
    model_class=model_factory,
    weights_path=saved_files[0],
    config_path=hyperparams_files[0],
    return_config=True,
    device=device,
    config=hyperparams
  )
  model.eval()

  return (model, hyperparams) if return_config else model


class BaseTorchScripter:
  """
  This class will be used in order to convert already existent neural models to torchscripts.
  """

  def __init__(
    self, log, model, model_name, input_shape,
    model_config=None, preprocess_method=None,
    matching_method=None,
    predict_method=None, use_fp16=False,
    use_amp=False
  ):
    """
    Parameters
    ----------
    log - Logger, swiss knife object used in all DecentrAI
    model - torch.nn.Module, the instantiated model to be converted
    model_name - str, name of the model
    input_shape - list, expected input shape by the model
    model_config - dict, configuration of the converted model
    preprocess_method - method, method of preprocessing the input in case it is needed
    - for the default value of this see self.preprocess_callback
    matching_method - method, method for checking if the output of the model matches the output of the torchscript
    - for the default value of this see self.matching_callback
    predict_method - method, method of running inferences given model and inputs
    - for the default value of this see self.default_predict
    use_fp16 - bool, whether to use fp16 when tracing a model
    use_amp - bool, whether to use amp when tracing a model
    """
    self.model = model
    self.model_name = model_name
    self.model_config = {} if model_config is None else model_config
    self.log = log
    self.input_shape = input_shape
    self.preprocess_method = preprocess_method if preprocess_method is not None else self.preprocess_callback
    self.matching_method = matching_method if matching_method is not None else self.matching_callback
    self.predict_method = predict_method if predict_method is not None else self.default_predict
    self.use_amp = use_amp
    self.use_fp16 = use_fp16
    if self.use_fp16:
      self.model.half()
    return

  def clear_cache(self):
    th.cuda.empty_cache()
    gc.collect()
    return

  def default_predict(self, model, inputs):
    return model(inputs)

  def __predict(self, model, inputs):
    if self.use_amp:
      with th.cuda.amp.autocast(enabled=self.use_amp):
        with th.no_grad():
          return self.predict_method(model, inputs)
    else:
      with th.no_grad():
        return self.predict_method(model, inputs)
    # endif use_amp

  def preprocess_callback(self, inputs, device='cpu', normalize=True, **kwargs):
    """
    This default callback will deal with resizing images, hence the majority of the models used in DecentrAI are
    computer vision models at the moment.
    Parameters
    ----------
    inputs - list, list of inputs on which to trace/test the model
    device - str or torch.device, device on which the input will be put

    Returns
    -------
    res - the data ready to enter the model
    """
    h, w = self.input_shape[:2]
    self.log.P("  Resizing from {} to {}".format([x.shape for x in inputs], (h, w)))
    results = th_resize_with_pad(
      img=inputs,
      h=h,
      w=w,
      device=device,
      normalize=normalize,
      return_original=False,
      half=self.use_fp16
    )
    if len(results) < 3:
      prep_inputs, lst_original_shapes = results
    else:
      prep_inputs, lst_original_shapes, lst_original_images = results
    return prep_inputs

  def convert_to_batch_size(self, inputs, batch_size):
    if hasattr(inputs, 'shape') and len(inputs.shape) == len(self.input_shape):
      # here inputs will be a single input
      inputs = [inputs for _ in range(batch_size)]
    else:
      # here inputs will be a list of inputs
      if hasattr(inputs, '__len__') and len(inputs) != batch_size:
        inputs = [inputs[i % len(inputs)] for i in range(batch_size)]
      # endif inputs of different length than the batch size needed
    # endif inputs is a single input
    return inputs

  def model_timing(self, model, model_name, inputs, nr_warmups=20, nr_tests=20, **kwargs):
    """
    Method for timing the speed of a certain model on given inputs.
    Parameters
    ----------
    model - torch.nn.Module, the model to be checked
    model_name - str, name of the timed model
    inputs - any, input data on which the model will be traced
    nr_warmups - int, how many inferences to make for warm up
    nr_test - int, how many inferences to make for timings

    Returns
    -------
    res - float, average time of inference for a batch
    """
    self.log.restart_timer(model_name)
    # warmup
    self.log.P(f"  Warming up ({nr_warmups} inferences)...")
    for _ in range(nr_warmups):
      print('.', flush=True, end='')
      preds = self.__predict(model, inputs)
    print('')

    # timing
    self.log.P(f"  Predicting ({nr_tests} inferences)...")
    for _ in range(nr_tests):
      print('.', flush=True, end='')
      self.log.start_timer(model_name)
      preds = self.__predict(model, inputs)
      self.log.stop_timer(model_name)
    print('')
    return self.log.get_timer_mean(model_name)

  def matching_callback(self, output1, output2, atol=0.00001, **kwargs):
    """
    Method for validating that 2 outputs are matching
    Parameters
    ----------
    output1 - any
    output2 - any
    atol - float, absolute tolerance for the comparison

    Returns
    -------
    True if outputs are matching, False otherwise
    """
    if type(output1) != type(output2):
      return False
    if isinstance(output1, list):
      return len(output1) == len(output2) and all([np.allclose(output1[i], output2[i]) for i in range(len(output1))])
    if isinstance(output1, th.Tensor):
      output1 = output1.detach().cpu().numpy()
      output2 = output2.detach().cpu().numpy()

    if (isinstance(output1, np.ndarray) and np.any(output1 != output2)) or (not isinstance(output1, np.ndarray) and output1 != output2):
      try:
        return np.allclose(output1, output2, atol=atol)
      except Exception as e:
        print(f'Error! {e}')
        return False
    # endif outputs are different
    return True

  def test(
      self, ts_path, model, inputs, batch_size=2, device='cpu',
      nr_warmups=20, nr_tests=20, skip_preprocess=False,
      use_fp16=False, **kwargs
  ):
    """
    Method for testing the torchscript at the same time with the model in order to
    validate that the outputs are the same and to check the speed difference between the 2.
    Parameters
    ----------
    ts_path - str, path to the torchscript
    model - torch.nn.Module, the model to be checked
    inputs - any, input data on which the model will be traced
    batch_size - int, Warning: the batch_size for testing and the batch
    size on which the model was traced should be different in order to
    further check that the model was traced correctly
    device - str or torch.device
    skip_preprocess - bool, whether to skip the preprocessing of the input
    nr_warmups - int, how many inferences to make for warm up
    nr_test - int, how many inferences to make for timings
    use_fp16 - bool, whether to convert both the model and the traced model to
    float16 before testing

    Returns
    -------
    res - (ok, timings), where:
    ok - bool, True if the outputs coincide, False otherwise
    timings - dict, the timers of both the model and the torchscript version
    """
    str_prefix = f'[bs={batch_size}]'
    self.log.P(f"{str_prefix}Testing the graph from {ts_path} for {self.model_name} with bs={batch_size} on device {device}")
    extra_files = {'config.txt': ''}
    ts_model = th.jit.load(
      f=ts_path,
      map_location=device,
      _extra_files=extra_files,
    )
    config = json.loads(extra_files['config.txt'].decode('utf-8'))
    ts_input_shape = config.get('input_shape')
    assert list(ts_input_shape) == list(self.input_shape), \
        f'Error! The input shape of the model and the .ths do not coincide! ' \
        f'The specified shape for the model was {self.input_shape}, while the ' \
        f'input shape in the .ths config is {ts_input_shape}'

    if not skip_preprocess:
      self.log.P(f'{str_prefix}Preprocessing...')
      inputs = self.convert_to_batch_size(inputs=inputs, batch_size=batch_size)
      inputs = self.preprocess_method(inputs=inputs, device=device, **kwargs)
    else:
      self.log.P(f'{str_prefix}Skipping preprocessing')
    # endif skip_preprocess

    suffix_str = ""
    if use_fp16:
      ts_model.half()
      model.half()
      inputs = inputs.half()
      suffix_str = "[FP16]"
    # endif use_fp16

    self.log.P(f'{str_prefix}Inferences...')
    model_output = self.__predict(model, inputs)
    ts_output = self.__predict(ts_model, inputs)

    self.log.P(f'{str_prefix}Validating...')
    ok = self.matching_method(model_output, ts_output, **kwargs)

    self.log.P(f'{str_prefix}Timing...')
    model_time = self.model_timing(
      model=model, model_name=self.model_name, inputs=inputs,
      nr_warmups=nr_warmups, nr_tests=nr_tests, **kwargs
    )
    self.clear_cache()
    ts_time = self.model_timing(
      model=ts_model, model_name=self.model_name + '_ts', inputs=inputs,
      nr_warmups=nr_warmups, nr_tests=nr_tests, **kwargs
    )
    self.clear_cache()
    timing_dict = {
      f'{self.model_name}_python': model_time,
      f'{self.model_name}_ts': ts_time,
      f'{self.model_name}_python_per_input': model_time / batch_size,
      f'{self.model_name}_ts_per_input': ts_time / batch_size,
    }
    self.log.P(f'{str_prefix}Model outputs are{"" if ok else " not"} matching!!{suffix_str}', color='g' if ok else 'r')
    total_gain = model_time - ts_time
    model_time_per_input = model_time / batch_size
    ts_time_per_input = ts_time / batch_size
    self.log.P(f'{str_prefix}Time for normal model: {model_time:.05f}s [{model_time_per_input:.05f} per input]{suffix_str}')
    self.log.P(f'{str_prefix}Time for traced model: {ts_time:.05f}s [{ts_time_per_input:.05f} per input]{suffix_str}')
    self.log.P(
      f'{str_prefix}Speed gain of {total_gain:.05f}s per batch and {total_gain / batch_size:.05f}s per input! ({total_gain/model_time*100:.02f}%){suffix_str}',
      color='g' if total_gain > 0 else 'r'
    )
    return ok, timing_dict

  def generate(
      self, inputs, batch_size=1, device='cpu', to_test=False,
      nr_warmups=20, nr_tests=20, no_grad_tracing=True,
      test_fp16=False, **kwargs
  ):
    """
    Method for generating the torchscript on given device, batch size and inputs.
    Parameters
    ----------
    inputs - any, input data on which the model will be traced
    batch_size - int
    device - str or torch.device
    to_test - bool, whether to test the model after torchscripting
    nr_warmups - int, how many inferences to make for warm up, relevant only if to_test=True
    nr_test - int, how many inferences to make for timings, relevant only if to_test=True
    no_grad_tracing - bool, whether to apply th.no_grad() when tracing the model

    Returns
    -------
    res - path of the saved torchscript
    """
    self.log.P(f"Generating torchscript for {self.model_name} on {device} with batch_size={batch_size} "
               f"and the following config:")
    self.log.P(f"{self.model_config}")
    config = {
      **self.model_config,
      'input_shape': self.input_shape,
      'python': sys.version.split()[0],
      'torch': th.__version__,
      'torchvision': tv.__version__,
      'device': device,
      'optimize': False,
      'date': time.strftime('%Y%m%d', time.localtime(time.time())),
      'model': self.model_name,
    }
    original_inputs = inputs
    inputs = self.convert_to_batch_size(inputs=original_inputs, batch_size=batch_size)
    inputs = self.preprocess_method(inputs=inputs, device=device, **kwargs)

    self.log.P("  Scripting...")
    self.model.to(device)
    if no_grad_tracing:
      with th.no_grad():
        if isinstance(inputs, dict):
          traced_model = th.jit.trace_module(self.model, inputs, strict=False)
        else:
          traced_model = th.jit.trace(self.model, inputs, strict=False)
        # endif inputs dict
      # endwith th.no_grad()
    else:
      if isinstance(inputs, dict):
        traced_model = th.jit.trace_module(self.model, inputs, strict=False)
      else:
        traced_model = th.jit.trace(self.model, inputs, strict=False)
      # endif inputs dict
    # endif no_grad_tracing

    self.log.P("  Forwarding using traced...")
    output = self.__predict(traced_model, inputs)

    extra_files = {'config.txt': json.dumps(config)}
    save_dir = os.path.join(self.log.get_models_folder(), 'traces')
    os.makedirs(save_dir, exist_ok=True)
    model_fn = self.model_name + f'{"_fp16" if self.use_fp16 else ""}_bs{batch_size}.ths'
    fn = os.path.join(save_dir, model_fn)
    self.log.P(f"  Saving '{fn}'...")
    traced_model.save(fn, _extra_files=extra_files)

    extra_files['config.txt'] = ''
    loaded_model = th.jit.load(fn, _extra_files=extra_files)
    config = json.loads(extra_files['config.txt'].decode('utf-8'))
    self.log.P("  Loaded config with {}".format(list(config.keys())))

    prep_inputs_test = self.convert_to_batch_size(inputs=original_inputs, batch_size=2 * batch_size)
    prep_inputs_test = self.preprocess_method(inputs=prep_inputs_test, device=device, **kwargs)

    self.log.P("  Running forward...")
    preds = self.__predict(loaded_model, prep_inputs_test)
    self.log.P(f"  Done running forward. Ouput:\n{preds}")

    if to_test:
      self.log.P('Starting validation phase...')
      test_kwargs = {
        'ts_path': fn,
        'model': self.model,
        'inputs': prep_inputs_test,
        'batch_size': batch_size * 2,
        'device': device,
        'nr_warmups': nr_warmups,
        'nr_tests': nr_tests,
        'skip_preprocess': True,
        **kwargs
      }
      self.test(**test_kwargs)
      if test_fp16:
        self.test(use_fp16=True, **test_kwargs)
    # endif to_test

    return fn
