import torch as th

from transformers import AutoTokenizer as LlmTokenizer
from transformers import AutoModelForCausalLM as LlmForCausalLM
from transformers import BitsAndBytesConfig


class LlmModelMixin(object):
  def __init__(self, *args, **kwargs):
    super(LlmModelMixin, self).__init__(*args, **kwargs)
    return


  def _get_placement_summary(self, indent=4):
    """Logs the device map from the model.

    Parameters
    ----------
      None.
    """

    def str_device(dev):
      str_place = dev
      str_place = str_place if str_place in ['cpu', 'disk'] else 'cuda:' + str_place
      return str_place

    str_indent = " " * indent
    result = ""
    if hasattr(self.model, 'hf_device_map'):
      self.placement = self.model.hf_device_map
      device = None
      prev_layer = None
      n = 0
      if len(self.placement) == 1:
        _layer = list(self.placement.keys())[0]
        result = str(self.placement[_layer])
      else:
        for layer in self.placement:
          if device != self.placement[layer]:
            if device is not None:
              result = result + prev_layer + ']({} layers): {}\n'.format(n, str_device(self.placement[layer]))
              n = 0
            device = self.placement[layer]
            result = result + str_indent + '[{} to '.format(layer)
            prev_layer = layer
          n += 1
        result = result + layer + ']({} layers): {}\n'.format(n, str_device(self.placement[layer]))
    return result


  def _get_model_load_config(self):
    """Creates the `model.from_pretrained` load config including the BitsAndBytes configuration.

    Returns
    -------
    tuple[dict, dict]
        the two configurations for the model and the quantization.
    """
    torch_dtype = "auto"
    load_in_8bit = False
    load_in_4bit = False
    quantization_params = None
    self.P("Preparing model load config for {}...".format(self.cfg_model_name))
    self.P("  Model weights size: {}".format(self.cfg_model_weights_size))
    self.P("  Has GPU: {}".format(self.has_gpu))
    # FIXME: multiple GPUs?
    if self.has_gpu and self.cfg_model_weights_size is not None:
      if self.cfg_model_weights_size in [4, 8]:
        # We're using 4 or 8 bit quantization and we need to prepare
        # the bitsandbytes quantization arguments.
        if self.cfg_model_weights_size == 4:
          load_in_4bit = True
          quantization_params = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # TODO: Check output quality for f16/bf16 <-- must
            bnb_4bit_compute_dtype=th.bfloat16, # no reason to use th.float16
          )
        else:
          load_in_8bit = True
          torch_dtype=None
          quantization_params = dict(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
          )
        #endif 4/8 bits
      else:
        # No BNB quatization is used so we're using either 16 or 32 bit floats.
        if self.cfg_model_weights_size == 16:
          torch_dtype = th.float16
        else:
          torch_dtype = th.float32
          self.P("WARNING: fp32 requested for a model that is usually fp16. This may cause OOM errors and if not will not yield better performance!", color='r')
        #endif 16 or BAD 32 bits
      #endif model quantization or normal
    #endif noconfig/config

    device_map = self._get_device_map()

    cache_dir = self.cache_dir
    token = self.hf_token

    model_params = dict(
      cache_dir=cache_dir,
      use_auth_token=token,
      low_cpu_mem_usage=True,
      torch_dtype=torch_dtype,
      device_map=device_map,
      # TODO: maybe change this to attn_implementation (got a warning that this was deprecated)
      use_flash_attention_2=self.cfg_use_flash_attention,
    )
    return model_params, quantization_params


  def _load_tokenizer(self):
    """Loads the tokenizer from the model and sets up padding.
    """
    # Load the tokenizer and output to log.
    cache_dir = self.cache_dir
    token = self.hf_token
    model_id = self.cfg_model_name
    self.P("Loading tokenizer for {} in '{}'...".format(model_id, cache_dir))
    self.tokenizer = LlmTokenizer.from_pretrained(
      model_id,
      cache_dir=cache_dir,
      use_auth_token=token,
    )

    # Fix for missing system roles in transformers.
    self._set_tokenizer_chat_template()

    # Use the unknown token as the padding token. It seems that at least
    # when quantized llama2 will look at the embeddings of padding tokens
    # so we should use something that is as ignorable as possible
    # embedding-wise.
    self.tokenizer.padding_side = 'right'
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.unk_token
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    self.padding_id = self.tokenizer.pad_token_id
    if self.padding_id is None:
      self.padding_id = self.tokenizer.unk_token_id
    if self.padding_id is None:
      self.padding_id = self.tokenizer.eos_token_id
    self.P(
      'Settting padding token to {} and padding token id to {}'
      .format(
        self.tokenizer.pad_token, self.tokenizer.pad_token_id
      )
    )

    self.P("  Loaded `{}` tokenizer".format(self.tokenizer.__class__.__name__))
    return


  def _load_model(self):
    """
    Load the model from the given configured model name and set up padding.
    Will first setup the model loading configuration and then load the model

    """
    model_id = self.cfg_model_name
    model_params, quantization_params = self._get_model_load_config()
    self.P("Loading {} with following parameters:\n{}\nQuantization params: {}".format(
      model_id,
      self.json_dumps(model_params, indent=4),
      self.json_dumps(quantization_params, indent=4),
      )
    )

    quantization_config = None
    if quantization_params is not None:
      quantization_config = BitsAndBytesConfig(**quantization_params)
    model_params['quantization_config'] = quantization_config

    self.model = LlmForCausalLM.from_pretrained(
        model_id,
        **model_params
    )

    compiled = self.cfg_th_compile
    if compiled:
      compile_mode = self.cfg_th_compile_mode
      self.P("Compiling model")
      self.model = th.compile(
        self.model,
        fullgraph=True,
        mode=compile_mode
      )
    #endif compile model

    self.P("  Loaded `{}` model".format(self.model.__class__.__name__))

    # Set the padding token to the chosen (<unk>) token.
    self.model.config.pad_token_id = self.padding_id
    self.P('Setting padding token ID to {}'.format(self.model.config.pad_token_id))

    # When the entire model is on the GPU we expect to get a {'':0} device map
    # which doesn't really tell us where the model is, only that it is on one
    # device. Additionally print the model device to avoid this corner case.
    self.P("Model {} loaded with dev map:\n{}".format(model_id, self._get_placement_summary()))
    device = next(self.model.parameters()).device
    self.P("First weight is on device: {}".format(device))

    return
