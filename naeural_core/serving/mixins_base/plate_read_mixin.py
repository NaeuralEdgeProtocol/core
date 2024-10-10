from naeural_core.local_libraries.nn.th.image_dataset_stage_preprocesser import PreprocessResizeWithPad
from naeural_core.constants import CarAccess as ctc


CHARS = [
  '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
  '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
  '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
  '新',
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
  'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
  'W', 'X', 'Y', 'Z', 'I', 'O', '-'
]


PLATE_READ_CONFIG = {
  "DEBUG_MODE": True,
  "WITNESS_INFO": False,

  ################### MODEL_NAME AND URL ###################
  "LPRNET_URL": "minio:LPRNet/LPR_5.2.5_it_3_fp32_bs1.ths",
  "LPRNET_PATH": None,
  "LPRNET_FILENAME": 'LPR_5.2.5_it_3_fp32_bs1.ths',
  "LPRNET_URL_FP16": "minio:LPRNet/LPR_5.2.5_it_3_fp16_bs1.ths",
  "LPRNET_FILENAME_FP16": 'LPR_5.2.5_it_3_fp16_bs1.ths',

  "STN_URL_FP16": 'minio:STN/stn3.6.1_0_fp16_bs10.ths',
  "STN_PATH": None,
  'STN_FILENAME_FP16': 'stn3.6.1_0_fp16_bs10.ths',
  "STN_URL": 'minio:STN/stn3.6.1_0_bs10.ths',
  'STN_FILENAME': 'stn3.6.1_0_bs10.ths',
  ################### END MODEL_NAME AND URL ###################

  "LPRNET_RESIZE_TYPE": "resize",
  "STN_RESIZE_TYPE": "resize",
  "STN_EXPANDING_PERCENT": 0.5,  # 0.25, 0.50
  "STN_LP_CENTER_SIZE": (24, 94),

  "REMOVE_DUPLICATES": True,
  "REMOVE_SPECIALS": True,

  'LPRNET_IMG_H': 24,
  'LPRNET_IMG_W': 94,

  'STN_OFFSET_W': 3,
  'STN_OFFSET_H': 1,

  'VALIDATION_RULES': {}
}


class _PlateReadMixin:
  """MODEL NAMES AND URLS"""
  if True:
    def get_lprnet_url(self):
      if self.cfg_fp16:
        return self.cfg_lprnet_url_fp16
      return self.cfg_lprnet_url

    def get_lprnet_filename(self):
      if self.cfg_fp16:
        return self.cfg_lprnet_filename_fp16
      return self.cfg_lprnet_filename

    def get_stn_url(self):
      if self.cfg_fp16:
        return self.cfg_stn_url_fp16
      return self.cfg_stn_url

    def get_stn_filename(self):
      if self.cfg_fp16:
        return self.cfg_stn_filename_fp16
      return self.cfg_stn_filename
  """END MODEL NAMES AND URLS"""

  """IMG SHAPE"""
  if True:
    @property
    def stn_img_shape(self):
      hyperparams = self.graph_config[self.get_stn_filename()]
      return hyperparams['input_shape'][:2]
  """END IMG SHAPE"""

  """ADDITIONALS"""
  if True:
    @property
    def _first_idx(self):
      if self.cfg_remove_specials:
        return 31
      return 0
  """END ADDITIONALS"""

  """WARMUPS"""
  if True:
    def _lprnet_warmup(self):
      self.model_warmup_helper(
        model=self.lprnet_model,
        input_shape=(3, self.cfg_lprnet_img_h, self.cfg_lprnet_img_w),
        max_batch_size=self.cfg_max_batch_second_stage,
        model_name=self.get_lprnet_filename()
      )
      return

    def _stn_warmup(self):
      self.model_warmup_helper(
        model=self.stn_model,
        input_shape=(3, *self.stn_img_shape),
        max_batch_size=self.cfg_max_batch_second_stage,
        model_name=self.get_stn_filename()
      )
      return
  """END WARMUPS"""

  """LOAD MODELS"""
  if True:
    def _load_lprnet_model(self):
      self.lprnet_model, self.graph_config[self.get_lprnet_filename()] = self._prepare_ts_model(
        fn_model=self.get_lprnet_filename(),
        url=self.get_lprnet_url(),
        return_config=True
      )

      self.lprnet_transforms = self.tv.transforms.Compose([
        self.tv.transforms.Resize(size=(self.cfg_lprnet_img_h, self.cfg_lprnet_img_w))
      ])

      self._lprnet_warmup()
      return

    def _load_stn_model(self):
      self.stn_model, self.graph_config[self.get_stn_filename()] = self._prepare_ts_model(
        fn_model=self.get_stn_filename(),
        url=self.get_stn_url(),
        return_config=True
      )

      img_shape = self.stn_img_shape

      if self.cfg_stn_resize_type == 'resize_with_pad':
        self.stn_transforms = self.tv.transforms.Compose([
          PreprocessResizeWithPad(h=img_shape[0], w=img_shape[1], normalize=False, fill_value=144),
        ])
      else:
        self.stn_transforms = self.tv.transforms.Compose([
          self.tv.transforms.Resize(size=img_shape),
        ])

      self._stn_warmup()
      return

    def load_reader_models(self):
      self._load_lprnet_model()
      self._load_stn_model()
      return
  """END LOAD MODELS"""

  def th_to_numpy(self, th_x, detach=False):
    return th_x.cpu().numpy() if not detach else th_x.detach().cpu().numpy()

  def th_image_to_numpy(self, th_img, detach=False):
    return self.th_to_numpy(th_img.permute(1, 2, 0), detach=detach)

  def maybe_add_extra_data(
      self, res, lpr_pred, lpr_crop, lpr_batch_img,
      lpr_pred_no_stn, lpr_batch_img_no_stn, is_valid=False
  ):
    if self.cfg_witness_info:
      res[ctc.LICENSE_PLATE_STN] = ''.join(lpr_pred['PLATE']) if is_valid else None
      res[ctc.LICENSE_PLATE_PRCS] = lpr_pred['CONFIDENCE'] if is_valid else None
      res[ctc.LICENSE_PLATE_CROP] = self.th_image_to_numpy(lpr_crop) if is_valid else None

      lprnet_input_img = self.th_image_to_numpy(lpr_batch_img)
      res[ctc.LICENSE_PLATE_INPUT] = lprnet_input_img if is_valid else None
      res[ctc.RAW_LICENSE_PLATE] = ''.join(lpr_pred['RAW_PLATE']) if is_valid else None
      res[ctc.RAW_LICENSE_PLATE_PRCS] = lpr_pred['RAW_CONFIDENCE'] if is_valid else None

      if self.cfg_debug_mode:
        res[ctc.LICENSE_PLATE_NO_STN] = ''.join(lpr_pred_no_stn['PLATE']) if is_valid else None
        res[ctc.LICENSE_PLATE_PRCS_NO_STN] = lpr_pred_no_stn['CONFIDENCE'] if is_valid else None
        lprnet_input_no_stn_img = self.th_image_to_numpy(lpr_batch_img_no_stn)
        res[ctc.LICENSE_PLATE_INPUT_NO_STN] = lprnet_input_no_stn_img if is_valid else None
        res[ctc.RAW_LICENSE_PLATE_NO_STN] = ''.join(lpr_pred_no_stn['RAW_PLATE']) if is_valid else None
        res[ctc.RAW_LICENSE_PLATE_PRCS_NO_STN] = lpr_pred_no_stn['RAW_CONFIDENCE'] if is_valid else None
      # endif self.cfg_debug_mode
    # endif self.cfg_witness_info
    return res

  def _aggregate_lprnet_second_stage_batch_predict(self, lst_results):
    return self.th.cat(lst_results)

  def _aggregate_stn_second_stage_batch_predict(self, lst_results):
    lst_th_images = [e[1] for e in lst_results]
    return self.th.cat(lst_th_images)

  def _post_process_stn(self, th_images, th_original_sizes):
    if self.cfg_stn_lp_center_size is None:
      N = th_images.shape[0] if type(th_images) == self.th.Tensor else len(th_images)
      lst_th_img_shp = [(th_images[i], (th_images[i].shape[-2:], th_original_sizes[i][-2:]))
                        for i in range(N)]
      expand_scale_factor = self.cfg_stn_expanding_percent / (1 + 2 * self.cfg_stn_expanding_percent)
      off_w = self.cfg_stn_offset_w if expand_scale_factor > 0 else 0
      off_h = self.cfg_stn_offset_h if expand_scale_factor > 0 else 0
      lst_th_images_cut = []

      for th_img, (th_after, th_before) in lst_th_img_shp:
        scale = min(th_after[0] / th_before[0],
                    th_after[1] / th_before[1])

        r_pad_h = (th_after[0] - th_before[0] * scale) / 2
        r_pad_w = (th_after[1] - th_before[1] * scale) / 2

        h, w = th_after

        plate_h = int((h - 2 * r_pad_h) * (1 - 2 * expand_scale_factor))
        plate_w = int((w - 2 * r_pad_w) * (1 - 2 * expand_scale_factor))

        pad_h = int(r_pad_h + (expand_scale_factor * (h - 2 * r_pad_h)))
        pad_w = int(r_pad_w + (expand_scale_factor * (w - 2 * r_pad_w)))

        th_result_img = th_img[:, pad_h - off_h: plate_h + pad_h + off_h, pad_w - off_w: plate_w + pad_w + off_w]
        lst_th_images_cut.append(th_result_img)
    else:
      h, w = th_images.shape[-2:]
      lp_h, lp_w = self.cfg_stn_lp_center_size

      t = (h - lp_h) // 2
      l = (w - lp_w) // 2
      b = t + lp_h
      r = l + lp_w

      lst_th_images_cut = th_images[:, :, t:b, l:r]
    return lst_th_images_cut

  def _transform_and_predict(
      self, model_name, transformation, lst_cropped_images,
      model, max_batch_size, aggregate_batch_predict_callback
  ):
    self._start_timer(model_name + '_transform')
    th_batch = self.th.cat([transformation(x.unsqueeze(0)) for x in lst_cropped_images])
    self._stop_timer(model_name + '_transform')

    self._start_timer(f'{model_name}_pred_b{len(th_batch)}_{max_batch_size}')
    preds = self._batch_predict(
      prep_inputs=th_batch,
      model=model,
      batch_size=max_batch_size,
      aggregate_batch_predict_callback=aggregate_batch_predict_callback
    )
    self._stop_timer(f'{model_name}_pred_b{len(th_batch)}_{max_batch_size}')
    return preds, th_batch

  def _stn_stage(self, cropped_images):
    th_original_sizes = self.th.tensor([list(x.shape) for x in cropped_images])
    th_stn_images, _ = self._transform_and_predict(
      model_name='stn',
      transformation=self.stn_transforms,
      lst_cropped_images=cropped_images,
      model=self.stn_model,
      max_batch_size=self.cfg_max_batch_second_stage,
      aggregate_batch_predict_callback=self._aggregate_stn_second_stage_batch_predict,
    )

    res_images = self._post_process_stn(
      th_images=th_stn_images,
      th_original_sizes=th_original_sizes
    )

    return res_images

  def lprnet_predict(self, lst_cropped_images):
    lprnet_preds, batch_stn_imgs = self._transform_and_predict(
      model_name='lpr',
      transformation=self.lprnet_transforms,
      lst_cropped_images=lst_cropped_images,
      model=self.lprnet_model,
      max_batch_size=self.cfg_max_batch_second_stage,
      aggregate_batch_predict_callback=self._aggregate_lprnet_second_stage_batch_predict
    )

    self._start_timer('lprnet_post_process')
    lprnet_preds = self._lprnet_post_process(lprnet_preds)
    self._stop_timer('lprnet_post_process')
    return lprnet_preds, batch_stn_imgs

  def stn_expand_borders(self, l, t, r, b):
    border_expansion_offset_h = (b - t) * self.cfg_stn_expanding_percent
    border_expansion_offset_w = (r - l) * self.cfg_stn_expanding_percent

    new_t = t - border_expansion_offset_h
    new_b = b + border_expansion_offset_h
    new_l = l - border_expansion_offset_w
    new_r = r + border_expansion_offset_w
    return new_l, new_t, new_r, new_b

  def _lpr_stage(self, lpl_stage_out, lpr_inputs, lpl_masks):
    lprnet_masks = []
    stn_crop_imgs = []
    lprnet_crop_imgs = []
    self._start_timer("lprnet_crop")
    try:
      for i in range(len(lpl_stage_out[0])):
        if lpl_stage_out[1][i] > self.cfg_lpl_threshold:
          lprnet_masks.append(True)
          l, t, r, b = lpl_stage_out[0][i].clamp(min=0)
          new_l, new_t, new_r, new_b = self.stn_expand_borders(l, t, r, b)
          stn_crop_imgs.append(
            self.tv.transforms.functional.crop(
              img=lpr_inputs[i],
              left=new_l.int(),
              top=new_t.int(),
              width=(new_r - new_l).clamp(min=4).int() + 1,
              height=(new_b - new_t).clamp(min=4).int() + 1,
            )
          )
          lprnet_crop_imgs.append(
            self.tv.transforms.functional.crop(
              img=lpr_inputs[i],
              left=l.int(),
              top=t.int(),
              width=(r - l).clamp(min=4).int() + 1,
              height=(b - t).clamp(min=4).int() + 1,
            )
          )
        else:
          lprnet_masks.append(False)
        # endif lpl_stage_out[1][i] > self.cfg_lpl_threshold
      # endfor i in range(len(lpl_stage_out[0]))
    finally:
      self._stop_timer("lprnet_crop")
    # endtry

    if len(stn_crop_imgs) == 0:
      return ([], [], []) if not self.cfg_debug_mode else ([], [], [], [], [])

    stn_res_images = self._stn_stage(cropped_images=stn_crop_imgs)

    lprnet_preds, batch_stn_imgs = self.lprnet_predict(lst_cropped_images=stn_res_images)
    if self.cfg_debug_mode:
      lprnet_no_stn_preds, batch_no_stn_imgs = self.lprnet_predict(lst_cropped_images=lprnet_crop_imgs)
    # endif self.cfg_debug_mode

    lprnet_stn_results = []
    lprnet_no_stn_results = []
    k1 = 0
    self._start_timer("lprnet_agg_res")
    lpl_cumsums = [self.np.cumsum(lpl_masks[i]) for i in range(len(lpl_masks))]
    lpr_cumsum = self.np.cumsum(lprnet_masks)
    for i in range(len(lpl_masks)):
      res = []
      res_no_stn = []
      for j in range(len(lpl_masks[i])):
        res.append(
          lprnet_preds[lpr_cumsum[lpl_cumsums[i][j] - 1 + k1] - 1]
          if lpl_masks[i][j] and lprnet_masks[lpl_cumsums[i][j] - 1 + k1] else None
        )
        if self.cfg_debug_mode:
          res_no_stn.append(
            lprnet_no_stn_preds[lpr_cumsum[lpl_cumsums[i][j] - 1 + k1] - 1]
            if lpl_masks[i][j] and lprnet_masks[lpl_cumsums[i][j] - 1 + k1] else None
          )
      # endfor j in range(len(lpl_masks[i]))

      lprnet_stn_results.append(res)
      if self.cfg_debug_mode:
        lprnet_no_stn_results.append(res_no_stn)
      current = sum(lpl_masks[i])
      k1 += current
    # endfor i in range(len(lpl_masks))

    self._stop_timer("lprnet_agg_res")

    res = (lprnet_stn_results, lprnet_crop_imgs, batch_stn_imgs)
    if self.cfg_debug_mode:
      res += (lprnet_no_stn_results, batch_no_stn_imgs)

    return res

  def _remove_duplicates(self, arr, confs):
    # self.np.ediff1d(arr) will give us an array arr1
    # of size len(arr) - 1 in which
    # arr1[i] = arr[i + 1] - arr[i]
    if isinstance(arr, self.th.Tensor):
      idx = self.th.Tensor(self.th.diff(arr, n=1)).bool().to(self.dev)
      res = self.th.empty(size=(self.th.sum(idx) + 1,), dtype=arr.dtype).to(self.dev)
      res_confs = self.th.empty(size=(self.th.sum(idx) + 1,), dtype=confs.dtype).to(self.dev)
    else:
      idx = self.np.array(self.np.ediff1d(arr), dtype=bool)
      res = self.np.empty(shape=(self.np.sum(idx) + 1,), dtype=arr.dtype)
      res_confs = self.np.empty(shape=(self.np.sum(idx) + 1,), dtype=confs.dtype)

    res[0] = arr[0]
    res[1:] = arr[1:][idx]
    res_confs[0] = confs[0]
    res_confs[1:] = confs[1:][idx]

    return res, res_confs

  def top_k_values(self, values, keys=None, k=1, use_max=True, use_sorting=True, return_keys=True):
    # TODO: maybe move in libraries
    if keys is None:
      keys = values
      return_keys = False
    assert len(values) == len(keys), 'Both the `values` array and the `keys` array should have the same length'
    k = int(k)
    assert k > 0, '`k` should be greater than or equal to 1'
    if len(values) <= k:
      return (values, keys) if return_keys else values
    if not use_sorting:
      return (values[:k], keys[:k]) if return_keys else values[:k]
    offset = -1 if use_max else 1
    if isinstance(values, self.th.Tensor) and isinstance(keys, self.th.Tensor):
      idxs = keys.argsort()
      if use_max:
        idxs = idxs.flip(dims=(0,))
    else:
      values = self.np.array(values)
      keys = self.np.array(keys)
      idxs = keys.argsort()[::offset]
    filtered_idxs = idxs[:k]
    if isinstance(filtered_idxs, self.th.Tensor):
      filtered_idxs = filtered_idxs.sort().values
    else:
      filtered_idxs.sort()

    return (values[filtered_idxs], keys[filtered_idxs]) if return_keys else values[filtered_idxs]

  def _lprnet_post_process(self, preds):
    if preds is None:
      return []
    preds = preds.softmax(dim=1)
    lpr_max_len = self.graph_config[self.get_lprnet_filename()].get('lpr_max_len', 10)
    blank_value = len(self.CHARS) - 1
    if self.cfg_remove_specials:
      # if we remove the special characters then we won't use
      # the first 31 probabilities in the predictions
      # preds = [pred[self.cfg_first_idx:] for pred in preds]
      preds = preds[:, self._first_idx:, :]
      # if we remove the first 31 probabilities then the
      # blank index will also decrease by 31
      blank_value -= self._first_idx
    lst_preds = []
    for pred in preds:
      initial_pred = self.th.argmax(pred, dim=0)
      initial_conf = self.th.Tensor([pred[initial_pred[i]][i] for i in range(pred.shape[1])]).to(self.dev)

      raw_pred, raw_conf = initial_pred.cpu().detach().numpy(), initial_conf.cpu().detach().numpy()
      if self.cfg_remove_duplicates:
        initial_pred, initial_conf = self._remove_duplicates(arr=initial_pred, confs=initial_conf)

      no_blanks_idxs = self.th.where(initial_pred != blank_value)
      no_blanks = initial_pred[no_blanks_idxs]
      no_blanks_conf = initial_conf[no_blanks_idxs]
      if len(no_blanks) > lpr_max_len:
        no_blanks, no_blanks_conf = self.top_k_values(values=no_blanks, keys=no_blanks_conf, k=lpr_max_len)
      final_pred = no_blanks.cpu().detach().numpy()
      final_conf = no_blanks_conf.cpu().detach().numpy()

      curr_dict = {
        'PLATE': '_' if len(final_pred) < 1 else self.CHARS[self._first_idx:][final_pred],
        'CONFIDENCE': final_conf,
        'RAW_PLATE': '_' if len(raw_pred) < 1 else self.CHARS[self._first_idx:][raw_pred],
        'RAW_CONFIDENCE': raw_conf
      }
      lst_preds.append(curr_dict)

    return lst_preds
