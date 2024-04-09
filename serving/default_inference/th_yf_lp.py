from core.constants import CarAccess as ctc
from core.serving.default_inference.th_yf8l import ThYf8l as BaseServingProcess
from core.serving.mixins_base.plate_read_mixin import _PlateReadMixin, CHARS, PLATE_READ_CONFIG as _PLATE_READ_CONFIG
from core.local_libraries.nn.th.image_dataset_stage_preprocesser import PreprocessResizeWithPad
from core.local_libraries.nn.th.utils import inverse_recompute_relative_y

_CONFIG = {
  **BaseServingProcess.CONFIG,
  'MAX_BATCH_SECOND_STAGE': 10,

  ################### MODEL_NAME AND URL ###################
  "LPL_URL": 'minio:LPL/DS_5.2.1_flip_it_0_fp32_bs1.ths',
  "LPL_PATH": None,
  'LPL_FILENAME': 'DS_5.2.1_flip_it_0_fp32_bs1.ths',
  "LPL_URL_FP16": 'minio:LPL/DS_5.2.1_flip_it_0_fp16_bs1.ths',
  'LPL_FILENAME_FP16': 'DS_5.2.1_flip_it_0_fp16_bs1.ths',

  **_PLATE_READ_CONFIG,
  ################### END MODEL_NAME AND URL ###################

  'COVERED_SERVERS': ['th_yf8l'],

  "LPL_THRESHOLD": 0.3,

  'HEIGHT_OFFSETS': [0, 5, 8],
  'WIDTH_OFFSETS': [0, 0, 5],
  'HW_THRESHOLDS': [0.29, 0.36],

  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
    **_PLATE_READ_CONFIG['VALIDATION_RULES'],
  },
}

DEBUG_COVERED_SERVERS = False


class ThYfLp(BaseServingProcess, _PlateReadMixin):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self._model_hyperparams = {}
    super(ThYfLp, self).__init__(**kwargs)

    self._has_second_stage_classifier = True
    return

  def _startup(self):
    super(ThYfLp, self)._startup()
    assert len(self.cfg_height_offsets) == len(self.cfg_width_offsets), \
        '"HEIGHT_OFFSETS" and "WIDTH_OFFSETS" should have the same length'
    assert len(self.cfg_height_offsets) - 1 == len(self.cfg_hw_thresholds), \
        '"HW_THRESHOLDS" should be one element shorter than "HEIGHT_OFFSETS"'

    self.CHARS = self.np.array(CHARS)
    return

  """MODEL NAMES AND URLS"""
  if True:
    # TODO: refactor this in order to by default consider the fp16
    #  model if the fp16 model is available instead of multiple get_X_url
    #  and get_X_filename methods
    def get_lpl_url(self):
      if self.cfg_fp16:
        return self.cfg_lpl_url_fp16
      return self.cfg_lpl_url

    def get_lpl_filename(self):
      if self.cfg_fp16:
        return self.cfg_lpl_filename_fp16
      return self.cfg_lpl_filename
  """END MODEL NAMES AND URLS"""

  """IMG SHAPE"""
  if True:
    @property
    def lpl_img_shape(self):
      hyperparams = self.graph_config[self.get_lpl_filename()]
      if 'image_height' in hyperparams.keys() and 'image_width' in hyperparams:
        return hyperparams['image_height'], hyperparams['image_width']
      if 'image_size' in hyperparams.keys():
        return hyperparams['image_size']
      return None
  """END IMG SHAPE"""

  """ADDITIONALS"""
  if True:
    @property
    def _vechicle_list(self):
      return ['truck', 'car', 'motorcycle', 'bus']
  """END ADDITIONALS"""

  """WARMUPS"""
  if True:
    def _lpl_warmup(self):
      self.model_warmup_helper(
        model=self.lpl_model,
        input_shape=(3, *self.lpl_img_shape),
        max_batch_size=self.cfg_max_batch_second_stage,
        model_name=self.get_lpl_filename(),
      )
      return
  """END WARMUPS"""

  """LOAD MODELS"""
  if True:
    def _load_lpl_model(self):
      self.lpl_model, self.graph_config[self.get_lpl_filename()] = self._prepare_ts_model(
        fn_model=self.get_lpl_filename(),
        url=self.get_lpl_url(),
        return_config=True
      )

      img_shape = self.lpl_img_shape

      self.lpl_transforms = self.tv.transforms.Compose([
        PreprocessResizeWithPad(h=img_shape[0], w=img_shape[1], normalize=False),
      ])

      self._lpl_warmup()
      return

    def _second_stage_model_load(self):
      self._load_lpl_model()
      self.load_reader_models()
      return
  """END LOAD MODELS"""

  def _ltrb_to_tlbr(self, ltrb):
    l, t, r, b = ltrb
    return t, l, b, r

  def _pre_process_images(self, images, **kwargs):
    return super(ThYfLp, self)._pre_process_images(
      images=images,
      return_original=True,
      half_original=self.cfg_fp16,  # if this is removed the inputs will be passed as uint8,
      # but the performance will slightly drop
      **kwargs
    )

  def _post_process(self, preds):
    yolo_preds, second_preds = preds
    lst_yolo_results = super(ThYfLp, self)._post_process(preds)
    if second_preds is None:
      return lst_yolo_results

    if len(second_preds) == 4:
      lpl_preds, lpr_preds, lpr_crops, lpr_batch_imgs = second_preds
    else:
      lpl_preds, lpr_preds, lpr_crops, lpr_batch_imgs, lpr_preds_no_stn, lpr_batch_imgs_no_stn = second_preds
    crop_id = -1
    heights, widths = self.cfg_height_offsets, self.cfg_width_offsets
    if lpl_preds is not None and len(lpl_preds) > 0:
      for i, yolo_result in enumerate(lst_yolo_results):
        lp_results = []
        for j, crop_results in enumerate(yolo_result):
          if crop_results['TYPE'] in self._vechicle_list:
            is_valid = lpr_preds[i][j] is not None
            crop_id += is_valid
            crop_results[ctc.LPL_TLBR_POS] = self._ltrb_to_tlbr(
              lpl_preds[i][j][0].int().clamp(min=0).cpu().flatten().numpy())
            crop_results[ctc.LPL_PROB_PRC] = round(float(lpl_preds[i][j][1][0]), 3)
            crop_results[ctc.HEIGHT_OFFSET] = heights[lpl_preds[i][j][2]] if is_valid else 0
            crop_results[ctc.WIDTH_OFFSET] = widths[lpl_preds[i][j][2]] if is_valid else 0

            crop_results = self.maybe_add_extra_data(
              res=crop_results,
              lpr_pred=lpr_preds[i][j],
              lpr_crop=lpr_crops[crop_id],
              lpr_batch_img=lpr_batch_imgs[crop_id],
              lpr_pred_no_stn=lpr_preds_no_stn[i][j] if self.cfg_debug_mode else None,
              lpr_batch_img_no_stn=lpr_batch_imgs_no_stn[crop_id] if self.cfg_debug_mode else None,
              is_valid=is_valid
            )

            lp_results.append({
              self.ct.TLBR_POS: crop_results[ctc.LPL_TLBR_POS],
              self.ct.PROB_PRC: crop_results[ctc.LPL_PROB_PRC],
              self.ct.TYPE: 'plate',
            })
          # endif crop_results['TYPE'] in self.cfg_vechicle_list
        for lp_res in lp_results:
          yolo_result.append(lp_res)
        # endfor j, crop_results in enumerate(yolo_result)
      # endfor i, yolo_result in enumerate(lst_yolo_results)
    # endif

    return lst_yolo_results

  def _aggregate_lpl_second_stage_batch_predict(self, lst_results):
    return self.th.cat([x[0] for x in lst_results]), self.th.cat([x[1] for x in lst_results])

  def _lpl_stage(self, first_stage_out):
    crop_imgs = []
    offsets = [self.th.zeros(size=(*inf.shape[:-1], 2), dtype=self.th.int16, device=self.dev) for inf in first_stage_out]
    lpl_masks = []
    idxs = [self.class_names.index(_class_name) for _class_name in self._vechicle_list]
    self._start_timer("lpl_crop")
    try:
      for i, pred in enumerate(first_stage_out):
        pred_mask = self.th.any(self.th.stack([self.th.eq(pred[:, 5], idx) for idx in idxs], dim=0), dim=0)
        lpl_masks.append(pred_mask.tolist())
        for i_crop in range(pred.shape[0]):
          if pred_mask[i_crop]:
            cropped_image, current_offsets = self._make_crop(
              frame=self.original_input_images[i],
              ltrb=pred[i_crop, :4].unsqueeze(0),
              original_shape=self._lst_original_shapes[i],
              return_offsets=True
            )
            crop_imgs.append(cropped_image)
            offsets[i][i_crop][0] = current_offsets[0]
            offsets[i][i_crop][1] = current_offsets[1]

          # endif pred_mask[i_crop]
        # endfor i_crop in range(pred.shape[0])
      # endfor i, pred in enumerate(first_stage_out)
    finally:
      self._stop_timer("lpl_crop")

    if len(crop_imgs) == 0:
      del offsets
      return None, [], [], []

    lpl_stage_out, _ = self._transform_and_predict(
      model_name='lpl',
      transformation=self.lpl_transforms,
      lst_cropped_images=crop_imgs,
      model=self.lpl_model,
      max_batch_size=self.cfg_max_batch_second_stage,
      aggregate_batch_predict_callback=self._aggregate_lpl_second_stage_batch_predict
    )

    offsets_idxs = []
    for i in range(len(lpl_stage_out[0])):
      # TODO: Razvan
      # TODO: add denormalize parameter
      lpl_stage_out[0][i] = inverse_recompute_relative_y(
        old_size=crop_imgs[i].shape[1:],
        new_size=self.lpl_img_shape,
        new_y=lpl_stage_out[0][i]
      )
      height, width = crop_imgs[i].shape[1:]
      lpl_stage_out[0][i][0] *= width  # left
      lpl_stage_out[0][i][1] *= height  # top
      lpl_stage_out[0][i][2] *= width  # right
      lpl_stage_out[0][i][3] *= height  # bottom

      # heuristic to adjust license plate localization
      h = (lpl_stage_out[0][i][3] - lpl_stage_out[0][i][1]).clamp(min=1)
      w = (lpl_stage_out[0][i][2] - lpl_stage_out[0][i][0]).clamp(min=1)
      ratio = (h / w)
      offset_idx = sum([(ratio > hw_threshold) for hw_threshold in self.cfg_hw_thresholds])
      # if not isinstance(offset_idx, int):
      #   offset_idx = offset_idx.int()
      height_offset = self.cfg_height_offsets[offset_idx]
      width_offset = self.cfg_width_offsets[offset_idx]
      offsets_idxs.append(offset_idx)

      if height_offset > 0:
        lpl_stage_out[0][i][1] = (lpl_stage_out[0][i][1] - height_offset).clamp(min=0)
        lpl_stage_out[0][i][3] = (lpl_stage_out[0][i][3] - height_offset).clamp(min=0)
      if width_offset > 0:
        lpl_stage_out[0][i][2] = (lpl_stage_out[0][i][2] + width_offset).clamp(min=0)
    # endfor i in range(len(lpl_stage_out[0]))

    lpl_results = []
    k = 0
    self._start_timer("lpl_agg_res")
    lpl_cumsums = [self.np.cumsum(lpl_masks[i]) for i in range(len(first_stage_out))]
    for i in range(len(first_stage_out)):
      lpl_results.append([
        (
          (lpl_stage_out[0][lpl_cumsums[i][j] - 1 + k].squeeze(-1) + offsets[i][j].tile(2)),
          lpl_stage_out[1][lpl_cumsums[i][j] - 1 + k],
          offsets_idxs[lpl_cumsums[i][j] - 1 + k]
        ) if lpl_masks[i][j] else None
        for j in range(len(lpl_masks[i]))
      ])
      k += sum(lpl_masks[i])
    self._stop_timer("lpl_agg_res")
    del offsets

    return lpl_results, lpl_stage_out, crop_imgs, lpl_masks

  def _second_stage_classifier(self, first_stage_out, th_inputs):
    empty_result = ([], [], [], []) if not self.cfg_debug_mode else ([], [], [], [], [], [])
    lpl_results, lpl_stage_out, lpr_inputs, lpl_masks = self._lpl_stage(first_stage_out=first_stage_out)

    if lpl_results is None:
      # In case we don't have any detections of vehicles there is no need for the lpr stage
      return empty_result

    out = self._lpr_stage(lpl_stage_out=lpl_stage_out, lpr_inputs=lpr_inputs, lpl_masks=lpl_masks)

    if self.cfg_debug_mode:
      lprnet_results, lprnet_crops, lprnet_batch_imgs, lprnet_no_stn_results, lprnet_no_stn_batch_imgs = out
    else:
      lprnet_results, lprnet_crops, lprnet_batch_imgs = out

    if len(lprnet_results) == 0:
      # In case we don't have any detections of license plates with high
      # enough confidence there is no need for the lpr stage
      return empty_result

    second_stage_results = (lpl_results, lprnet_results, lprnet_crops, lprnet_batch_imgs)
    if self.cfg_debug_mode:
      second_stage_results += (lprnet_no_stn_results, lprnet_no_stn_batch_imgs)

    return second_stage_results

