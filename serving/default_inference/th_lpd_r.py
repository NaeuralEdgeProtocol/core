from core.constants import CarAccess as ctc
from core.serving.default_inference.th_lpd import ThLpd as BaseServingProcess
from core.serving.mixins_base.plate_read_mixin import PLATE_READ_CONFIG, CHARS, _PlateReadMixin


_CONFIG = {
  **BaseServingProcess.CONFIG,
  **PLATE_READ_CONFIG,
  'MAX_BATCH_SECOND_STAGE': 10,

  'COVERED_SERVERS': ['th_lpd'],

  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
    **PLATE_READ_CONFIG['VALIDATION_RULES'],
  },
}

DEBUG_COVERED_SERVERS = False


class ThLpdR(BaseServingProcess, _PlateReadMixin):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    self._model_hyperparams = {}
    super(ThLpdR, self).__init__(**kwargs)

    self._has_second_stage_classifier = True
    return

  def _startup(self):
    super(ThLpdR, self)._startup()
    self.CHARS = self.np.array(CHARS)
    return

  """LOAD MODELS"""
  if True:
    def _second_stage_model_load(self):
      self.load_reader_models()
      return
  """END LOAD MODELS"""

  def _pre_process_images(self, images, **kwargs):
    return super(ThLpdR, self)._pre_process_images(
      images=images,
      return_original=True,
      half_original=self.cfg_fp16,  # if this is removed the inputs will be passed as uint8,
      # but the performance will slightly drop
      **kwargs
    )

  def _post_process(self, preds):
    yolo_preds, second_preds = preds
    lst_yolo_results = super(ThLpdR, self)._post_process(preds)
    if second_preds is None:
      return lst_yolo_results

    if len(second_preds) == 3:
      lpr_preds, lpr_crops, lpr_batch_imgs = second_preds
    else:
      lpr_preds, lpr_crops, lpr_batch_imgs, lpr_preds_no_stn, lpr_batch_imgs_no_stn = second_preds
    crop_id = -1
    for i, yolo_result in enumerate(lst_yolo_results):
      for j, crop_results in enumerate(yolo_result):
        crop_id += 1
        crop_results[ctc.LPL_TLBR_POS] = crop_results[self.consts.TLBR_POS]
        crop_results[ctc.LPL_PROB_PRC] = crop_results[self.consts.PROB_PRC]
        crop_results[ctc.HEIGHT_OFFSET] = 0
        crop_results[ctc.WIDTH_OFFSET] = 0
        crop_results = self.maybe_add_extra_data(
          res=crop_results,
          lpr_pred=lpr_preds[i][j],
          lpr_crop=lpr_crops[crop_id],
          lpr_batch_img=lpr_batch_imgs[crop_id],
          lpr_pred_no_stn=lpr_preds_no_stn[i][j] if self.cfg_debug_mode else None,
          lpr_batch_img_no_stn=lpr_batch_imgs_no_stn[crop_id] if self.cfg_debug_mode else None,
          is_valid=True
        )
      # endfor j, crop_results in enumerate(yolo_result)
    # endfor i, yolo_result in enumerate(lst_yolo_results)

    return lst_yolo_results

  def _lpr_stage2(self, first_stage_out):
    stn_crop_imgs = []
    lprnet_crop_imgs = []
    self._start_timer("lprnet_crop")
    try:
      for i, pred in enumerate(first_stage_out):
        for i_crop in range(pred.shape[0]):
          l, t, r, b = self.scale_coords(
            img1_shape=self.cfg_input_size,
            coords=self.deepcopy(pred[i_crop, :4].unsqueeze(0)),
            img0_shape=self._lst_original_shapes[i]
          ).int().flatten().clamp(min=0)

          new_l, new_t, new_r, new_b = self.stn_expand_borders(l, t, r, b)
          stn_crop_imgs.append(
            self.tv.transforms.functional.crop(
              img=self.original_input_images[i],
              left=new_l.int(),
              top=new_t.int(),
              width=(new_r - new_l).clamp(min=4).int() + 1,
              height=(new_b - new_t).clamp(min=4).int() + 1,
            )
          )
          lprnet_crop_imgs.append(
            self.tv.transforms.functional.crop(
              img=self.original_input_images[i],
              left=l.int(),
              top=t.int(),
              width=(r - l).clamp(min=4).int() + 1,
              height=(b - t).clamp(min=4).int() + 1,
            )
          )
        # endfor i_crop in range(pred.shape[0])
      # endfor i, pred in enumerate(first_stage_out)
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
    k = 0
    self._start_timer("lprnet_agg_res")
    for i in range(len(first_stage_out)):
      res = []
      res_no_stn = []
      for j in range(first_stage_out[i].shape[0]):
        res.append(lprnet_preds[k])
        if self.cfg_debug_mode:
          res_no_stn.append(lprnet_no_stn_preds[k])

        k += 1
      # endfor j in range(len(lpl_masks[i]))
      lprnet_stn_results.append(res)
      if self.cfg_debug_mode:
        lprnet_no_stn_results.append(res_no_stn)
    # endfor i in range(len(lpl_masks))

    self._stop_timer("lprnet_agg_res")

    res = (lprnet_stn_results, lprnet_crop_imgs, batch_stn_imgs)
    if self.cfg_debug_mode:
      res += (lprnet_no_stn_results, batch_no_stn_imgs)

    return res

  def _second_stage_classifier(self, first_stage_out, th_inputs):
    return self._lpr_stage2(first_stage_out=first_stage_out)
