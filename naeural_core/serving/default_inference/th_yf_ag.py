from naeural_core.serving.default_inference.th_yf8l import ThYf8l as BaseServingProcess
from naeural_core.local_libraries.nn.th.image_dataset_stage_preprocesser import PreprocessResizeWithPad


_CONFIG = {
  **BaseServingProcess.CONFIG,

  "DEBUG_TIMERS": False,
  'MAX_BATCH_SECOND_STAGE': 20,

  'COVERED_SERVERS': ['th_yf8l'],

  #####################
  "AGE_GENDER_URL": "minio:AgeGender/AGE_GENDER_V3_12_bs1.ths",
  "AGE_GENDER_FILENAME": "AGE_GENDER_V3_12_bs1.ths",

  "AGE_GENDER_URL_FP16": "minio:AgeGender/AGE_GENDER_V3_12_fp16_bs1.ths",
  "AGE_GENDER_FILENAME_FP16": "AGE_GENDER_V3_12_fp16_bs1.ths",

  "AGE_GENDER_ONNX_URL" : None,
  "AGE_GENDER_ONNX_FILENAME" : None,
  "AGE_GENDER_FORCE_BACKEND" : None,

  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
  },
}


AGE_CLASSES = [
  'child[0-11]',
  'teen[12-19]',
  'young[20-39]',
  'middle[40-59]',
  'senior[60-100]'
]


GENDER_CLASSES = [
  'male',
  'female'
]


class ThYfAg(BaseServingProcess):
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(ThYfAg, self).__init__(**kwargs)
    self._has_second_stage_classifier = True
    return

  @property
  def age_gender_img_shape(self):
    return self.graph_config[self.get_age_gender_filename()]['image_size']

  def get_age_gender_url(self):
    if self.cfg_fp16:
      return self.cfg_age_gender_url_fp16
    return self.cfg_age_gender_url

  def get_age_gender_filename(self):
    if self.cfg_fp16:
      return self.cfg_age_gender_filename_fp16
    return self.cfg_age_gender_filename

  def age_gender_warmup(self):
    self.model_warmup_helper(
      model=self.age_gender_model,
      input_shape=(3, *self.age_gender_img_shape),
      max_batch_size=self.cfg_max_batch_second_stage,
      model_name=self.get_age_gender_filename()
    )
    return

  def load_age_gender_model(self):
    self.age_gender_model, self.graph_config[self.get_age_gender_filename()] = self._prepare_ts_model(
      fn_model=self.get_age_gender_filename(),
      url=self.get_age_gender_url(),
      return_config=True
    )
    img_shape = self.age_gender_img_shape
    self.age_gender_transforms = self.tv.transforms.Compose([
      PreprocessResizeWithPad(h=img_shape[0], w=img_shape[1], normalize=False),
    ])
    self.age_gender_warmup()
    return

  def _second_stage_model_load(self):
    self.load_age_gender_model()
    return

  def _pre_process_images(self, images, **kwargs):
    return super(ThYfAg, self)._pre_process_images(
      images=images,
      return_original=True,
      half_original=self.cfg_fp16,
      **kwargs
    )

  def _post_process(self, preds):
    yolo_preds, second_preds = preds
    lst_yolo_results = super(ThYfAg, self)._post_process(preds)
    if second_preds is None:
      return lst_yolo_results
    for i, yolo_result in enumerate(lst_yolo_results):
      for j, crop_results in enumerate(yolo_result):
        if crop_results['TYPE'] == 'person':
          crop_results["GENDER"] = GENDER_CLASSES[int(second_preds[i][j][0])]
          crop_results["AGE"] = AGE_CLASSES[int(second_preds[i][j][1])]
        # endif
      # endfor
    # endfor
    return lst_yolo_results

  def _aggregate_age_gender_second_stage_batch_predict(self, lst_results):
    return [self.th.cat([x[1] for x in lst_results]), self.th.cat([x[3] for x in lst_results])]

  def transform_and_predict(
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
    return preds

  def _second_stage_classifier(self, pred_nms, th_inputs):
    crop_imgs = []
    age_gender_masks = []
    self._start_timer("ag_crop")
    person_idx = self.class_names.index('person')
    try:
      for i, pred in enumerate(pred_nms):
        pred_mask = (pred[:, 5] == person_idx)
        age_gender_masks.append(pred_mask.tolist())
        for i_crop in range(pred.shape[0]):
          if pred_mask[i_crop]:
            crop_imgs.append(
              self._make_crop(
                self.original_input_images[i],
                ltrb=pred[i_crop, :4].unsqueeze(0),
                original_shape=self._lst_original_shapes[i],
                return_offsets=False
              )
            )
          # endif pred_mask[i_crop]
        # endfor i_crop in range(pred.shape[0])
      # endfor i, pred in enumerate(first_stage_out)
    finally:
      self._stop_timer("ag_crop")

    if len(crop_imgs) == 0:
      return []

    age_gender_preds = self.transform_and_predict(
      model_name='ag',
      transformation=self.age_gender_transforms,
      lst_cropped_images=crop_imgs,
      model=self.age_gender_model,
      max_batch_size=self.cfg_max_batch_second_stage,
      aggregate_batch_predict_callback=self._aggregate_age_gender_second_stage_batch_predict
    )
    age_gender_preds[0] = age_gender_preds[0] > 0.5
    age_gender_preds[1] = self.th.sum(age_gender_preds[1] > 0.5, axis=1)
    np_preds = (age_gender_preds[0].cpu().numpy(), age_gender_preds[1].cpu().numpy())

    age_gender_results = []
    k = 0
    self._start_timer("ag_agg_res")
    cumsums = [self.np.cumsum(mask) for mask in age_gender_masks]
    for i in range(len(pred_nms)):
      age_gender_results.append([(
          np_preds[0][cumsums[i][j] - 1 + k],
          np_preds[1][cumsums[i][j] - 1 + k]
        )
        if age_gender_masks[i][j] else None
        for j in range(len(age_gender_masks[i]))
      ])
      k += sum(age_gender_masks[i])
    self._stop_timer("ag_agg_res")

    return age_gender_results

