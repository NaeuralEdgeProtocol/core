from naeural_core.serving.default_inference.th_yf_mspn import ThYfMspn as BaseServingProcess


_CONFIG = {
  **BaseServingProcess.CONFIG,

  "DEBUG_TIMERS": False,

  'COVERED_SERVERS': [
    *BaseServingProcess.CONFIG['COVERED_SERVERS'],
    'th_yf8l'
  ],

  #####################
  # "POSE_CLF_URL": "minio:POSE_ALERT/POSE_ALERT_V15_DS3_bs1.ths",
  # "POSE_CLF_FILENAME": "POSE_ALERT_V15_DS3_bs1.ths",

  # "POSE_CLF_URL": "minio:POSE_ALERT/POSE_ALERT_V15_DS3_it532_bs1.ths",
  # "POSE_CLF_FILENAME": "POSE_ALERT_V15_DS3_it532_bs1.ths",

  "POSE_CLF_URL": "minio:POSE_ALERT/POSE_ALERT_V15_DS4_bs1.ths",
  "POSE_CLF_FILENAME": "POSE_ALERT_V15_DS4_bs1.ths",

  "POSE_CLF_CLASSES": [
    'DISTRESS',
    'FALLEN',
    'OTHER'
  ],

  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
  },
}
DEBUG_COVERED_SERVERS = False


class ThYfMspnPc(BaseServingProcess):
  CONFIG = _CONFIG

  def pose_clf_warmup(self):
    self.model_warmup_helper(
      model=self.pose_clf_model,
      input_shape=self.graph_config[self.cfg_pose_clf_filename]['input_dim'],
      max_batch_size=self.cfg_max_batch_second_stage,
      model_name='pose_clf'
    )

  def load_pose_clf_model(self):
    self.pose_clf_model, self.graph_config[self.cfg_pose_clf_filename] = self._prepare_ts_model(
      url=self.cfg_pose_clf_url,
      fn_model=self.cfg_pose_clf_filename,
      return_config=True
    )
    self.pose_clf_warmup()
    return

  def _second_stage_model_load(self):
    super(ThYfMspnPc, self)._second_stage_model_load()
    self.load_pose_clf_model()
    return

  def _post_process(self, preds):
    yolo_preds, second_preds = preds
    move_preds, pc_preds = second_preds if second_preds is not None else (None, None)
    lst_yolo_results = super(ThYfMspnPc, self)._post_process((yolo_preds, move_preds))
    if second_preds is not None:
      for i, yolo_result in enumerate(lst_yolo_results):
        for j, crop_results in enumerate(yolo_result):
          if crop_results['TYPE'] == 'person':
            crop_results["POSE_TYPE"] = {
              self.cfg_pose_clf_classes[it]: pc_preds[i][j][it]
              for it in range(len(pc_preds[i][j]))
            }
          # endif
        # endfor
      # endfor
    # endif
    return lst_yolo_results

  def _second_stage_classifier(self, pred_nms, th_inputs):
    mspn_preds, masks = self.mspn_stage(pred_nms=pred_nms, th_inputs=th_inputs)
    if len(mspn_preds) == 0:
      return None
    results = []

    self._start_timer("p_clf_p_b{}_{}".format(len(mspn_preds), self.cfg_max_batch_second_stage))
    pc_preds = self._batch_predict(
      prep_inputs=mspn_preds,
      model=self.pose_clf_model,
      batch_size=self.cfg_max_batch_second_stage,
      aggregate_batch_predict_callback=self._aggregate_second_stage_batch_predict
    )
    self._stop_timer("p_clf_p_b{}_{}".format(len(mspn_preds), self.cfg_max_batch_second_stage))

    self._start_timer("pproc")
    move_preds = self._post_process_mspn(mspn_preds)
    self._stop_timer("pproc")

    self._start_timer("agg_res")
    move_preds_c = move_preds.cpu().numpy()
    pc_preds_c = pc_preds.cpu().numpy()
    l_idxs = self.compute_pred_indexes(masks)
    for cpu_preds in [move_preds_c, pc_preds_c]:
      pred_results = []
      for idxs in l_idxs:
        pred_results.append([
          self.np.round(cpu_preds[idx], 3) if idx is not None else None
          for idx in idxs
        ])
      results.append(pred_results)
    self._stop_timer("agg_res")
    return results
