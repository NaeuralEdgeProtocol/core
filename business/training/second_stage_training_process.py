from core.business.training.general_training_process import GeneralTrainingProcessPlugin
from core.business.training.general_training_process import _CONFIG as BASE_CONFIG

_CONFIG = {
  **BASE_CONFIG,
  'VALIDATION_RULES' : {
    **BASE_CONFIG['VALIDATION_RULES'],
  },

  'AUTO_DEPLOY' : {},
}


class SecondStageTrainingProcessPlugin(GeneralTrainingProcessPlugin):
  def _prepare_inference_plugin_config(self) -> dict:
    return {
      'SECOND_STAGE_MODEL_ARCHITECTURE_PATH' : self._training_output['METADATA']['MODEL_ARCHITECTURE_PATH'],
      'SECOND_STAGE_MODEL_HYPERPARAMS' : self._training_output['STATUS']['BEST']['model_kwargs'],
      'SECOND_STAGE_CLASS_NAMES' : list(self._training_output['METADATA']['CLASSES'].keys()),
      'SECOND_STAGE_MODEL_URL' : self._model_uri['URL'], # TODO self._model_uri['CLOUD_PATH']
      'SECOND_STAGE_INPUT_SIZE': self._training_output['STATUS']['BEST']['dct_grid_option']['input_size'],
      'SECOND_STAGE_TARGET_CLASS' : self._training_output['METADATA']['FIRST_STAGE_TARGET_CLASS'],
      'SECOND_STAGE_PREPROCESS_DEFINITIONS': self._training_output['METADATA']['INFERENCE_PREPROCESS_DEFINITIONS'],
    }

  def _auto_deploy(self):
    box_auto_deploy = self.cfg_auto_deploy.get('BOX_ID', None)
    streams = self.cfg_auto_deploy.get('STREAMS', [])

    for s in streams:
      if 'PLUGINS' not in s:
        s['PLUGINS'] = []

      s['PLUGINS'].append({
        'SIGNATURE' : 'second_stage_detection',
        'INSTANCES' : [
          {
            'INSTANCE_ID' : self.log.now_str(),
            'AI_ENGINE' : 'custom_second_stage_detector',
            'OBJECT_TYPE' : [self._training_output['METADATA']['FIRST_STAGE_TARGET_CLASS']],
            'SECOND_STAGE_DETECTOR_CLASSES' : list(self._training_output['METADATA']['CLASSES'].keys()),
            'STARTUP_AI_ENGINE_PARAMS' : {
              'CUSTOM_DOWNLOADABLE_MODEL_URL' : self._inference_config_uri['URL'], # TODO self._inference_config_uri['CLOUD_PATH']
              'MODEL_INSTANCE_ID' : self._training_output['METADATA']['MODEL_NAME'],
            }
          }
        ]
      })

      self.cmdapi_start_stream_by_config_on_other_box(box_id=box_auto_deploy, config_stream=s)
    #endfor

    return
