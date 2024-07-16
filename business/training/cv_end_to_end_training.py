"""
{
	"NAME" : "auto_full_process",
	"TYPE" : "void",
	"PLUGINS" : [
		{
			"SIGNATURE" : "cv_end_to_end_training",
			"INSTANCES" : [
				{
					"INSTANCE_ID" : "default",
					"OBJECTIVE_NAME" : "weapons",
					"GENERAL_DETECTOR_OBJECT_TYPE" : "person",
					"GENERAL_DETECTOR_AI_ENGINE" : "general_detector",

					"DATA" : {
						"SOURCES" : [
							{
								"NAME" : "terasa",
								"TYPE" : "VideoStream",
								"URL"  : "__URL__",
								"LIVE_FEED" : true,
								"CAP_RESOLUTION" : 0.5,
								"RECONNECTABLE" : true,
								"STREAM_CONFIG_METADATA" : {
									"INTERVALS" : {
										"ziua" : ["10:00", "17:00"],
										"noaptea" : ["21:00", "23:59"]
									}
								}
							},

							{
								"NAME" : "openspace_est",
								"TYPE" : "VideoStream",
								"URL"  : "__URL__",
								"LIVE_FEED" : true,
								"RECONNECTABLE" : true,
								"CAP_RESOLUTION" : 0.5,
								"STREAM_CONFIG_METADATA" : {
									"INTERVALS" : {
										"ziua" : ["10:00", "17:00"],
										"noaptea" : ["21:00", "23:59"]
									}
								}
							}
						],

						"CROP_PLUGIN_PARAMS" : {
							"REPORT_PERIOD" : 60,
							"ALIVE_UNTIL" : "2022-08-10 14:55"
						},

						"CLOUD_PATH" : "DATASETS/"
					},

					"TRAINING" : {
						"BOX_ID" : "hidra-training",
						"DEVICE_LOAD_DATA" : "cuda:3",
						"DEVICE_TRAINING"  : "cuda:3",
						"TRAINING_PIPELINE_SIGNATURE" : "weapons",
						"GRID_SEARCH" : {},
						"BATCH_SIZE" : 64,
						"EPOCHS" : 4
					},

					"AUTO_DEPLOY" : {
						"STREAMS" : [
							{
								"NAME" : "terasa",
								"TYPE" : "VideoStream",
								"URL"  : "__URL__",
								"LIVE_FEED" : true,
								"CAP_RESOLUTION" : 20,
								"RECONNECTABLE" : true
							}
						]
					}
				}
			]
		}
	]
}
  
"""

from core.business.base import BasePluginExecutor as BasePlugin

__VER__ = '0.1.0.0'

DEFAULT_DATA_CONFIG = {
  'SOURCES': [],
  "CROP_PLUGIN_PARAMS": {
    "REPORT_PERIOD": 10
  },
  "COLLECT_UNTIL": None,
  "CLOUD_PATH": "DATASETS/",
  "TRAIN_SIZE": 0.8,
}

DEFAULT_TRAIN_CONFIG = {
  'BOX_ID': None,
  'DEVICE_LOAD_DATA': None,  # None for no preloading data, string for device otherwise
  'DEVICE_TRAINING': 'cuda:0',
  'TRAINING_PIPELINE_SIGNATURE': 'custom',
  'MODEL_ARCHITECTURE': 'BASIC_CLASSIFIER',
  'GRID_SEARCH': {},
  'BATCH_SIZE': 8,
  'EPOCHS': 4,
}

_CONFIG = {
  **BasePlugin.CONFIG,
  'OBJECTIVE_NAME': None,
  'GENERAL_DETECTOR_OBJECT_TYPE': ['person'],
  'DATA': {},
  'TRAINING': {},
  'AUTO_DEPLOY': {},
  "CLASSES": {},
  "DESCRIPTION": "",
  "ALLOW_EMPTY_INPUTS": True,

  'PLUGIN_LOOP_RESOLUTION': 1/5,  # once at 5s

  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES'],
  },
}


class CVEndToEndTrainingPlugin(BasePlugin):
  def on_init(self):
    super(CVEndToEndTrainingPlugin, self).on_init()
    self.__executed = False
    self._one_time_commands_performed = False
    return

  def get_auto_deploy(self):
    auto_deploy = self.cfg_auto_deploy
    if 'BOX_ID' not in auto_deploy or "NODE_ADDRESS" not in auto_deploy:
      auto_deploy['BOX_ID'] = self.ee_id
      auto_deploy['NODE_ADDRESS'] = self.node_addr
    if 'STREAMS' not in auto_deploy:
      auto_deploy['STREAMS'] = []
    return auto_deploy

  def get_data(self):
    data_cfg = {
      **DEFAULT_DATA_CONFIG,
      **self.cfg_data
    }
    return data_cfg

  def get_training(self):
    training_cfg = {
      **DEFAULT_TRAIN_CONFIG,
      **self.cfg_training
    }
    return training_cfg

  def classes_dict(self):
    """
    The `CLASSES` parameter should be a dictionary with the following structure:
    {
      "class_1": "Class 1 description",
      "class_2": "Class 2 description",
      ...
    }
    Returns
    -------
    dict : The classes dictionary
    """
    return self.cfg_classes

  def classes_list(self):
    """
    Utility function to return the classes as a list
    Returns
    -------
    list : The classes list
    """
    return list(self.cfg_classes.keys())


  """DATA SECTION"""
  if True:
    @property
    def data_sources(self):
      return self.get_data().get('SOURCES', [])

    @property
    def data_crop_plugin_params(self):
      return self.get_data().get('CROP_PLUGIN_PARAMS', {})

    @property
    def data_cloud_path(self):
      return self.get_data().get('CLOUD_PATH', '')

    @property
    def data_train_size(self):
      return self.get_data().get('TRAIN_SIZE', 0.8)
  """END DATA SECTION"""

  """TRAINING SECTION"""
  if True:
    @property
    def training_box_id(self):
      return self.get_training().get('BOX_ID')

    @property
    def training_model_architecture(self):
      return self.get_training().get('MODEL_ARCHITECTURE')

    @property
    def training_device_load_data(self):
      return self.get_training().get('DEVICE_LOAD_DATA')

    @property
    def device_training(self):
      return self.get_training().get('DEVICE_TRAINING')

    @property
    def training_pipeline_signature(self):
      return self.get_training().get('TRAINING_PIPELINE_SIGNATURE')

    @property
    def training_grid_search(self):
      return self.get_training().get('GRID_SEARCH')

    @property
    def training_batch_size(self):
      return self.get_training().get('BATCH_SIZE')

    @property
    def training_epochs(self):
      return self.get_training().get('EPOCHS')
  """END TRAINING SECTION"""

  @property
  def _dataset_object_name(self):
    return self.os_path.join(self.data_cloud_path, self.cfg_objective_name)

  """CONFIG PIPELINE SECTION"""
  if True:
    def _configured_download_dataset_pipeline(self):
      config = {
        "NAME" : "download_dataset_{}".format(self.cfg_objective_name),
        "TYPE" : "VOID",
        "PLUGINS" : [
          {
            "SIGNATURE" : "minio_download_dataset",
            "INSTANCES" : [
              {
                "INSTANCE_ID" : "default",
                "DATASET_OBJECT_NAME" : self.os_path.join(self.data_cloud_path, self.cfg_objective_name + '_RAW'),
                "DATASET_LOCAL_PATH" : "<TO_BE_COMPLETED>"
              }
            ]
          }
        ]
      }
      return config

    def _configured_upload_dataset_pipeline(self):
      config = {
        "NAME": "upload_dataset_{}".format(self.cfg_objective_name),
        "TYPE": "VOID",
        "PLUGINS": [
          {
            "SIGNATURE": "minio_upload_dataset",
            "INSTANCES": [
              {
                "INSTANCE_ID": "default",
                "DATASET_OBJECT_NAME": self.os_path.join(self.data_cloud_path, self.cfg_objective_name),
                "DATASET_LOCAL_PATH": "<TO_BE_COMPLETED>"
              }
            ]
          }
        ]
      }
      return config

    def process_data_gather_config(self, config):
      """
      Fill in the gaps in the data gather config
      """
      return {
        **config,
        'CAP_RESOLUTION': config.get('CAP_RESOLUTION', 1),
      }

    def _configured_metastream_collect_data(self):
      if not isinstance(self.cfg_general_detector_object_type, list):
        object_type = [self.cfg_general_detector_object_type]
      else:
        object_type = self.cfg_general_detector_object_type

      cfg_instance = {
        "INSTANCE_ID": f"gather_data_{self.cfg_objective_name}",
        "OBJECTIVE_NAME": self.cfg_objective_name,
        "DESCRIPTION": self.cfg_description,
        "CLOUD_PATH": self.data_cloud_path,
        "OBJECT_TYPE": object_type,
        'CLASSES': self.classes_dict(),
        'TRAIN_SIZE': self.data_train_size,
        **self.data_crop_plugin_params,
      }

      current_ai_engine = self.cfg_ai_engine
      if current_ai_engine is not None and len(current_ai_engine) > 0 and cfg_instance.get('AI_ENGINE', None) is None:
        cfg_instance['AI_ENGINE'] = current_ai_engine

      config = {
        "NAME" : "metastream_collect_data_{}".format(self.cfg_objective_name),
        "TYPE" : "MetaStream",
        "COLLECTED_STREAMS": [x['NAME'] for x in self.data_sources],
        "PLUGINS": [
          {
            "SIGNATURE": "cv_crop_training_data",
            "INSTANCES": [cfg_instance]
          }
        ]
      }
      return config

    def _configured_training_pipeline(self):
      config = {
        "NAME" : "training_{}".format(self.cfg_objective_name),
        "TYPE" : "minio_dataset",
        "STREAM_CONFIG_METADATA": {
          "DATASET_OBJECT_NAME": self._dataset_object_name,
        },
        "PLUGINS" : [
          {
            "SIGNATURE" : "second_stage_training_process",
            "INSTANCES" : [
              {
                "INSTANCE_ID": f"training_{self.cfg_objective_name}",
                'AI_ENGINE': f'th_training?{self.cfg_objective_name}',
                'STARTUP_AI_ENGINE_PARAMS': {
                  'PIPELINE_SIGNATURE': self.training_pipeline_signature,
                  'PIPELINE_CONFIG': {
                    'CLASSES': self.classes_list(),
                    'MODEL_ARCHITECTURE': self.training_model_architecture,
                    'MODEL_NAME': self.cfg_objective_name,
                    'PRELOAD_DATA': self.training_device_load_data is not None,
                    'DEVICE_LOAD_DATA': self.training_device_load_data,
                    'DEVICE_TRAINING': self.device_training,
                    'GRID_SEARCH': self.training_grid_search,
                    'BATCH_SIZE': self.training_batch_size,
                    'EPOCHS': self.training_epochs,
                    'FIRST_STAGE_TARGET_CLASS': self.cfg_general_detector_object_type,
                  },
                },
                'AUTO_DEPLOY': self.get_auto_deploy(),
                "DESCRIPTION": self.cfg_description,
                "OBJECTIVE_NAME": self.cfg_objective_name,
              }
            ]
          }
        ]
      }
      return config
  """END CONFIG PIPELINE SECTION"""

  def _process(self):
    if not self.__executed:
      self.P(f"Starting end to end training for {self.cfg_objective_name}")
      for config in self.data_sources:
        processed_config = self.process_data_gather_config(config)
        self.P(f"Starting pipeline {config['NAME']} for data acquisition")
        self.cmdapi_start_stream_by_config_on_current_box(config_stream=processed_config)
      # endfor data sources
      self.P(f"Starting metastream for data collection")
      self.cmdapi_start_metastream_by_config_on_current_box(config_metastream=self._configured_metastream_collect_data())
  
      if not self._one_time_commands_performed:
        training_box_addr = self.net_mon.network_node_addr(self.training_box_id)
        self.P(f"Starting training pipeline for {self.cfg_objective_name} on {training_box_addr}")
        self._cmdapi_start_stream_by_config(config_stream=self._configured_training_pipeline(), node_address=training_box_addr)
        config_download = self._configured_download_dataset_pipeline()
        # config_upload = self._configured_upload_dataset_pipeline()
        now_str = self.now_str()
        self.diskapi_save_json_to_output(dct=config_download, filename=f"{now_str}_{config_download['NAME']}.json")
        # self.diskapi_save_json_to_output(dct=config_upload, filename=f"{now_str}_{config_upload['NAME']}.json")
        self._one_time_commands_performed = True
      # endif one time commands performed
      self.__executed = True
    # endif not executed
    return
