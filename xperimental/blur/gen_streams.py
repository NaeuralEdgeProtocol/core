import os
import random

from naeural_core import Logger

def get_template_stream_uninode(cfg_name, cfg_url, cfg_use_local_system):
  return {
    "CAP_RESOLUTION": 50,
    "INITIATOR_ID": "test_1234",
    "LIVE_FEED": False,
    "NAME": cfg_name,
    "PLUGINS": [
        {
            "INSTANCES": [
                {
                    "AI_ENGINE": "general_detector",
                    "COORDS": "NONE",
                    "DETECTIONS_RESOURCE": None,
                    "FPS_OUTPUT": 20,
                    "INSTANCE_ID": "UN_FULL_BLUR",
                    "OBJECT_TYPE": ["person", "car", "bus", "truck"],
                    "PRC_PERSON_BLUR": 1,
                    "PRC_VEHICLES_BLUR": 1,
                    "RESEND_GOLDEN": False,
                    "SELECTIVE_ZONES": {},
                    "TRACKING_MODE": 0,
                    "TRACKING_OBJECTS": {},
                    "USE_LOCAL_SYSTEM_TO_SAVE": cfg_use_local_system
                }
            ],
            "SIGNATURE": "BLUR_MOVIE_02"
        }
    ],
    "RECONNECTABLE": False,
    "STREAM_CONFIG_METADATA": {
      "NOTIFY_DOWNLOAD_DOWNSTREAM": True,
    },
    "STREAM_WINDOW": 1,
    "TYPE": "VideoFile",
    "URL": cfg_url
  }

def get_template_stream_multinode(cfg_name, cfg_url, cfg_use_local_system, cfg_video_file_ext, cfg_workers):
  return {
    "CAP_RESOLUTION": 50,
    "INITIATOR_ID": "test_1234",
    "LIVE_FEED": False,
    "NAME": cfg_name,
    "RECONNECTABLE": False,
    "PLUGINS": [
        {
            "INSTANCES": [
                {
                    "WORKER_AI_ENGINE": "general_detector",
                    "COORDS": "NONE",
                    "DETECTIONS_RESOURCE": None,
                    "FPS_OUTPUT": 20,
                    "INSTANCE_ID": "MN_FULL_BLUR_0",
                    "OBJECT_TYPE": ["person", "car", "bus", "truck"],
                    "PRC_PERSON_BLUR": 1,
                    "PRC_VEHICLES_BLUR": 1,
                    "RESEND_GOLDEN": False,
                    "SELECTIVE_ZONES": {},
                    "TRACKING_MODE": 0,
                    "TRACKING_OBJECTS": {},
                    "USE_LOCAL_SYSTEM_TO_SAVE": cfg_use_local_system
                }
            ],
            "SIGNATURE": "BLUR_MOVIE_MAP_REDUCE_02"
        }
    ],
    "STREAM_CONFIG_METADATA": {
        "NOTIFY_DOWNLOAD_DOWNSTREAM": True,
        "USE_LOCAL_SYSTEM_TO_SAVE": cfg_use_local_system,
        "VIDEO_FILE_EXTENSION": cfg_video_file_ext,
        "WORKERS": cfg_workers,
    },
    "STREAM_WINDOW": 1,
    "TYPE": "video_file_map_reduce",
    "URL": cfg_url
}

if __name__ == '__main__':
  log = Logger(
    lib_name='EE_TST',
    base_folder='.',
    app_folder='_local_cache',
    config_file='config_startup.txt',
    max_lines=1000,
    TF_KERAS=False
  )

  dct_movies = {
    "NAME" : {
      "URL": "__URL__",
      "PATH": "__PATH__",
      }
  }

  lst_workers   = ["NODE"]
  nr_workers    = [1,2,4]
  lst_storages  = ["minio", "local"]

  for movie, source_movie in dct_movies.items():
    for w in nr_workers:
      for storage in lst_storages:

        if storage == 'minio':
          url = source_movie['URL']
          use_local_system_to_save = False
        else:
          url = source_movie['PATH']
          use_local_system_to_save = True
        #endif

        name = "{}_{}_{}w".format(movie, storage, w)
        video_file_ext = os.path.splitext(source_movie['PATH'])[1]
        workers = lst_workers[:w]

        if w == 1:
          config = get_template_stream_uninode(cfg_name=name, cfg_url=url, cfg_use_local_system=use_local_system_to_save)
        else:
          config = get_template_stream_multinode(
            cfg_name=name, cfg_url=url,
            cfg_use_local_system=use_local_system_to_save,
            cfg_video_file_ext=video_file_ext,
            cfg_workers=workers
          )
        #endif

        log.save_output_json(
          data_json=config,
          fname=name + ".txt",
          subfolder_path='blur_configs'
        )
      #endfor
    #endfor
  #endfor


