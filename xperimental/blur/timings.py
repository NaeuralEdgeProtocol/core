import os
import pandas as pd
import random
from datetime import datetime, timedelta

from naeural_core import Logger

def get_timings(sz, fs, nw):
  dct = dct_timings[sz.upper()][fs.lower()][nw.lower()]
  return dct

if __name__ == '__main__':
  log = Logger(
    lib_name='EE_TST',
    base_folder='.',
    app_folder='_local_cache',
    config_file='config_startup.txt',
    max_lines=1000,
    TF_KERAS=False
  )
  log.set_nice_prints()

  path_golden = log.get_data_subfolder('golden_payloads')
  lst_golden_files = os.listdir(path_golden)

  lst_indexes = [
    '100_MB',
    '500_MB',
    '1_GB',
    '3_GB'
  ]

  dct_timings = {k: {} for k in lst_indexes}

  dct_data = {
    'LOCAL_1w': [],
    'LOCAL_2w': [],
    'LOCAL_4w': [],
    'MINIO_1w': [],
    'MINIO_2w': [],
    'MINIO_4w': [],
    'SPDUP_1w': [],
    'SPDUP_2w': [],
    'SPDUP_4w': []
  }

  date_ref = datetime.strptime('1900-01-01', '%Y-%m-%d')
  for idx in lst_indexes:
    lst_file_names = [x for x in lst_golden_files if idx.upper() in x.upper()]
    for name in lst_file_names:
      parts = name.split('_')
      file_system = parts[2]
      nr_workers = parts[3]

      path_file = os.path.join(path_golden, name)
      payload = log.load_json(fname=path_file)

      key_name = '{}_{}'.format(file_system.upper(), nr_workers)

      timings = payload['TIMINGS']
      total_time = list(timings.values())[-1]
      date_total = datetime.strptime(total_time, "%H:%M:%S.%f")
      dct_data[key_name].append(round((date_total-date_ref).total_seconds(), 1))

      time_01_total = timings['01_prepare']
      time_01_prepare_download = str(timedelta(seconds=payload['_C_download_elapsed_time']))
      time_01_prepare_split = payload.get('_C_split_elapsed_time', '0:00:00.0')
      time_01_prepare_upload = payload.get('_C_upload_elapsed_time', '0:00:00.0')

      timings['01_prepare'] = {
        'download': time_01_prepare_download,
        'split': time_01_prepare_split,
        'upload': time_01_prepare_upload,
        'total': time_01_total
      }

      timings['workers'] = payload.get('TIMINGS_WORKERS', None)

      if file_system not in dct_timings[idx]:
        dct_timings[idx][file_system] = {}
      if nr_workers not in dct_timings[idx][file_system]:
        dct_timings[idx][file_system][nr_workers] = {}

      dct_timings[idx][file_system][nr_workers] = timings
    #endfor
  #endfor

  for i in range(len(dct_data['LOCAL_1w'])):
    for nr_w in [1, 2, 4]:
      secs_minio = dct_data['MINIO_{}w'.format(nr_w)][i]
      secs_local = dct_data['LOCAL_{}w'.format(nr_w)][i]
      dct_data['SPDUP_{}w'.format(nr_w)].append("{:.2f}%".format(100 * (1 - secs_local / secs_minio)))
    #endfor
  #endfor

  df = pd.DataFrame(data=dct_data, index=lst_indexes)
  log.p('Blur timings summary:\n{}'.format(df))

  # Initial uploads:
  #  * 100mb - 1.41s
  #  * 500mb - 6.05s
  #  * 1gb   - 11.76s
  #  * 3gb   - 30.66s
