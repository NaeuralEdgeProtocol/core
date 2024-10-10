# global dependencies
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

# local dependencies
from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils
from base.app_setup import ApplicationSetup
from decentrai_inference.graphs_benchmark import EFFDET2_640x1132_BS4, \
  EFFDET2_768x1358_BS4, EFFDET2_768x1358_BS5, EFFDET2_768x1358_BS7, \
  EFFDET3_768x1358_BS4, EFFDET3_768x1358_BS5, EFFDET3_768x1358_BS7


def load_imgs():
  path = 'xperimental/_images/H1080_W1920'
  lst = painter.read(
    path=path,
    reverse_channels=True
  )
  return lst


if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(lib_name='EE_BMW_BENCH', config_file=cfg_file, max_lines=1000)

  N_ITERS = 10
  N_STREAMS = 14

  # load images
  painter = DrawUtils(log=log)
  lst_imgs = load_imgs()

  # ensure graphs are available
  app_setup = ApplicationSetup(log=log)

  l = [
      'EFFDET2_640x1132_BS4',

      'EFFDET2_768x1358_BS4',
      'EFFDET2_768x1358_BS5',
      'EFFDET2_768x1358_BS7',

      'EFFDET3_768x1358_BS4',
      'EFFDET3_768x1358_BS5',
      'EFFDET3_768x1358_BS7'
    ]

  for model_name in l:
    app_setup.maybe_download_benchmark_model(
      model_name=model_name
    )

  dct_graphs = {
    'ED2_640x1132': [
      EFFDET2_640x1132_BS4
    ],

    'ED2_768x1358': [
      EFFDET2_768x1358_BS4,
      EFFDET2_768x1358_BS5,
      EFFDET2_768x1358_BS7
    ],

    'ED3_768x1358': [
      EFFDET3_768x1358_BS4,
      EFFDET3_768x1358_BS5,
      EFFDET3_768x1358_BS7
    ]
  }

  # simple benchmark
  dct_times_simple = {'CATEG': [], 'GRAPH': [], 'TOTAL_TIME': [], 'PER_IMG': []}

  for categ, l_graphs in dct_graphs.items():
    for graph_class in l_graphs:
      graph = graph_class(
        log=log,
        config_path='_local_cache/_data/config/config_inference_benchmark.txt'
      )
      bs = graph.config_graph['BATCH_SIZE']
      np_imgs = np.array(lst_imgs[:bs])
      tn = graph.name + '_bench'
      for _ in tqdm(range(N_ITERS)):
        np_imgs += 1
        np_imgs = np.clip(np_imgs, 0, 255)
        log.start_timer(tn)
        graph.predict(np_imgs)
        log.stop_timer(tn)

      total_time = log.get_timer_mean(tn)
      dct_times_simple['CATEG'].append(categ)
      dct_times_simple['GRAPH'].append(graph.name)
      dct_times_simple['TOTAL_TIME'].append(total_time)
      dct_times_simple['PER_IMG'].append(total_time / bs)
    # endfor

  df_times_simple = pd.DataFrame(dct_times_simple)
  log.p('\n\n{}'.format(df_times_simple))

  plt.figure()
  colors = ['green', 'blue', 'orange']
  markers = ['o', 's', 'P']
  for i, (categ, group) in enumerate(df_times_simple.groupby('CATEG')):
    names = group['GRAPH']
    names = [x.replace('EFFDET', 'ED') for x in names]
    vals = group['TOTAL_TIME']

    plt.plot(names, vals, color=colors[i], marker=markers[i])
    plt.xticks(rotation=45)
  plt.title('Time per graph/batch')
  plt.xlabel('Graph')
  plt.ylabel('Time per batch')
  plt.show()

  # multiple runs benchmark
  log.reset_timers()
  dct_times_batch = {'CATEG': [], 'GRAPH': [], 'TOTAL_TIME': [], 'PER_IMG': [], 'ITERS': []}
  for categ, l_graphs in dct_graphs.items():
    for graph_class in l_graphs:
      graph = graph_class(
        log=log,
        config_path='_local_cache/_data/config/config_inference_benchmark.txt'
      )
      bs = graph.config_graph['BATCH_SIZE']
      np_imgs = np.array(lst_imgs[:bs])
      tn = graph.name + '_bench'
      nr_batches = math.ceil(N_STREAMS / bs)
      for x in tqdm(range(N_ITERS)):
        log.start_timer(tn)
        for y in range(nr_batches):
          np_imgs += 1
          np_imgs = np.clip(np_imgs, 0, 255)
          graph.predict(np_imgs)
        log.stop_timer(tn)
      # endfor
      total_time = log.get_timer_mean(tn)
      dct_times_batch['CATEG'].append(categ)
      dct_times_batch['GRAPH'].append(graph.name)
      dct_times_batch['TOTAL_TIME'].append(total_time)
      dct_times_batch['PER_IMG'].append(total_time / bs)
      dct_times_batch['ITERS'].append(nr_batches)
    # endfor
  # endfor

  df_times_batch = pd.DataFrame(dct_times_batch)
  log.p('\n\n{}'.format(df_times_batch))
