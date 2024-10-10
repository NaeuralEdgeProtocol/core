import os

from naeural_core import Logger
from decentrai_inference.graphs import EffDet0InferenceGraph, EffDet3768x1358BS1InferenceGraph, \
  EffDet2768x1358BS1InferenceGraph, EffDet4InferenceGraph, EffDet5BS1InferenceGraph, EffDet7BS1InferenceGraph
from decentra_vision.draw_utils import DrawUtils


if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='EE_BMW',
    config_file=cfg_file,
    max_lines=1000,
    TF_KERAS=False
  )
  painter = DrawUtils(log=log)

  folder_name = '20210923_20210924'
  path_base = os.path.join(
    log.get_dropbox_drive(),
    '_vapor_data',
    '__sources',
    'images',
    'report_bmw',
    folder_name
  )

  path_src = os.path.join(path_base, 'orig')

  l = [
    EffDet2768x1358BS1InferenceGraph
    # EffDet3768x1358BS1InferenceGraph
  ]
  for g_class in l:
    graph = g_class(
      log=log,
      config_path='_local_cache/_data/config/config_inference.txt'
    )

    path_dst = os.path.join(path_base, graph.name)

    painter.detect_and_plot(
      path_src=path_src,
      path_dst=path_dst,
      graph=graph,
      lst_filter=['person'],
      save_inferences=True,
      font=2,
      font_scale=1.5,
      color=(0, 0, 255)
    )
