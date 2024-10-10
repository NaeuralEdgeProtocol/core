#global dependencies
import sys

from PyQt5.QtWidgets import QApplication

#local dependencies
from naeural_core import Logger 
from xperimental.player.widget_play import PlayWidget
from naeural_core.serving.serving_manager import ServingManager


####USAGE
# cd to DecentrAI
# run: winrun.bat
# run: python xperimental/player/main.py


if __name__ == '__main__':  
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='CAVI', 
    config_file=cfg_file, 
    max_lines=1000, 
    TF_KERAS=False
    )
  
  serving_manager = ServingManager(
    log=log, 
    server_names= [], 
    prefix_log='[TFMGR]',
    no_parallel=True,
    config_fn=None #'inference/simple_tf_config.txt'
    )
  path_coords = None
  
  lst_inference_graphs = [
      {
        'SERVER_NAME': 'eff_det0',
        'SERVER_NAME_OUTPUT': 'eff_det0',
        'FILTER': ['person']
      },
      {
        'SERVER_NAME': 'face_covid',
        'SERVER_NAME_OUTPUT': 'FACE_DETECTION',
        'FILTER': ['face']
      }
    ]
  
  for inf_graph in lst_inference_graphs:
    serving_manager.start_server(
      server_name=inf_graph['SERVER_NAME'],
      inprocess=True
      )
  
  app = QApplication(sys.argv)
  player = PlayWidget(
    log=log, 
    path_coords=path_coords, 
    serving_manager=serving_manager,
    inference_graphs=lst_inference_graphs,
    draw_inference_labels=False
    )
  player.show()
  
  try:
    sys.exit(app.exec_())
  except SystemExit:
    log.P('Closing App', color='y')
    
  log.set_nice_prints()

