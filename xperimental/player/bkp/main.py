#global dependencies
import sys

from PyQt5.QtWidgets import QApplication

#local dependencies
from naeural_core import Logger 
from naeural_core.xperimental.player.widget_play import PlayWidget
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
  
  graph_face, graph_effdet4, path_coords = None, None, None
  if True:
    # graph_effdet4 = EffDet4InferenceGraph(
    #   log=log,
    #   config_path='_local_cache/_data/config/config_inference.txt'
    #   )
    
    # graph_face = FaceInferenceGraph(
    #   log=log,
    #   config_path='_local_cache/_data/config/config_inference.txt'
    #   )
    
    # start engine with face det
    graph_effdet4 = ServingManager(
      log=log, 
      server_names= [], 
      prefix_log='[TFMGR]',
      no_parallel=True,
      config_fn=None #'inference/simple_tf_config.txt'
      ) 
    
    graph_effdet4.start_server(
      server_name='eff_det4',
      inprocess=True
      )
    
    
    
    # path_coords = log.get_output_file('20211015023356264937_step_coords.txt')
    # path_coords = 'C:/Users/ETA/Dropbox/MKT/Blur Alex/Inregistrare 2 magazin 246_blured_json/tricou_verde_4_new.txt'
  
  # path_base = 'C:/Users/ETA/Dropbox/MKT/Blur Alex/Inregistrarea 1, magazin 207'
  #path = os.path.join(path_base, 'Intrare magazin.avi')
  # path = os.path.join(path_base, 'Culoar case.avi')

  # path = os.path.join(path_base, 'Culoar case.avi')  
  # path = os.path.join(path_base, 'Iesire magazin.avi')
  # path = os.path.join(path_base, 'Intrare magazin.avi')
  # path = os.path.join(path_base, 'Speed dome gang 1.avi')
  # path = os.path.join(path_base, 'Speed dome gang 3.avi')
  # path = os.path.join(path_base, 'Speed dome gang 4.avi')
  
  # path_base = 'C:/Users/ETA/Dropbox/MKT/Blur Alex/inredistrari render'
  # path = os.path.join(path_base, 'Tricou verde 1.avi')
  # path = os.path.join(path_base, 'Tricou verde 2.avi')
  # path = os.path.join(path_base, 'Tricou verde 3.1.avi')
  # path = os.path.join(path_base, 'Tricou verde 3.2.avi')
  # path = os.path.join(path_base, '7c41f4c9122ba14d.avi')
  
  
  app = QApplication(sys.argv)
  player = PlayWidget(
    log=log, 
    path_coords=path_coords,    
    inference_graphs={
      graph_effdet4: ['person']
      },
    draw_inference_labels=False
    )
  player.show()
  
  try:
    sys.exit(app.exec_())
  except SystemExit:
    log.P('Closing App', color='y')
    
  log.set_nice_prints()
    

  