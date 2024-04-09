from PyQt5 import uic
from PyQt5.QtWidgets import QDialog

class SelectedZones(QDialog):
  def __init__(self,
               parent,
               log,
               selected_zones
               ):
    super().__init__(parent=parent)
    
    self.log = log
    uic.loadUi('xperimental/player/selected_zones.ui', self)
    s = str(selected_zones)
    s = s.replace("'", '"')
    s = '"EXCLUDE_ZONES": {}'.format(s)
    self.tb_selected_zones.setText(s)
    return
  
  # for k,lst in selected_zones.items():
  #   for v in lst:
  #     if isinstance(v, (tuple)):
  #       v = list(v)
  