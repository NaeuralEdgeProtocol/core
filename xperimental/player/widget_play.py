import os
import PIL
import ast
import numpy as np

from io import BytesIO
from copy import deepcopy
from collections import defaultdict

from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QDialog
from decentra_vision.draw_utils import DrawUtils
from naeural_core.business.utils import intersect_box_irregular_target
from naeural_core.xperimental.player.movie_reader import SimpleMovieReader
from naeural_core.xperimental.player.selected_zones import SelectedZones


class PlayWidget(QDialog):
  def __init__(self,
               log,
               path_coords=None,
               serving_manager=None,
               inference_graphs=None,
               draw_inference_labels=None,
               parent=None,
               ):
    super().__init__(parent=parent)

    self.log = log
    self._parent = parent
    self._serving_manager = serving_manager
    self._inference_graphs = inference_graphs

    self._painter = DrawUtils(log=self.log)
    self._path_coords = path_coords
    self._draw_inference_labels = draw_inference_labels

    uic.loadUi('xperimental/player/play_plugin.ui', self)
    self.gb_menu.setTitle('Play movie')

    self.btn_load_movie.clicked.connect(self._load)
    self.btn_close.clicked.connect(self._close)
    self.btn_reset.clicked.connect(self._reset_coords)
    self.btn_rect.clicked.connect(self._to_rect)
    self.btn_full.clicked.connect(self._set_full_screen)

    self.btn_next_step.clicked.connect(self._next_step)
    self.btn_prev_step.clicked.connect(self._previous_step)
    self.btn_skip_forward.clicked.connect(self._skip_forward)
    self.btn_skip_backward.clicked.connect(self._skip_backward)
    self.btn_apply_previous_draw.clicked.connect(self._apply_previous_draw)
    self.btn_export_coords.clicked.connect(self._export_coords)

    self.tb_skip_frames.textChanged.connect(self._skip_frames_changed)
    self.btn_skip_forward.setVisible(False)
    self.btn_skip_backward.setVisible(False)

    self._step_coords = {}

    self._reset_coords()
    self._setup_coords()
    return

  #
  # START HANDLERS
  #
  def _load(self):
    fname = QtWidgets.QFileDialog.getOpenFileName(
      self,
      'Open file',
      '/',
      "Video files (*.avi *.mp4)")[0]
    if fname == '':
      return

    self._path_file = fname
    self._movie_reader = SimpleMovieReader(
      log=self.log,
      path_file=fname
    )
    self.lbl_path_movie.setText(fname[-25:])
    self._setup_image()

    # just to fit the image into the frame
    self._next_step()
    self._previous_step()
    return

  def _next_step(self):
    has_frame, frame = self._movie_reader.read_next()
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _previous_step(self):
    has_frame, frame = self._movie_reader.read_previous()
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _skip_forward(self):
    nr_skip = int(self.tb_skip_frames.toPlainText())
    has_frame, frame = self._movie_reader.skip_forward(steps=nr_skip)
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _skip_backward(self):
    nr_skip = int(self.tb_skip_frames.toPlainText())
    has_frame, frame = self._movie_reader.skip_backward(steps=nr_skip)
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _apply_previous_draw(self):
    self._fill_steps_with_previous_value()
    return

  def _export_coords(self):
    # check if we can create intervals
    new_dct = defaultdict(list)
    for k, v in self._step_coords.items():
      new_dct[str(v)].append(k)

    dct = {}
    for k, v in new_dct.items():
      l = ast.literal_eval(k)
      key = '_'.join(str(x) for x in v)
      dct[key] = l

    self.log.save_output_json(
      data_json=dct,
      fname='{}_{}_step_coords.txt'.format(self.log.now_str(), os.path.basename(self._path_file))
    )
    coords_window = SelectedZones(
      parent=self,
      log=self.log,
      selected_zones=dct
    )
    coords_window.show()
    return

  def _skip_frames_changed(self):
    txt = self.tb_skip_frames.toPlainText()
    try:
      nr = int(txt)
    except:
      nr = None

    if nr:
      self.btn_skip_forward.setVisible(True)
      self.btn_skip_backward.setVisible(True)

      self.btn_skip_forward.setText('Forward {} steps'.format(nr))
      self.btn_skip_backward.setText('Backward {} steps'.format(nr))
    else:
      self.btn_skip_forward.setVisible(False)
      self.btn_skip_backward.setVisible(False)
    return
  #
  # END HANDLERS
  #

  def _setup_coords(self):
    if not self._path_coords:
      return

    dct = self.log.load_json(self._path_coords)
    for k, v in dct.items():
      if isinstance(k, (int, float)):
        lst_steps = [k]
      else:
        lst_steps = k.split('_')
      lst_steps = [float(x) for x in lst_steps]
      for step in lst_steps:
        self._step_coords[step] = v
    return

  def _maybe_display_coords(self):
    crt_step = self._movie_reader.get_idx_selected()
    crt_coords = self._step_coords.get(crt_step, None)
    if crt_coords is not None:
      self._done_target_area = True
      self._coords = crt_coords
      self._refresh_coord_box()
    return

  def _show_frame(self, frame):
    frame = frame.copy()
    text = 'Frame: {}/{}'.format(self._movie_reader.get_idx_selected(), self._movie_reader.get_frame_count())

    if self._serving_manager:
      lst_all_inf = []
      for inf_graph in self._inference_graphs:
        server_name = inf_graph['SERVER_NAME']
        server_name_output = inf_graph['SERVER_NAME_OUTPUT']
        filter_objects = inf_graph['FILTER']

        np_imgs = np.expand_dims(frame[:, :, ::-1], axis=0)
        inf = self._serving_manager.predict(
          server_name=server_name,
          inputs=np_imgs
        )['INFERENCES']
        if isinstance(inf, (list)):
          lst_inf = inf[0]
        elif isinstance(inf, (dict)):
          lst_inf = inf[server_name_output][0]
        lst_inf = [x for x in lst_inf if x['TYPE'] in filter_objects]
        lst_all_inf += lst_inf

      crt_step = self._movie_reader.get_idx_selected()
      coords = self._step_coords.get(crt_step, [])
      if coords:
        lst_intersect, lst_no_intersect = [], []
        for dct_inf in lst_all_inf:
          if intersect_box_irregular_target(dct_inf['TLBR_POS'], coords):
            lst_intersect.append(dct_inf)
          else:
            lst_no_intersect.append(dct_inf)

        frame = self._painter.draw_inference_boxes(
          image=frame,
          lst_inf=lst_intersect,
          color=(0, 0, 255),
          draw_label=self._draw_inference_labels
        )
        frame = self._painter.draw_inference_boxes(
          image=frame,
          lst_inf=lst_no_intersect,
          draw_label=self._draw_inference_labels
        )
      else:
        frame = self._painter.draw_inference_boxes(
          image=frame,
          lst_inf=lst_all_inf,
          draw_label=self._draw_inference_labels
        )

    frame = self._painter.alpha_text_rectangle(
      image=frame,
      text=text,
      top=10,
      left=10
    )
    frame = frame[:, :, ::-1]
    frame = self._np_img_to_binary(frame)
    self._setup_image(frame)
    return

  def _fill_steps_with_previous_value(self):
    crt_step = self._movie_reader.get_idx_selected()
    increment_val = 1

    # get previous step
    steps = np.array(list(self._step_coords.keys()))
    sel_idx = steps < crt_step
    sel_steps = steps[sel_idx]
    prev_step = float(sel_steps[-1])
    prev_coords = self._step_coords[prev_step]

    for step in np.arange(prev_step, crt_step + 0.01, increment_val):
      self._step_coords[step] = prev_coords
    return

  def _np_img_to_binary(self, np_rgb):
    image = PIL.Image.fromarray(np_rgb)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

  def _reset_coords(self):
    self.x1 = None
    self.y1 = None
    self.x2 = None
    self.y2 = None
    self.ox = None
    self.oy = None
    self._move_id = None
    self._coords = []
    self._TLBR = None
    self._done_target_area = False
    self.start_x, self.start_y = None, None
    self._need_draw = False
    self._need_draw_tlbr = False
    self._need_full_repaint = False
    self._need_move_draw = False
    self._need_line = False
    self.txt_coords.clear()
    return

  def _close(self):
    # self._parent._stop_drawing = False
    self.close()
    return

  def _setup_image(self, frame_binary=None):
    gb_w = self.gb_img.frameSize().width()
    gb_h = self.gb_img.frameSize().height()
    box_w = gb_w - 20
    box_h = gb_h - 20
    self._pixmap = QtGui.QPixmap()
    self._pixmap.loadFromData(frame_binary)

    if frame_binary is None:
      frame_binary = self._next_step()

    pixmap_original_size = self._pixmap.size()
    self.orig_h = pixmap_original_size.height()
    self.orig_w = pixmap_original_size.width()
    self._pixmap = self._pixmap.scaled(box_w, box_h, Qt.KeepAspectRatio)
    self.lbl_img.setPixmap(self._pixmap)
    pixmap_size = self._pixmap.size()
    img_h = pixmap_size.height()
    img_w = pixmap_size.width()
    self.orig_new_ratio = self.orig_h / img_h
    self.lbl_img.resize(img_w, img_h)
    top = gb_h // 2 - img_h // 2 + 7
    left = gb_w // 2 - img_w // 2
    self.lbl_img.setGeometry(left, top, img_w, img_h)
    self.lbl_img.setPixmap(self._pixmap)

    self.lbl_img.setMouseTracking(True)
    self.lbl_img.installEventFilter(self)
    return

  def point_to_coords(self, x, y):
    return [int(x * self.orig_new_ratio), int(y * self.orig_new_ratio)]

  def coords_to_point(self, coord):
    return [int(coord[0] // self.orig_new_ratio), int(coord[1] // self.orig_new_ratio)]

  def is_point_in_coords(self, x, y):
    _MARGIN = 10
    result = None
    for i, coord in enumerate(self._coords):
      px, py = self.coords_to_point(coord)
      if abs(x - px) <= _MARGIN and abs(y - py) <= _MARGIN:
        result = i
        break
    return result

  def _refresh_coord_box(self):
    self.txt_coords.clear()
    for i, coord in enumerate(self._coords):
      _s = 'P'
      if i == 1:
        _s = 'S'
      elif i == len(self._coords) - 1 and self._done_target_area:
        _s = 'E'
      self.txt_coords.appendPlainText("{}:{:4d},{:4d}".format(
        _s,
        coord[0], coord[1]))
    # self._setup_image()
    self._need_full_repaint = True
    self.update()
    return

  def add_coord(self, x, y):
    _MARGIN = 10

    if len(self._coords) >= 3:
      if abs(x - self.start_x) <= _MARGIN and abs(y - self.start_y) <= _MARGIN:
        self._done_target_area = True
        x = self.start_x
        y = self.start_y
    self._coords.append(self.point_to_coords(x, y))
    _s = 'P'

    if len(self._coords) == 1:
      _s = 'S'
    elif self._done_target_area:
      _s = 'E'

      key = self._movie_reader.get_idx_selected()
      self._step_coords[key] = deepcopy(self._coords)

    self.txt_coords.appendPlainText("{}:{:4d},{:4d}".format(
      _s,
      self._coords[-1][0], self._coords[-1][1]))
    return

  def check_orig_dest(self):
    w = abs(self.x1 - self.x2)
    h = abs(self.y1 - self.y2)
    d = pow(pow(w, 2) + pow(h, 2), 0.5)
    if d > 20:
      return True
    else:
      return False

  def _set_full_screen(self):
    self.lbl_coords.setText('Done area.')
    self._set_TLBR(0, 0, self.orig_h, self.orig_w)
    return

  def _set_TLBR(self, top, left, bottom, right):
    self._TLBR = [top, left, bottom, right]
    self._done_target_area = True
    self._setup_image()
    self.txt_coords.clear()
    self.txt_coords.appendPlainText('T: {}'.format(top))
    self.txt_coords.appendPlainText('L: {}'.format(left))
    self.txt_coords.appendPlainText('B: {}'.format(bottom))
    self.txt_coords.appendPlainText('R: {}'.format(right))
    self._need_draw_tlbr = True
    self.update()
    return

  def _to_rect(self):
    if not self._done_target_area:
      return
    all_points_x = [x[0] for x in self._coords]
    all_points_y = [x[1] for x in self._coords]
    self._coords = []
    top = min(all_points_y)
    left = min(all_points_x)
    bottom = max(all_points_y)
    right = max(all_points_x)
    self._set_TLBR(top, left, bottom, right)
    return

  def paintEvent(self, event):
    PEN_SIZE = 5
    if self._need_full_repaint:
      painter = QtGui.QPainter(self.lbl_img.pixmap())

      pen = QtGui.QPen()
      pen.setWidth(PEN_SIZE)
      pen.setColor(QtGui.QColor(0, 255, 0))

      brush = QtGui.QBrush()
      brush.setColor(QtGui.QColor(0, 255, 0))
      brush.setStyle(Qt.SolidPattern)

      painter.setBrush(brush)
      painter.setPen(pen)

      circle_size = 8

      for i in range(1, len(self._coords)):
        x1, y1 = self.coords_to_point(self._coords[i - 1])
        x2, y2 = self.coords_to_point(self._coords[i])
        painter.drawEllipse(x2 - circle_size // 2, y2 - circle_size // 2, circle_size, circle_size)
        painter.drawLine(x1, y1, x2, y2)
        x1, y1 = x2, y2
      x2, y2 = self.coords_to_point(self._coords[0])
      painter.drawEllipse(x2 - circle_size // 2, y2 - circle_size // 2, circle_size, circle_size)
      painter.drawLine(x1, y1, x2, y2)
      painter.end()
      self._need_full_repaint = False

    elif self._need_draw:
      painter = QtGui.QPainter(self.lbl_img.pixmap())

      pen = QtGui.QPen()
      pen.setWidth(PEN_SIZE)
      pen.setColor(QtGui.QColor(0, 255, 0))

      brush = QtGui.QBrush()
      brush.setColor(QtGui.QColor(0, 255, 0))
      brush.setStyle(Qt.SolidPattern)

      painter.setBrush(brush)
      painter.setPen(pen)

      painter.setCompositionMode(QtGui.QPainter.RasterOp_SourceXorDestination)
      painter.drawLine(self.x1, self.y1, self.ox, self.oy)
      painter.drawEllipse(self.x1, self.y1, 8, 8)
      painter.drawLine(self.x1, self.y1, self.x2, self.y2)
      painter.end()
      self._need_draw = False
    elif self._need_draw_tlbr:
      painter = QtGui.QPainter(self.lbl_img.pixmap())

      pen = QtGui.QPen()
      pen.setWidth(PEN_SIZE)
      pen.setColor(QtGui.QColor(0, 255, 0))
      painter.setPen(pen)

      y1, x1, y2, x2 = np.array(self._TLBR) / self.orig_new_ratio
      painter.drawRect(x1, y1, x2 - x1, y2 - y1)
      painter.end()
      self._need_draw_tlbr = False
    elif self._need_line:
      painter = QtGui.QPainter(self.lbl_img.pixmap())

      pen = QtGui.QPen()
      pen.setWidth(PEN_SIZE)
      pen.setColor(QtGui.QColor(0, 255, 0))

      brush = QtGui.QBrush()
      brush.setColor(QtGui.QColor(0, 255, 0))
      brush.setStyle(Qt.SolidPattern)

      painter.setBrush(brush)
      painter.setPen(pen)

      circle_size = 8
      painter.drawEllipse(self.lx1 - circle_size // 2, self.ly1 - circle_size // 2, circle_size, circle_size)
      painter.drawLine(self.lx1, self.ly1, self.lx2, self.ly2)
      painter.end()
      self._need_line = False
    elif self._need_move_draw:
      painter = QtGui.QPainter(self.lbl_img.pixmap())

      pen = QtGui.QPen()
      pen.setWidth(PEN_SIZE)
      pen.setColor(QtGui.QColor(0, 255, 0))

      brush = QtGui.QBrush()
      brush.setColor(QtGui.QColor(0, 255, 0))
      brush.setStyle(Qt.SolidPattern)

      painter.setBrush(brush)
      painter.setPen(pen)

      circle_size = 8

      painter.setCompositionMode(QtGui.QPainter.RasterOp_SourceXorDestination)

      if self._move_old_list is not None:
        for i in range(1, len(self._coords)):
          x1, y1 = self.coords_to_point(self._move_old_list[i - 1])
          x2, y2 = self.coords_to_point(self._move_old_list[i])
          painter.drawEllipse(x2 - circle_size // 2, y2 - circle_size // 2, circle_size, circle_size)
          painter.drawLine(x1, y1, x2, y2)
          x1, y1 = x2, y2
        x2, y2 = self.coords_to_point(self._move_old_list[0])
        painter.drawEllipse(x2 - circle_size // 2, y2 - circle_size // 2, circle_size, circle_size)
        painter.drawLine(x1, y1, x2, y2)

      for i in range(1, len(self._coords)):
        x1, y1 = self.coords_to_point(self._move_new_list[i - 1])
        x2, y2 = self.coords_to_point(self._move_new_list[i])
        painter.drawEllipse(x2 - circle_size // 2, y2 - circle_size // 2, circle_size, circle_size)
        painter.drawLine(x1, y1, x2, y2)
        x1, y1 = x2, y2
      x2, y2 = self.coords_to_point(self._move_new_list[0])
      painter.drawEllipse(x2 - circle_size // 2, y2 - circle_size // 2, circle_size, circle_size)
      painter.drawLine(x1, y1, x2, y2)

      painter.end()

      self._need_move_draw = False
    return

  def eventFilter(self, source, event):
    is_mouse = False

    if event.type() == QEvent.MouseMove:
      if event.buttons() == Qt.NoButton:
        if self.x1 is not None and not self._need_draw and not self._done_target_area:
          self.ox = self.x2
          self.oy = self.y2
          self.x2 = event.pos().x()
          self.y2 = event.pos().y()
          is_mouse = True
          self._need_draw = True
          self.update()
        elif self._done_target_area and self._move_id is not None:
          is_mouse = True
          c_x = event.pos().x()
          c_y = event.pos().y()
          self._move_old_list = self._move_new_list.copy()
          self._move_new_list[self._move_id] = self.point_to_coords(c_x, c_y)
          self._move_new_list[-1] = self._move_new_list[0]
          self._need_move_draw = True
          self.update()

    if event.type() == QEvent.MouseButtonPress:
      if event.button() == Qt.LeftButton:
        is_mouse = True
        c_x = event.pos().x()
        c_y = event.pos().y()
        if self._done_target_area:  # check if we want to move something
          if self._move_id is not None:
            coord = self.point_to_coords(c_x, c_y)
            self._coords[self._move_id] = coord
            self._coords[-1] = self._coords[0]
            self._refresh_coord_box()
            self._move_id = None
            self._move_coord = None
          else:
            move_coord = self.is_point_in_coords(c_x, c_y)
            if move_coord is not None:
              self._move_id = move_coord
              self._move_coord = self._coords[move_coord]
              self._move_new_list = self._coords.copy()
              self._move_new_list[self._move_id] = self.point_to_coords(c_x, c_y)
              self._move_new_list[-1] = self._move_new_list[0]
              self._move_old_list = None
              # self._setup_image()
              self._need_move_draw = True
              self.update()
        else:
          if self.x1 is None:
            self.x1 = c_x
            self.y1 = c_y
            self.x2 = self.x1
            self.y2 = self.y1
            self.start_x, self.start_y = self.x1, self.y1
            self.add_coord(self.x1, self.y1)
          else:
            if self.check_orig_dest():
              self.x2 = c_x
              self.y2 = c_y
              self.add_coord(self.x2, self.y2)
              self.lx1 = self.x1
              self.lx2 = self.x2
              self.ly1 = self.y1
              self.ly2 = self.y2
              self.x1 = self.x2
              self.y1 = self.y2
              self._need_line = True
              self.update()

    if is_mouse:
      if self._move_id is not None:
        self.lbl_coords.setText("Move {} ({}) - {} ({}) ".format(
          self._move_coord,
          self.coords_to_point(self._move_coord),
          self._move_new_list[self._move_id],
          self.coords_to_point(self._move_new_list[self._move_id]),
        ))
      elif self._done_target_area:
        self.lbl_coords.setText('Done area.')
      else:
        self.lbl_coords.setText("{}-{}".format(
          self.point_to_coords(self.x1, self.y1),
          self.point_to_coords(self.x2, self.y2),
        ))
    return super().eventFilter(source, event)
