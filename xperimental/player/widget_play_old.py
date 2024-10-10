import cv2
import PIL
import ast
import numpy as np

from io import BytesIO
from copy import deepcopy
from collections import defaultdict, deque

from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QDialog
from decentra_vision.draw_utils import DrawUtils
from naeural_core.business.utils import intersect_box_irregular_target


class PlayWidget(QDialog):
  def __init__(self,
               log,
               path_file,
               inference_graph=None,
               filter_objects=None,
               parent=None,
               ):
    super().__init__(parent=parent)

    self.log = log
    self._parent = parent
    self._inference_graph = inference_graph
    self._filter_objects = filter_objects

    self._painter = DrawUtils(log=self.log)
    self._path_file = path_file

    uic.loadUi('xperimental/player/play_plugin.ui', self)
    self.gb_menu.setTitle('Play movie')

    self._frame_current_displayed = 0
    self._frame_current = 0
    self._time_current = 0
    self._read_method = 'simple'

    self._init_capture()

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

    self.cb_use_frame_steps.setChecked(True)

    self._buffer = deque(maxlen=100)

    self._step_coords = {}

    self._reset_coords()
    self._setup_image()
    return

  #
  # START HANDLERS
  #
  def _next_step(self):
    has_frame, frame = self._read_next(steps=1)
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _previous_step(self):
    has_frame, frame = self._read_prev(steps=1)
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _skip_forward(self):
    nr_skip = int(self.tb_skip_frames.toPlainText())
    has_frame, frame = self._read_next(steps=nr_skip)
    if has_frame:
      self._show_frame(frame)
      self._reset_coords()
      self._maybe_display_coords()
    return

  def _skip_backward(self):
    nr_skip = int(self.tb_skip_frames.toPlainText())
    has_frame, frame = self._read_prev(steps=nr_skip)
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
      fname='{}_step_coords.txt'.format(self.log.now_str())
    )
    return
  #
  # END HANDLERS
  #

  def _read_frame_with_set(self):
    if self._use_frame_increments():
      self._capture.set(cv2.CAP_PROP_POS_FRAMES, self._frame_current)
    else:
      self._capture.set(cv2.CAP_PROP_POS_MSEC, self._time_current)

    has_frame, frame = self._capture.read()
    return has_frame, frame

  def _read_frame_simple(self, steps=1):
    steps = max(1, steps)
    for _ in range(steps):
      has_frame, frame = self._capture.read()
      if has_frame:
        self._frame_current += 1
        self._frame_current_displayed += 1
        self._buffer.append((self._frame_current, frame))
    return has_frame, frame

  def _read_next(self, steps):
    has_frame, frame = None, None
    if self._read_method == 'simple':
      # check in buffer
      dct = {idx: frame for idx, frame in self._buffer}
      max_idx = max(dct.keys()) if dct else 0
      search_idx = self._frame_current_displayed + steps
      if search_idx not in dct:
        # read from movie
        steps = search_idx - max_idx
        self.log.p('Reading from movie the next {} frames. Crt frame: {}'.format(steps, self._frame_current))
        has_frame, frame = self._read_frame_simple(steps=steps)
      else:
        has_frame, frame = True, dct[search_idx]
    else:
      self._increment_step(steps=steps)
      has_frame, frame = self._read_frame_with_set()

    if not has_frame:
      self.log.p('No frames received!')
    return has_frame, frame

  def _read_prev(self, steps):
    has_frame, frame = None, None
    if self._read_method == 'simple':
      has_frame, frame = self._read_buffer_frame(steps_back=steps)
    else:
      self._decrement_step(steps=steps)
      has_frame, frame = self._read_frame_with_set()

    if not has_frame:
      self.log.p('No frames received!')
    return has_frame, frame

  def _read_buffer_frame(self, steps_back):
    dct = {idx: frame for idx, frame in self._buffer}
    max_idx = max(dct.keys())
    search_idx = max(1, max_idx - steps_back)
    if search_idx not in dct:
      # need to re-read movie in order to extract the desired idx
      self.log.p('Need to re-read movie up to index {} in order to serve the requested frame'.format(search_idx))
      self._close_video_capture()
      self._init_capture()
      self._frame_current = 0
      self._frame_current_displayed = 0
      self._buffer = deque(maxlen=100)
      self._read_frame_simple(steps=search_idx)
    else:
      new_buffer = deque(maxlen=100)
      for idx, frame in self._buffer:
        if idx <= search_idx:
          new_buffer.append((idx, frame))
      self._buffer = new_buffer
    dct = {idx: frame for idx, frame in self._buffer}
    self._frame_current_displayed = search_idx
    return True, dct[search_idx]

  def _maybe_display_coords(self):
    crt_step = self._frame_current_displayed if self._use_frame_increments() else self._time_current
    crt_coords = self._step_coords.get(crt_step, None)
    if crt_coords is not None:
      self._done_target_area = True
      self._coords = crt_coords
      self._refresh_coord_box()
    return

  def _np_img_to_binary(self, np_rgb):
    image = PIL.Image.fromarray(np_rgb)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

  def _create_video_capture(self):
    cap = cv2.VideoCapture(self._path_file)
    return cap

  def _close_video_capture(self):
    self._capture.release()
    return

  def _get_number_of_frames(self):
    i = 0
    while True:
      has_frame, frame = self._capture.read()
      if not has_frame:
        break
      i += 1
    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return i

  def _init_capture(self):
    self._capture = self._create_video_capture()
    self._fps = int(self._capture.get(cv2.CAP_PROP_FPS))
    self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    self._capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    self._video_time = self._capture.get(cv2.CAP_PROP_POS_MSEC)
    self._capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    if self._read_method == 'simple':
      self._frame_count = self._get_number_of_frames()
    else:
      self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
    self._iterative_frame_count = self._get_number_of_frames()
    self._frame_msec = 1000 / self._fps
    return

  def _increment_step(self, steps=1):
    if self._use_frame_increments():
      if self._frame_current >= self._frame_count:
        self.log.p('End of movie')
        return

      go_to = self._frame_current + steps * self._frame_msec
      go_to = min(self._frame_count, go_to)

      self._frame_current = go_to
    else:
      if self._time_current == self._video_time:
        self.log.p('End of movie')
        return

      go_to = self._time_current + steps * self._frame_msec
      go_to = min(self._video_time, go_to)
      self._time_current = go_to
    return

  def _decrement_step(self, steps=1):
    if self._use_frame_increments():
      if self._frame_current <= 0:
        self.log.p('End of movie')
        return

      go_to = self._frame_current - steps * self._frame_msec
      go_to = max(0, min(self._frame_count, go_to))

      self._frame_current = go_to
    else:
      if self._time_current == 0:
        self.log.p('End of movie')
        return

      go_to = self._time_current - steps * self._frame_msec
      go_to = max(0, min(self._video_time, go_to))
      self._time_current = go_to
    return

  def _show_frame(self, frame):
    frame = frame.copy()
    if self._use_frame_increments():
      text = 'Frame: {}/{}'.format(self._frame_current_displayed, self._frame_count)
    else:
      text = 'Time: {}/{}'.format(self._time_current, self._video_time)

    if self._inference_graph:
      lst_inf = self._inference_graph.predict(frame)['INFERENCES'][0]
      lst_inf = [x for x in lst_inf if x['TYPE'] in self._filter_objects]

      crt_step = self._frame_current_displayed if self._use_frame_increments() else self._time_current
      coords = self._step_coords.get(crt_step, [])
      if coords:
        lst_intersect, lst_no_intersect = [], []
        for dct_inf in lst_inf:
          if intersect_box_irregular_target(dct_inf['TLBR_POS'], coords):
            lst_intersect.append(dct_inf)
          else:
            lst_no_intersect.append(dct_inf)

        frame = self._painter.draw_inference_boxes(
          image=frame,
          lst_inf=lst_intersect,
          color=(0, 0, 255)
        )
        frame = self._painter.draw_inference_boxes(
          image=frame,
          lst_inf=lst_no_intersect
        )
      else:
        frame = self._painter.draw_inference_boxes(
          image=frame,
          lst_inf=lst_inf
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

  def _use_frame_increments(self):
    return self.cb_use_frame_steps.isChecked()

  def _fill_steps_with_previous_value(self):
    crt_step = self._frame_current_displayed if self._use_frame_increments() else self._time_current
    increment_val = 1 if self._use_frame_increments() else self._frame_msec

    # get previous step
    steps = np.array(list(self._step_coords.keys()))
    sel_idx = steps < crt_step
    sel_steps = steps[sel_idx]
    prev_step = float(sel_steps[-1])
    prev_coords = self._step_coords[prev_step]

    for step in np.arange(prev_step, crt_step + 0.01, increment_val):
      self._step_coords[step] = prev_coords
    return

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
    self.lbl_coords.setText('Click/move on img to setup area')
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
    return int(x * self.orig_new_ratio), int(y * self.orig_new_ratio)

  def coords_to_point(self, coord):
    return int(coord[0] // self.orig_new_ratio), int(coord[1] // self.orig_new_ratio)

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

      key = self._frame_current_displayed if self._use_frame_increments() else self._time_current
      self._step_coords[key] = deepcopy(self._coords)
      print(self._step_coords)

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
