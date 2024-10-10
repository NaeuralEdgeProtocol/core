# global dependencies
import os
import cv2
import json
import traceback
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from shutil import move, copyfile

# local dependencies
import decentra_vision.vision.constants as draw_ct

from naeural_core import DecentrAIObject
from naeural_core import Logger
from naeural_core.xperimental.bmw.utils import connect_postgresql, select_data_postgresql, connect_sqlserver, select_data_sqlserver
from decentra_vision.draw_utils import DrawUtils, DEFAULT_FONT, DEFAULT_FONT_SIZE


class Report(DecentrAIObject):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return

  def startup(self):
    super().startup()
    self._painter = DrawUtils(log=self.log)
    db_config = self.log.load_json('xperimental/bmw/db_config.txt')
    has_conn, self.conn_postgresql = connect_postgresql(
      log=self.log,
      db_config=db_config['POSTGRESQL']
    )
    if not has_conn:
      raise ValueError(
        'Script will stop because no connection to POSTGRESQL could be established! Please verify your configs!')

    has_conn, self.conn_sqlserver = connect_sqlserver(
      log=self.log,
      db_config=db_config['SQLSERVER']
    )
    if not has_conn:
      raise ValueError(
        'Script will stop because no connection to SQLSERVER could be established! Please verify your configs!')
    return

  def total_payloads(self, start, stop):
    query = '''
    SELECT COUNT(*)
    FROM vaporboxbmw."Events" 
    WHERE ("MESSAGETIME" >= '{}') AND ("MESSAGETIME" <= '{}')
    '''.format(start, stop)

    df = select_data_postgresql(
      conn=self.conn_postgresql,
      query=query
    )
    nr = 0
    if df.shape[0] > 0:
      nr = df.iloc[0]['count']
    self.log.p('Total number of payloads between `{}` and `{}`: {}'.format(start, stop, nr), noprefix=True)
    return df

  def total_events(self, start, stop):
    query = '''
    SELECT COUNT(*)
    FROM vaporboxbmw."Events" 
    WHERE ("MESSAGETIME" >= '{}') AND ("MESSAGETIME" <= '{}') AND ("IS_ALERT" IN (true, false))
    '''.format(start, stop)

    df = select_data_postgresql(
      conn=self.conn_postgresql,
      query=query
    )
    nr = 0
    if df.shape[0] > 0:
      nr = df.iloc[0]['count']
    self.log.p('Total number of events between `{}` and `{}`: {}'.format(start, stop, nr), noprefix=True)
    return df

  def events_distribution(self, start, stop):
    query = '''
    SELECT "IS_ALERT", COUNT(1) AS "COUNT"
    FROM vaporboxbmw."Events" 
    WHERE ("MESSAGETIME" >= '{}') AND ("MESSAGETIME" <= '{}')
    GROUP BY "IS_ALERT"
    '''.format(start, stop)

    df = select_data_postgresql(
      conn=self.conn_postgresql,
      query=query
    )
    self.log.p('Events distribution `{}` and `{}`: \n{}'.format(start, stop, df), noprefix=True)
    self.log.p('', noprefix=True)
    return df

  def events_distribution_detailed(self, start, stop):
    query = '''
    SELECT "STREAM", "LOCATION", "IS_ALERT", COUNT(1) AS "COUNT"
    FROM vaporboxbmw."Events" 
    WHERE ("MESSAGETIME" >= '{}') AND ("MESSAGETIME" <= '{}') AND ("IS_ALERT" IN (true, false))
    GROUP BY "STREAM", "LOCATION", "IS_ALERT"
    ORDER BY "STREAM", "LOCATION", "IS_ALERT"
    '''.format(start, stop)

    df = select_data_postgresql(
      conn=self.conn_postgresql,
      query=query
    )
    self.log.p('Events detailed distribution `{}` and `{}`: \n{}'.format(start, stop, df), noprefix=True)
    self.log.p('', noprefix=True)
    return df

  def alerts_distribution(self, start, stop):
    query = '''
    SELECT "STREAM", "LOCATION", COUNT(1) AS "COUNT"
    FROM vaporboxbmw."Events" 
    WHERE ("IS_ALERT" = True)  AND ("MESSAGETIME" >= '{}')  AND ("MESSAGETIME" <= '{}')
    GROUP BY "STREAM", "LOCATION"
    ORDER BY "COUNT" DESC
    '''.format(start, stop)
    df = select_data_postgresql(
      conn=self.conn_postgresql,
      query=query
    )
    self.log.p('Alerts distribution between `{}` and `{}`: \n{}'.format(start, stop, df), noprefix=True)
    self.log.p('', noprefix=True)
    return df

  def get_alerts(self, start, stop):
    query = '''
    SELECT *
    FROM vaporboxbmw."Events" 
    WHERE ("MESSAGETIME" >= '{}') AND ("MESSAGETIME" <= '{}') AND ("IS_ALERT" = true)
    '''.format(start, stop)
    df = select_data_postgresql(
      conn=self.conn_postgresql,
      query=query
    )
    return df

  def draw(self,
           path_img,
           full_payload,
           draw_debug_objects=False
           ):

    img = cv2.imread(path_img)

    img_height = img.shape[0]

    font = DEFAULT_FONT
    font_size = DEFAULT_FONT_SIZE
    thickness = 3

    if img_height >= 720 and img_height <= 1080:
      font = 1
      font_size = 1.4
      thickness = 2
    elif img_height == 1080:
      font = 1
      font_size = 2
      thickness = 4
    elif img_height > 1080:
      font = 2
      font_size = 1.5
      thickness = 5

    stream_payload = full_payload['PAYLOAD']
    lst_results = stream_payload['RESULTS']
    for result in lst_results:
      stream = result['STREAM']
      sign = result['SIGNATURE']
      location = result['LOCATION']
      plugin_payload = result['PAYLOAD']

      debug_objects = plugin_payload.get('DEBUG_OBJECTS', [])
      alert_objects = plugin_payload.get('ALERT_OBJECTS', [])
      location_coords = plugin_payload.get('_P_LOCATION', {})
      alert_helper = plugin_payload.get('_P_ALERT_HELPER', [])
      capture_res = plugin_payload.get('_C_actual_dps', -1)
      plugin_res = plugin_payload.get('_P_PLUGIN_REAL_RESOLUTION', -1)
      graph_type = plugin_payload.get('_P_GRAPH_TYPE', None)
      if not graph_type:
        # try extracting from alert objects
        if alert_objects:
          graph_type = list(set(x['GRAPH_TYPE'] for x in alert_objects))
        # if graph_type is still none, extract from debug_objects
        if not graph_type:
          graph_type = list(set(x['GRAPH_TYPE'] for x in debug_objects))

      if location_coords:
        # draw location
        if 'POINTS' in location_coords:
          img = self._painter.polygon(
            image=img,
            pts=location_coords['POINTS'],
            color=draw_ct.GREEN,
            thickness=4
          )
        else:
          top = location_coords['TOP']
          left = location_coords['LEFT']
          bottom = location_coords['BOTTOM']
          right = location_coords['RIGHT']
          img = self._painter.rectangle(
            image=img,
            pt1=(left, top),
            pt2=(right, bottom),
            color=draw_ct.GREEN
          )

      # draw alert objects
      if not alert_objects:
        log.p('There are no alert_objects received!', color='y')
      else:
        img = self._painter.draw_inference_boxes(
          image=img,
          lst_inf=alert_objects,
          color=draw_ct.RED,
          font=font,
          font_scale=font_size
        )

      # draw debug objects
      if draw_debug_objects:
        if not debug_objects:
          self.log.p('There are no debug objects received!')
        else:
          img = self._painter.draw_inference_boxes(
            image=img,
            lst_inf=debug_objects,
            font=font,
            font_scale=font_size
          )

      # draw alert helper info
      l_texts = []
      if not alert_helper:
        self.log.p('There are no alert_helper info received!', color='y')
      else:
        l_texts.append(alert_helper)

      if not graph_type:
        self.log.p('There are no graph_type info received!', color='y')
      else:
        l_texts.append('Used graphs: {}'.format(', '.join(graph_type)))

      l_texts.append('Capture resolution: {}'.format(capture_res))
      l_texts.append('Plugin resolution:  {}'.format(plugin_res))

      img = self._painter.multi_line_text(
        image=img,
        lst_texts=l_texts,
        org=(10, 10),
        font=font,
        font_scale=font_size,
        thickness=thickness
      )
    return img

  def draw_alerts_witness(self, start, stop, draw_debug_objects):
    df = self.get_alerts(
      start=start,
      stop=stop
    )

    if df.shape[0] == 0:
      self.log.p('There are no alerts to draw!')
      return

    # create output folder
    for x in [':']:
      start = start.replace(x, '-')
      stop = stop.replace(x, '-')

    name = '{}_{} -- {}'.format(self.log.now_str(), start, stop)
    path_output = os.path.join(
      log.get_output_folder(),
      name
    )
    os.makedirs(path_output, exist_ok=True)

    path_output_orig = os.path.join(path_output, 'orig')
    os.makedirs(path_output_orig, exist_ok=True)

    self.log.p('Output will be stored in: {}'.format(path_output))

    for idx, row in df.iterrows():
      _id = row['ID']
      try:
        full_payload = json.loads(row['TEXT'])
        url_img = row['IMG']
        if url_img == 'none':
          self.log.p('Record {} has no image attached'.format(_id))
          return

        # download image
        img_name = '{}.png'.format(_id)
        paths, msg = log.maybe_download(
          url=url_img,
          fn=img_name,
          target='output',
          force_download=True,
          print_progress=False,
          verbose=False
        )

        # move image into session folder
        path_src = paths[0]

        # process image
        copyfile(path_src, os.path.join(path_output_orig, img_name))

        path_dst = os.path.join(path_output, img_name)
        move(path_src, path_dst)
        full_payload = json.loads(row['TEXT'])
        img = self.draw(
          full_payload=full_payload,
          path_img=path_dst,
          draw_debug_objects=draw_debug_objects
        )
        self._painter.save(
          image=img,
          fn=path_dst,
          folder=None
        )
        self.log.save_json(
          dct=full_payload,
          fname=path_dst + '.txt'
        )
      except:
        self.log.p('Exception on alert {}: {}'.format(_id, traceback.format_exc()))
    return

  def get_trassir_armed_timestamp(self, day_str):
    year = datetime.now().year
    month = datetime.now().month
    yearmonth = '{}{}'.format(year, month)

    query = '''
    SELECT TOP 1 *
    FROM EV_{}
    WHERE (ID_SubEchipament = 9627) AND (CodEveniment = 'T5') AND  (DT BETWEEN '{} 20:00:00' AND '{} 23:59:59')
    ORDER BY DT DESC
    '''.format(yearmonth, day_str, day_str)

    df = select_data_sqlserver(
      conn=self.conn_sqlserver,
      query=query
    )
    date_str = None
    if df.shape[0] > 0:
      date_str = str(df.iloc[0]['DT'])
    return date_str

  def get_trassir_disarmed_timestamp(self, day_str):
    year = datetime.now().year
    month = datetime.now().month
    yearmonth = '{}{}'.format(year, month)

    query = '''
    SELECT TOP 1 *
    FROM EV_{}
    WHERE (ID_SubEchipament = 9627) AND (CodEveniment = 'T6') AND  (DT BETWEEN '{} 05:00:00' AND '{} 14:00:00')
    ORDER BY DT DESC
    '''.format(yearmonth, day_str, day_str)

    df = select_data_sqlserver(
      conn=self.conn_sqlserver,
      query=query
    )
    date_str = None
    if df.shape[0] > 0:
      date_str = str(df.iloc[0]['DT'])
    return date_str

  def get_trassir_info(self, start, stop):
    for _ in range(3):
      self.log.p('##########', noprefix=True)
    self.log.p('TRASSIR', noprefix=True)
    for _ in range(3):
      self.log.p('##########', noprefix=True)
    year = datetime.now().year
    month = datetime.now().month

    query = '''
      SELECT *
      FROM  EV_{}{}
      WHERE  (ID_SubEchipament = 9627) AND (DT >= '{}') AND (DT <= '{}')
      ORDER BY DT ASC
      '''.format(year, month, start, stop)

    df = select_data_sqlserver(
      conn=self.conn_sqlserver,
      query=query
    )

    df = df[['DT', 'CodEveniment', 'Text']]

    # saving full event report
    base_name = '{}_{}_Trassir'.format(
      start.split(' ')[0].replace('-', ''),
      stop.split(' ')[0].replace('-', '')
    )

    path = os.path.join(
      log.get_output_folder(),
      base_name + '.xlsx'
    )
    df.to_excel(path, index=False)
    self.log.p('Trassir events between `{}` and `{}`: \n{}'.format(start, stop, df), noprefix=True)
    self.log.p('', noprefix=True)

    # saving events grouped report
    lst_events = df['Text'].tolist()
    lst_events = [x.split(' ')[0] for x in lst_events]
    values, counts = np.unique(lst_events, return_counts=True)

    df_events = pd.DataFrame(
      columns=['Event', 'Ocurrance'],
      data=zip(values, counts)
    )
    self.log.p('Trassir events grouped by event type between `{}` and `{}`: \n{}'.format(
      start, stop, df_events), noprefix=True)
    self.log.p('', noprefix=True)

    path = os.path.join(
      log.get_output_folder(),
      base_name + '_grouped.xlsx'
    )
    df_events.to_excel(path, index=False)

    # saving GOC rezolutions
    query = '''
        SELECT AL_#yearmonth.*, FL_#yearmonth.Observatii
        FROM AL_#yearmonth 
        INNER JOIN FL_#yearmonth ON AL_#yearmonth.IdUnicAlarma = FL_#yearmonth.IDUnicAlarma
        WHERE (AL_#yearmonth.IDElement = 1336) AND (AL_#yearmonth.TimpGenerare >= '{}' AND  AL_#yearmonth.TimpGenerare <= '{}')
        ORDER BY AL_#yearmonth.TimpGenerare ASC
      '''.format(start, stop)
    query = query.replace('#yearmonth', '{}{}'.format(year, month))

    df_goc = select_data_sqlserver(
      conn=self.conn_sqlserver,
      query=query
    )
    df_goc = df_goc[['TimpGenerare', 'TimpInchidere', 'Observatii']]
    path = os.path.join(
      log.get_output_folder(),
      base_name + '_goc.xlsx'
    )
    df_goc.to_excel(path, index=False)
    return

  def run(self, start, stop, draw_debug_objects=False):
    self.total_payloads(
      start=start,
      stop=stop
    )

    self.total_events(
      start=start,
      stop=stop
    )

    self.events_distribution(
      start=start,
      stop=stop
    )

    self.events_distribution_detailed(
      start=start,
      stop=stop
    )

    self.alerts_distribution(
      start=start,
      stop=stop
    )

    self.draw_alerts_witness(
      start=start,
      stop=stop,
      draw_debug_objects=draw_debug_objects
    )

    self.get_trassir_info(
      start=start,
      stop=stop
    )
    return


if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='EE_BMW_REP',
    config_file=cfg_file,
    max_lines=1000,
    TF_KERAS=False
  )

  # start = '2021-10-07 22:22:23'
  # stop = '2021-10-08 06:28:46'

  # start = '2021-10-08 17:00:23'
  # stop = '2021-10-08 17:28:46'

  # start = '2021-10-08 21:57:31'
  # stop = '2021-10-09 06:58:51'

  # start = '2021-10-09 13:39:53'
  # stop = '2021-10-11 06:34:04'

  stats = Report(log=log)

  yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
  today = datetime.now().strftime("%Y-%m-%d")

  # start = stats.get_trassir_armed_timestamp(yesterday)
  # stop = stats.get_trassir_disarmed_timestamp(today)

  start = '2021-10-23 13:30:30'
  stop = '2021-10-25 06:32:41'

  stats.run(
    start=start,
    stop=stop
  )
