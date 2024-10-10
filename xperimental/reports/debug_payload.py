# global dependencies
import os
import cv2
import json
import argparse
import psycopg2
import pandas as pd

from shutil import move, copyfile
from psycopg2.extras import RealDictCursor

# local dependencies
import decentra_vision.vision.constants as draw_ct

from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils, DEFAULT_FONT, DEFAULT_FONT_SIZE


def connect_db(log, db_config):
  has_conn = False
  conn = None
  for i in range(10):
    try:
      log.p('Try #{} to connect to database'.format(i))
      conn = psycopg2.connect(
        host=db_config['HOST'],
        port=db_config['PORT'],
        database=db_config['DATABASE'],
        user=db_config['USER'],
        password=db_config['PASSWORD']
      )
      has_conn = True
      log.p('Connection done')
      break
    except:
      log.p('Failed connecting to database @ try #{}'.format(i))
  # endfor
  return has_conn, conn


def create_query(log,
                 db_config,
                 **kwargs
                 ):
  lst_ids = kwargs.get('lst_ids', None)
  if lst_ids:
    lst_ids = lst_ids.replace(' ', '').split(',')
    lst_ids = [int(x) for x in lst_ids]
  stream = kwargs.get('stream', None)
  plugin_type = kwargs.get('plugin_type', None)
  location = kwargs.get('location', None)
  is_alert = kwargs.get('is_alert', None)
  start = kwargs.get('start', None)
  stop = kwargs.get('stop', None)

  l = [lst_ids, stream, plugin_type, location, is_alert, start, stop]
  assert any([x is not None for x in l]), 'Please ensure that you specify \
    at least one argument as otherwise the entire database will be selected and processed!. \
    Available arguments: {}'.format(kwargs.keys()
                                    )

  # Print PostgreSQL version
  query = 'SELECT * FROM {}."{}" WHERE (1 = 1) '.format(
    db_config['SCHEMA'],
    db_config['TABLE_EVENTS']
  )
  if lst_ids:
    query += ' AND ("ID" IN ({})) '.format(', '.join([str(x) for x in lst_ids]))
  if stream:
    query += ' AND ("STREAM" = \'{}\') '.format(stream)
  if plugin_type:
    query += ' AND ("PLUGIN_TYPE" = \'{}\') '.format(plugin_type)
  if location:
    query += ' AND ("LOCATION" = \'{}\') '.format(location)
  if is_alert:
    query += ' AND ("IS_ALERT" = {}) '.format(is_alert)
  if start:
    query += ' AND ("MESSAGETIME" >= \'{}\') '.format(start)
  if stop:
    query += ' AND ("MESSAGETIME" <= \'{}\') '.format(stop)
  return query


def select_data(log, db_config, conn, **kwargs):
  cursor = conn.cursor(cursor_factory=RealDictCursor)

  query = create_query(
    log=log,
    db_config=db_config,
    **kwargs
  )
  log.p('Generated the following query: \n{}'.format(query))
  cursor.execute(query)

  query_results = cursor.fetchall()
  df = pd.DataFrame(query_results)

  cursor.close()
  conn.close()
  return df


def process(args, df, painter):
  now = log.now_str()
  path_output = os.path.join(
    log.get_output_folder(),
    now
  )
  os.makedirs(path_output, exist_ok=True)

  if args.get('sop', False):
    path_output_orig = os.path.join(path_output, 'orig')
    os.makedirs(path_output_orig, exist_ok=True)

  # path_output_draw = os.path.join(path_output, 'draw')
  # os.makedirs(path_output_draw, exist_ok=True)

  for idx, row in df.iterrows():
    _id = row['ID']
    try:
      full_payload = json.loads(row['TEXT'])
      url_img = row['IMG']
      if url_img == 'none':
        log.p('Record {} has no image attached'.format(_id))
        continue

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
      if args.get('sop', False):
        copyfile(path_src, os.path.join(path_output_orig, img_name))

      path_dst = os.path.join(path_output, img_name)
      move(path_src, path_dst)
      img = cv2.imread(path_dst)

      img_height = img.shape[0]

      font = DEFAULT_FONT
      font_size = DEFAULT_FONT_SIZE
      thickness = 3

      if img_height >= 720 and img_height <= 1080:
        font = 1
        font_size = 1
        thickness = 4
      elif img_height == 1080:
        font = 1
        font_size = 1
        thickness = 1
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

        if not location_coords:
          log.p('Record {} does not contain location details'.format(_id), color='y')
        else:
          # draw location
          if 'POINTS' in location_coords:
            img = painter.polygon(
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
            img = painter.rectangle(
              image=img,
              pt1=(left, top),
              pt2=(right, bottom),
              color=draw_ct.GREEN
            )

        # draw alert objects
        if not alert_objects:
          log.p('There are no alert_objects received!', color='y')
        else:
          img = painter.draw_inference_boxes(
            image=img,
            lst_inf=alert_objects,
            color=draw_ct.RED,
            font=font,
            font_scale=font_size,
            scale_percent=100
          )

        # draw debug objects
        if args['debug_objects']:
          if not debug_objects:
            log.p('There are no debug objects received!')
          else:
            img = painter.draw_inference_boxes(
              image=img,
              lst_inf=debug_objects,
              font=font,
              font_scale=font_size,
              scale_percent=100
            )

        # draw alert helper info
        l_texts = []
        if not alert_helper:
          log.p('There are no alert_helper info received!', color='y')
        else:
          l_texts.append(alert_helper)

        if not graph_type:
          log.p('There are no graph_type info received!', color='y')
        else:
          l_texts.append('Used graphs: {}'.format(', '.join(graph_type)))

        l_texts.append('Capture resolution: {}'.format(capture_res))
        l_texts.append('Plugin resolution:  {}'.format(plugin_res))

        img = painter.multi_line_text(
          image=img,
          lst_texts=l_texts,
          org=(10, 10),
          font=font,
          font_scale=font_size,
          thickness=thickness
        )

        painter.save(
          image=img,
          fn=path_dst,
          folder=None
        )
        log.save_json(
          dct=full_payload,
          fname=path_dst + '.txt'
        )
      # endfor
    except Exception as e:
      log.p('Exception on processing ID {}: {}'.format(_id, str(e)))
    # end try-except
  # end for
  return


def main(log, args):
  try:
    log.p('Start script with following arguments:')
    for k, v in args.items():
      log.p(' * {}: {}'.format(k, v))

    # handle database connection
    db_config = log.config_data['DATABASE']
    has_conn, conn = connect_db(log, db_config)
    if not has_conn:
      log.p('Script will not continue because database connection could not be established')
      return

    # select records based on arguments
    painter = DrawUtils(log=log)

    df = select_data(
      log=log,
      db_config=db_config,
      conn=conn,
      **args
    )
    log.p('Found {} record(s) from database according to input arguments'.format(df.shape[0]))

    if df.shape[0] == 0:
      log.p('Nothing to process. The script will finish execution')
    else:
      log.p('Start processing records')
      process(
        args=args,
        df=df,
        painter=painter
      )
    # endif
  except Exception as e:
    log.p('Exception while running script: {}'.format(str(e)), color='r')
  return


if __name__ == '__main__':
  # run this script from VaporBoxPCAVI/Explorer folder:
  # python debug_payload.py -lst_ids "592416"
  # python debug_payload.py -start "2021-09-23 16:06:43"
  # python debug_payload.py -start "2021-09-23 16:06:43" -is_alert=True

  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='CAVI_DP',
    config_file=cfg_file,
    max_lines=1000,
    TF_KERAS=False
  )

  parser = argparse.ArgumentParser(
    description='Payload debug script'
  )
  parser.add_argument(
    "-lst_ids",
    "--lst_ids",
    help='DB: List of database `ID` to use for data filtering. This will be provided as a comma separated list of IDs: -lst_ids "1, 2, 3"',
    type=str,
    default=None
  )

  parser.add_argument(
    "-stream",
    "--stream",
    help="DB: `STREAM` to use for data filtering",
    type=str,
    default=None
  )

  parser.add_argument(
    "-plugin_type",
    "--plugin_type",
    help="DB: `PLUGIN_TYPE` to use for data filtering",
    type=str,
    default=None
  )

  parser.add_argument(
    "-location",
    "--location",
    help="DB: `LOCATION` to use for data filtering",
    type=str,
    default=None
  )

  parser.add_argument(
    "-is_alert",
    "--is_alert",
    help="DB: `IS_ALERT` to use for data filtering",
    type=bool,
    default=None
  )

  parser.add_argument(
    "-start",
    "--start",
    help="DB: `MESSAGETIME` start to use for data filtering",
    type=str,
    default=None
  )

  parser.add_argument(
    "-stop",
    "--stop",
    help="DB: `MESSAGETIME` stop to use for data filtering",
    type=str,
    default=None
  )

  parser.add_argument(
    "-debug_objects",
    "--debug_objects",
    help="DRAW: Decide if debug objects will be draw on image",
    type=str,
    default=None
  )

  parser.add_argument(
    "-sop",
    "--save_original_picture",
    help="SAVE: Decide if saving original image, without plot infos",
    type=bool,
    default=None
  )

  args = parser.parse_args()

  args = vars(args)

  args['start'] = '2021-10-06 22:03:49'
  args['stop'] = '2021-10-07 06:51:24'
  args['is_alert'] = True
  args['sop'] = True

  main(
    log=log,
    args=args
  )

  import numpy as np
  np.mean([0.481, 0.438, 0.356, 0.372, 0.392, 0.41, 0.479])
