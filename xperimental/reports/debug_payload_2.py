#global dependencies
import json
import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psycopg2.extras import RealDictCursor
  
#local dependencies

from naeural_core import Logger


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
  #endfor
  return has_conn, conn

def select_data(query):
  cursor = conn.cursor(cursor_factory=RealDictCursor)    
  
  cursor.execute(query)
  
  query_results = cursor.fetchall()
  df = pd.DataFrame(query_results)
  
  cursor.close()
  conn.close()    
  return df

if __name__ == '__main__':
  cfg_file = 'main_config.txt'
  log = Logger(
    lib_name='CAVI_DP', 
    config_file=cfg_file, 
    max_lines=1000, 
    TF_KERAS=False
    )
  
  db_config = log.config_data['DATABASE']
  has_conn, conn = connect_db(log, db_config)
  
  start_date = '2021-10-06 22:03:49'
  stop_date = '2021-10-07 06:51:24'
  
  # start_date = '2021-10-05 21:36:26'
  # stop_date = '2021-10-06 06:30:08'
  
  is_alert = False
  
  query = '''
  SELECT *
  FROM vaporboxbmw."Events" 
  WHERE ("MESSAGETIME" >= '{}')  AND ("MESSAGETIME" <= '{}') AND ("IS_ALERT" = {})
  ORDER BY "ID" ASC
  '''.format(start_date, stop_date, str(is_alert).lower())
  query
  
  df = select_data(query)
  lst_json_str = df['TEXT'].tolist()
  lst_json = [json.loads(x) for x in lst_json_str]
  
  lst_means, lst_all_vals = [], []
  for full_payload in lst_json:
    stream_payload = full_payload['PAYLOAD']
    lst_results = stream_payload['RESULTS']
    for result in lst_results:
      stream = result['STREAM']
      sign = result['SIGNATURE']
      location = result['LOCATION']
      plugin_payload = result['PAYLOAD']
      
      alert_helper = plugin_payload.get('_P_ALERT_HELPER', None)
      if not alert_helper:
        continue
      
      lst_str_vals = alert_helper.split('[')[1].split(']')[0].split(',')
      lst_vals = [float(x) for x in lst_str_vals]
      if lst_vals:
        lst_means.append(round(np.mean(lst_vals), 3))
        lst_all_vals+= lst_vals
  #endfor
    
  #plot means 
  unique, counts = np.unique(lst_means, return_counts=True)
  plt.bar(
    x=unique,
    height=counts,
    width=0.001
    )
  plt.xlabel('Confidence %')
  plt.ylabel('Number of occurances')
  plt.title('AlertHelper lowering: mean value confidence distribution')
  plt.show()

        
  #plot vals 
  unique, counts = np.unique(lst_all_vals, return_counts=True)
  plt.bar(
    x=unique,
    height=counts,
    width=0.001
    )
  plt.xlabel('Confidence %')
  plt.ylabel('Number of occurances')
  plt.title('AlertHelper lowering: evaluation items confidence distribution')
  plt.show()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        