#global dependencies
import pyodbc
import psycopg2
import pandas as pd

from psycopg2.extras import RealDictCursor

def connect_postgresql(log, db_config):
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

def select_data_postgresql(conn, query):
  cursor = conn.cursor(cursor_factory=RealDictCursor)    
  cursor.execute(query)
  query_results = cursor.fetchall()
  df = pd.DataFrame(query_results)
  cursor.close()
  return df

def connect_sqlserver(log, db_config):
  has_conn = False
  conn = None
  for i in range(10):
    try:
      log.p('Try #{} to connect to database'.format(i))
      
      server = db_config['SERVER']
      database = db_config['DATABASE']
      username = db_config['USER']
      password = db_config['PASSWORD']
      conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
  
      has_conn = True
      log.p('Connection done')
      break
    except:
      log.p('Failed connecting to database @try #{}'.format(i))
  #endfor  
  return has_conn, conn

def select_data_sqlserver(conn, query):
  df = pd.read_sql(query, conn)
  return df