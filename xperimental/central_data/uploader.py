import pandas as pd
import cx_Oracle
from time import time


if __name__ == '__main__':
  
  FN = 'C:/Dropbox/DATA/__DATASETS/two_years_full_data_for_tableau.parquet'
    
  # Read the Parquet file
  print("Loading data from Parquet file...", flush=True)
  df = pd.read_parquet(FN)
  nr_rows = df.shape[0]
  
  print(f"Loaded {nr_rows:,} rows from the Parquet file.", flush=True)

  # Oracle DB connection details
  username = 'prev_user'
  password = 'Hyfy1234'
  dsn = '172.31.255.153/XEPDB1'  # Adjust as necessary
  port = 1521
  encoding = 'UTF-8'

  # Establish the DB connection
  cx_Oracle.init_oracle_client(lib_dir=r"C:/instantclient_21_13")
  connection = cx_Oracle.connect(username, password, dsn, encoding=encoding)
  print("Established connection to the Oracle DB: ", connection, flush=True)
  cursor = connection.cursor()
  
  # first get all the date from the tabl, display the first 10 rows
  # if the total number of rows is below parquet file, then 
  # delete all rows and load the data
  
  # Prepare the SELECT statement
  select_stmt = """
  SELECT * FROM prev_user.history
  """
  cursor.execute(select_stmt)
  rows = cursor.fetchall()
  print("The data from the table is: ")
  print(rows[:10])
  print("The total number of rows in the table is: ", len(rows))
  
  if len(rows) < nr_rows:
    print("Deleting all rows from the table...")
    delete_stmt = """
    DELETE FROM prev_user.history
    """
    cursor.execute(delete_stmt)
    connection.commit()
    print("All rows deleted from the table.")

    # Prepare the INSERT INTO statement
    # Adjust the columns and table name as per your database schema
    insert_stmt = """
    INSERT INTO prev_user.history (DATETIME, FUNCTIONALITY, LOCATION, ZONE, VALUE, IS_ALERT)
    VALUES (:1, :2, :3, :4, :5, :6)
    """

    # Iterate over DataFrame rows as tuples, inserting each into the DB
    i = 0
    print("Initiating data load...", flush=True)
    start_time = time()
    for row in df.itertuples(index=False, name=None):
      i += 1
      if i > 100:
        break
      if i % 1000 == 0:
        prc = (i * 100) / nr_rows
        elapsed = time() - start_time
        remaining = (nr_rows - i) * (elapsed / i) / 60
        print(f"Processed {i} rows ({prc:.2f}% complete). Remaining {remaining:.1f} min.", flush=True)
      cursor.execute(insert_stmt, row)

    # Commit the transaction
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    print("Data loaded successfully.")
    
    # now select the first 10 rows from the table to verify the data
    # Prepare the SELECT statement
    connection = cx_Oracle.connect(username, password, dsn, encoding=encoding)
    cursor = connection.cursor()
    select_stmt = """
    SELECT * FROM prev_user.history
    """
    cursor.execute(select_stmt)
    rows = cursor.fetchall()
    print("The data from the table is: ")
    print(rows[:10])
    print("The total number of rows in the table is: ", len(rows))
    

  else:
    print("The table already contains all the data from the Parquet file.")
    print("No data will be loaded.")
    cursor.close()
    connection.close()

