import pandas as pd


if __name__ == '__main__':
  
  FN      = 'C:/Dropbox/DATA/__DATASETS/two_years_full_data_for_tableau.parquet'
  FN_CSV  = 'C:/Dropbox/DATA/__DATASETS/two_years_full_data_for_tableau.csv'
    
  # Read the Parquet file
  print("Loading data from Parquet file...", flush=True)
  
  df = pd.read_parquet(FN)
  
  # replace long strings

  # Mapping for FUNCTIONALITY transformation
  functionality_mapping = {
      "MULTIZONE_PEOPLE_PRESENCE_AGGREGATED_01"   : "P_PRS",
      "PEOPLE_COUNTING_02"                        : "P_CNT",
      "ATM_PERSON_TIME_01"                        : "ATM_T",
      "MULTIZONE_PEOPLE_PRESENCE_CROWDED_01"      : "CROWD"
  }
  
  zones_mapping = {
    'GHISEE'    : 'GH-1',
    'INTRARE-1' : 'IN-1', 
    'INTRARE-2' : 'IN-2',
    'INTRARE-BT24' : 'IN24',
    'ZONA-1' : 'ZO-1',
    'ZONA-2' : 'ZO-2',
  }

  print("Mapping FUNCTIONALITY column...", flush=True)
  # Apply mapping to FUNCTIONALITY column
  df['FUNCTIONALITY'] = df['FUNCTIONALITY'].replace(functionality_mapping)

  print("Mapping LOCATION column...", flush=True)
  # Mapping for LOCATION transformation using lambda function
  df['LOCATION'] = df['LOCATION'].apply(lambda x: x.replace('agentia-', 'ag-').replace('sucursala-', 'sc-'))
  df['LOCATION'] = df['LOCATION'].str[:20]
  
  print("Mapping ZONE column...", flush=True)
  # Apply mapping to ZONE column
  df['ZONE'] = df['ZONE'].replace(zones_mapping)
  
  # now calculate max row size
  print("Calculating max row size...", flush=True)
  tsize = 0
  for col in df.columns:
    if col in ['LOCATION', 'ZONE', 'FUNCTIONALITY']:
      maxlen = max([len(x) for x in df[col].unique()])
      print("Column: ", col, " - Max length: ", maxlen)
      tsize += maxlen
  
  print("Max row size is: ", tsize)      
    
  print("Saving as CSV file...", flush=True)
  
  df.to_csv(FN_CSV, index=False)
  
  print("Done saving as CSV file.", flush=True)