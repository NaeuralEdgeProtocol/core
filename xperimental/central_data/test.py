import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

BRANCHES = [
    'agentia-andrei-muresanu', 'agentia-focsani-nord', 'agentia-soseaua-chitilei', 'agentia-intim',
    'agentia-bals', 'agentia-pipera', 'agentia-dragasani', 'punct-de-lucru-mall-atrium',
    'agentia-iulius-mall', 'agentia-campia-turzii-pta-m-viteazu', 'agentia-sighet-2', 'agentia-republicii',
    'agentia-falticeni', 'agentia-giurgiu', 'agentia-rogerius', 'agentia-mosilor',
    'agentia-capitol', 'agentia-sasar', 'agentia-iosefin', 'agentia-buna-ziua',
    'agentia-memorandumului', 'agentia-tudor', 'agentia-mihai-bravu', 'agentia-universitate',
    'agentia-piata-romana', 'agentia-baneasa-mall', 'agentia-electroputere-craiova', 'punct-de-lucru-gold-plaza',
    'agentia-alba-mall', 'punct-de-lucru-mall-aurora', 'agentia-vivo-mall', 'punct-de-lucru-iulius-mall',
    'agentia-militari-shopping-mall', 'agentia-mega-mall', 'agentia-park-lake-mall', 'agentia-lotus',
    'agentia-brasov-coresi-mall', 'agentia-musat-value-center-mall', 'agentia-shopping-city-mall', 'agentia-promenada-mall',
    'agentia-sun-plaza', 'agentia-bucuresti-plaza-mall', 'agentia-roman', 'agentia-lugoj',
    'agentia-apusului', 'agentia-ramnicu-sarat', 'agentia-cimpeni', 'agentia-motilor',
    'agentia-medias', 'agentia-sebes', 'agentia-cetate', 'agentia-alesd',
    'agentia-dorohoi', 'agentia-nehoiu', 'agentia-piata-marasti', 'agentia-brazda-lui-novac',
    'agentia-tecuci', 'agentia-targu-neamt', 'agentia-ampoi', 'agentia-salonta',
    'agentia-craiova-est', 'agentia-valea-rosie', 'agentia-gheorgheni', 'agentia-hateg',
    'agentia-orastie', 'sucursala-drobeta-turnu-severin', 'sucursala-alba-iulia', 'sucursala-oradea',
    'sucursala-satu-mare', 'sucursala-craiova', 'sucursala-piatra-neamt', 'sucursala-baia-mare',
    'sucursala-tg-jiu', 'sucursala-deva', 'sucursala-timisoara', 'sucursala-focsani',
    'sucursala-arad', 'sucursala-botosani', 'sucursala-cluj', 'sucursala-sf-gheorghe',
    'sucursala-mures', 'sucursala-pitesti', 'sucursala-buzau', 'sucursala-zalau',
    'sucursala-bucuresti-nord', 'sucursala-bacau', 'sucursala-miercurea-ciuc', 'sucursala-tulcea',
    'sucursala-bucuresti-vest', 'sucursala-bistrita', 'sucursala-braila', 'sucursala-bucuresti-est',
    'sucursala-targoviste', 'sucursala-slatina', 'sucursala-resita', 'agentia-millennium',
    'agentia-campina', 'agentia-onesti', 'agentia-gara-iasului', 'sucursala-vaslui'
]


SIGNATURES_X_ZONES = {
  'MULTIZONE_PEOPLE_PRESENCE_AGGREGATED_01' : {
    'ZONES' : ['CASE', 'GHISEE', 'BT24', ],
    'IS_ALERT' : 'No',
    'MIN_VALUE' : 1,
    'CONTINUOUS' : True,
    'IS_24H'  : False,
    'AGGREGATED' : True,  # Is 1-minute-aggregated
    'MAYBE_ZERO'  : False,
    'USE_WORKING_HOURS' : True,
    'BUSINESS_HOURS' : True,
  },
  
  'PEOPLE_COUNTING_02' : {
    'ZONES' : ['INTRARE-1', 'INTRARE-2', 'INTRARE-BT24'],
    'IS_ALERT' : 'No',
    'MIN_VALUE' : 1,
    'CONTINUOUS' : True,   # will skip some timestamps if not continuous
    'IS_24H'  : False,
    'AGGREGATED' : True,    # Is 1-minute-aggregated ?
    'INCREMENTAL' : True,
    'MAYBE_ZERO'  : True,
    'USE_WORKING_HOURS' : True,
    'DELTA' : 3,
    'BUSINESS_HOURS' : True,
  },
  
  'ATM_PERSON_TIME_01' : {
    'ZONES' : ['ATM-1', 'ATM-2', 'ATM-3', 'ATM-4'],
    'IS_ALERT' : 'Maybe',
    'MIN_VALUE' : 30,
    'DELTA'     : 600,
    'ALERT_THRESHOLD' : 180,
    'CONTINUOUS' : False,
    'IS_24H'  : True,
    'AGGREGATED' : False, # Is 1-minute-aggregated
    'USE_WORKING_HOURS' : False,
    'BUSINESS_HOURS' : False,
  },
  
  'MULTIZONE_PEOPLE_PRESENCE_CROWDED_01' : {
    'ZONES' : ['CASE', 'GHISEE', 'BT24', 'ZONA-1', 'ZONA-2'],
    'IS_ALERT' : 'Yes',
    'MIN_VALUE' : 7,
    'CONTINUOUS' : False,
    'IS_24H'  : False,
    'AGGREGATED' : False, # Is 1-minute-aggregated
    'USE_WORKING_HOURS' : True,    
    'BUSINESS_HOURS' : True,
  },
}


if __name__ == '__main__':  
  from naeural_core.xperimental.central_data.generator_v1 import DataGenerator as DataGeneratorV1, data_worker, COLUMNS
  from naeural_core.xperimental.central_data.generator_v2 import DataGenerator as DataGeneratorV2
  
  pd.set_option('display.max_rows', 500)
  
  START_DATE = '2024-08-01'
  
  DIRECT_TEST = False
  RUN_TESTS = False
  branches = {}

  for branch in BRANCHES:
    location = deepcopy(SIGNATURES_X_ZONES)
    for signature, details in location.items():
      min_val = details['MIN_VALUE']
      details['MIN_VALUE'] = np.random.randint(min_val, min_val + 3)
    branches[branch] = location
  
  if DIRECT_TEST:
    branch = 'agentia-andrei-muresanu'
    args = (branch, 20, branches[branch], START_DATE, {}, None, True)
    data = data_worker(args)
    df = pd.DataFrame(data).astype(COLUMNS)
  elif RUN_TESTS:
    tests = [
      DataGeneratorV1, #DataGeneratorV2, DataGeneratorV1,
    ]
    dfs = []
    for i, Engine in enumerate(tests):
      eng = Engine(
        start_date=START_DATE,
        branches=branches,
      )
      df = eng.execute(nr_days=20)
      dfs.append(df)
    #end for each engine      
  else:
    eng = DataGeneratorV1(
      start_date=START_DATE,
      branches=branches,
    )
    
    df = eng.execute(nr_days=700)
    
    print(df.head(10))
    print(df.tail(10))
    df.describe()
    df.to_parquet("C:/Dropbox/DATA/__DATASETS/two_years_full_data_for_tableau.parquet", index=False)
  #endif tests or normal generation
  print("Finished generating data.", flush=True)
  df.info()
  df1 = df[
    (df.FUNCTIONALITY == 'PEOPLE_COUNTING_02') & 
    (df.LOCATION.isin(['agentia-andrei-muresanu'])) &
    (df.ZONE == 'INTRARE-1') & 
    (
      ((df.DATETIME > START_DATE + ' 00:00:00') & (df.DATETIME < START_DATE + ' 23:59:59'))
      # | ((df.DATETIME > '2024-09-05 00:00:00') & (df.DATETIME < '2024-09-05 23:59:59')) 
      # | ((df.DATETIME > '2024-09-05 00:00:00') & (df.DATETIME < '2024-09-05 23:59:59')) 
      # | ((df.DATETIME > '2024-09-05 00:00:00') & (df.DATETIME < '2024-09-05 23:59:59')) 
    )
  ]  
  
  print(df1.iloc[:100])
  print(df1.iloc[-100:])  