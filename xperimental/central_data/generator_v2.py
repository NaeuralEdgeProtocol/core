from multiprocessing import Pool, Manager, cpu_count
import pandas as pd
import numpy as np
from time import sleep, time

from .worker import data_worker
from .worker import _COLUMNS as COLUMNS


class DataGenerator:
  def __init__(self, branches, start_date='2021-01-01', debug=True):
    self.debug = debug
    self.start_date = start_date
    self.branches = branches
    branch0 = list(self.branches.keys())[0]
    self.signatures = list(self.branches[branch0].keys())
    return
  
  def initialize_progress(self):
    self.manager = Manager()
    self.progress = self.manager.dict() 
    for branch in self.branches:
      self.progress[branch] = {"current_date": None, "step": 0}
    return

  def execute(self, nr_days: int):
    tm_start = time()
    print("Data generation started...")
    self.nr_days = nr_days

    self.initialize_progress()
            
    nr_processes = cpu_count()
    args = [(
      branch, nr_days, self.branches[branch], self.start_date, self.progress, COLUMNS, 
      True # return_dict
      ) for branch in self.branches]
    
    total_steps = len(self.signatures) * self.nr_days
    with Pool(processes=nr_processes) as pool:
      result_async = pool.map_async(data_worker, args)
      while not result_async.ready():
        if self.debug:
          # Print progress information
          raw_dates = [info['current_date'] for branch, info in self.progress.items()]
          dates = [x for x in raw_dates if x is not None]
          steps = [info['step'] for branch, info in self.progress.items()]
          if len(dates) > 0:
            avg_step = np.sum(steps) / total_steps
            step_prc = avg_step / total_steps * 100
            min_date = min(dates)
            max_date = max(dates)
            msg = "{} branches progressing between {} and {}, progress {:.1f}%".format(
              len(dates), min_date, max_date, step_prc
            )
            print(msg, end='\r')
        sleep(1)
      #end while check progress
      print("\nData generation completed. Getting results", flush=True)
      # results = [r.get() for r in result_async] # method #1
      results = result_async.get()  # method #2
    #end with pool
    print("Aggregating results...", flush=True)
    df = pd.concat([pd.DataFrame(data) for data in results], ignore_index=True).astype(COLUMNS)
    elapsed = time() - tm_start
    print("Data generation took {:.2f} seconds".format(elapsed))
    return df
