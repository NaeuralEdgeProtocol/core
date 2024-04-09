from multiprocessing import Pool, Manager, cpu_count
import pandas as pd
import numpy as np
import datetime
import time

from .worker import data_worker
from .worker import _COLUMNS as COLUMNS



class DataGenerator:
  def __init__(self, branches, start_date='2021-01-01', debug=True):
    self.debug = debug
    self.start_date = start_date
    self.branches = branches
    manager = self.maybe_create_manager()
    self.progress = manager.dict() 
    branch0 = list(self.branches.keys())[0]
    self.signatures = list(self.branches[branch0].keys())
    return
  
  
  def maybe_create_manager(self):
    # now create class-level manager that will be used to share progress information
    try:
      _process_manager = getattr(type(self), '_process_manager')
    except AttributeError:
      # create a new manager only once
      _process_manager = type(self)._process_manager = Manager()
    return _process_manager


  def initialize_progress(self):
    for branch in self.branches:
      self.progress[branch] = {"current_date": None, "step": 0}
    return


  def update_progress(self, branch, current_date, step):
    self.progress[branch] = {"current_date": current_date.strftime('%Y-%m-%d'), "step": step}
    return


  def generate_branch_data(self, args):
    branch, nr_days = args    
    data_worker_args = (
      branch, nr_days, self.branches[branch], self.start_date, self.progress, COLUMNS,
      False, # return_dict
    )
    data = data_worker(data_worker_args)
    return data


  def execute(self, nr_days: int):
    tm_start = time.time()
    print("Data generation started for {} days and {} features...".format(nr_days, len(self.signatures)), flush=True)
    self.nr_days = nr_days
    self.initialize_progress()
    args = [(branch, self.nr_days) for branch in self.branches]
    
    nr_processes = cpu_count()
    total_steps = len(self.signatures) * self.nr_days
    
    with Pool(processes=nr_processes) as pool:
      # One approach would be:
      # result_async = [pool.apply_async(self.generate_branch_data, args=(arg,)) for arg in args]
      # while not all([r.ready() for r in result_async]):
      #   # etc
      # Another approach
      result_async = pool.map_async(self.generate_branch_data, args)
      while not result_async.ready():
      #   result_async.wait(5)  # Wait a bit for completion
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
            msg = "{} branches progressing between {} and {}, progress: {:.1f}%".format(
              len(dates), min_date, max_date, step_prc
            )
            print(msg, end='\r')
        time.sleep(1)
      #end while check progress
      print("\nData generation completed. Getting results", flush=True)
      # results = [r.get() for r in result_async] # method #1
      results = result_async.get()  # method #2
    #end with pool
    print("Aggregating results...", flush=True)
    all_data = [item for sublist in results for item in sublist]
    columns = ['DATETIME', 'FUNCTIONALITY', 'LOCATION', 'ZONE', 'VALUE', 'IS_ALERT']
    df = pd.DataFrame(all_data, columns=columns).astype(COLUMNS)
    elapsed = time.time() - tm_start
    print("Data generation completed in {:.2f} seconds".format(elapsed))
    return df

