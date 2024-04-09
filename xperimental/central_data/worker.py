"""
All:
  - increase 9-12
  - decrease 12-13
  - increase 13-16
  - decrease 16-18

PPL:
  - "INCREMENTAL"  from 0
"""
import pandas as pd
import numpy as np

import datetime
import time


_COLUMNS = {
  'DATETIME': 'datetime64[ns]',
  'FUNCTIONALITY': 'category',
  'LOCATION': 'category',
  'ZONE': 'category',
  'VALUE': 'int32',
  'IS_ALERT': 'int8',
}

WORKING_HOURS = [
  {"start": "08:00", "end": "09:00", "factor" : 0.01},
  {"start": "09:00", "end": "12:30", "factor" : "increase"},
  {"start": "12:30", "end": "14:00", "factor" : 0.05},
  {"start": "14:00", "end": "16:00", "factor" : "increase"},
  {"start": "16:00", "end": "18:00", "factor" : "decrease"},
]

def str_to_time(s):
  str_time = s.split(' ')[-1]
  h, m, s = 0, 0, 0
  if len(str_time) < 6:
    fmt = '%H:%M'
    h, m = str_time.split(':')
  else:
    fmt = '%H:%M:%S'
    h, m, s = str_time.split(':')
  
  ctime = int(h) * 3600 + int(m) * 60 + int(s)
  return ctime


def time_to_str(t):
    hours, rem = divmod(t, 3600)
    minutes, seconds = divmod(rem, 60)
    s = "{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds))
    return s
  

def get_working_hours_factor(timestamp):
  """
  Calculate the working hours factor based on a given timestamp.

  Parameters
  ----------
  timestamp : float
    The timestamp for which to calculate the working hours factor.

  Returns
  -------
  float
    The working hours factor based on the timestamp.
  """
  # Convert timestamp to time for comparison
  result = 0.01  # Default fallback value

  for interval in WORKING_HOURS:
    # Convert start and end times to datetime.time objects
    start = str_to_time(interval['start'])
    end = str_to_time(interval['end'])

    # Check if current time is within the interval
    if start < timestamp <= end:
      # Convert start and end times to datetime.datetime objects for subtraction
      rlen = end - start

      if interval['factor'] == "increase":
        v = (timestamp - start) / rlen
      elif interval['factor'] == "decrease":
        v = (end - timestamp) / rlen
      else:
        # Assuming interval['factor'] can be a direct numerical value
        v = interval['factor']
      #endif 
      result = v
      break
    #endif
  #endfor
  return result

def data_worker(args):
  branch, nr_days, signatures, start_date, shared_data, COLUMNS, return_dict = args
  if COLUMNS is None:
    COLUMNS = _COLUMNS
  if return_dict:
    data = {x: [] for x in COLUMNS}
  else:
    data = []
  step = 0
  START_DAY = str_to_time('08:00')
  END_DAY = str_to_time('18:00')
  end_date = pd.to_datetime(start_date) + pd.Timedelta(days=nr_days)  
  for signature, signature_details in signatures.items():
    only_business_hours = signature_details.get('BUSINESS_HOURS', False)    
    signature_freq = 'B' if only_business_hours else 'D'
    date_range = pd.date_range(start=start_date, end=end_date, freq=signature_freq).astype(str).to_list()
    for single_date in date_range:
      step += 1 
      shared_data[branch] = {"current_date": single_date, "step": step}
      for zone in signature_details['ZONES']:       
        use_working_hours = signature_details.get('USE_WORKING_HOURS', False)
        maybe_zero = signature_details.get('MAYBE_ZERO', False)
        is_incremental = signature_details.get('INCREMENTAL', False)
        moving_value = 0
        min_val = signature_details['MIN_VALUE']
        is_24h = signature_details['IS_24H']
        is_continuous = signature_details['CONTINUOUS']
        is_alert = signature_details['IS_ALERT'].upper()
        is_aggregated = signature_details.get('AGGREGATED', False)
        alert_threshold = signature_details.get('ALERT_THRESHOLD', 0)
        freq = 'T' if is_continuous else np.random.choice(['5T', '10T', '15T'])
        start_time = single_date if is_24h else single_date + ' 08:00:00'
        end_time = single_date + ' 23:59:59' if is_24h else single_date + ' 18:00:00'
        timestamps = pd.date_range(start=start_time, end=end_time, freq=freq).astype(str).to_list()
        for str_timestamp in timestamps:
          # now the value calculation
          ts = str_to_time(str_timestamp)
          delta = signature_details.get('DELTA', 15)
          working_hours_factor = 0
          if use_working_hours:
            working_hours_factor = get_working_hours_factor(ts)
            delta = int(delta * working_hours_factor)

          if not is_continuous:
            is_night = ts < START_DAY or ts > END_DAY
            has_value_proba = 0.1 if is_night else 0.5
            if np.random.rand() > has_value_proba:
              # skip this timestamp
              continue
          #end if not is_continuous
          if not is_aggregated:
            seconds = np.random.randint(0, 59)
            _timestamp = ts + seconds
          else:
            _timestamp = ts
          #end if is_aggregated or not
          
          final_timestamp = single_date.split(' ')[0] + ' ' + time_to_str(_timestamp)
          
          if is_incremental:
            increase = np.random.randint(0, min_val + delta + 1)
            zero_proba = 0.9 if maybe_zero else 0.5
            if maybe_zero and np.random.rand() < zero_proba:
              increase = 0
            moving_value += increase
            value = moving_value
          else:
            value = min_val
            if delta > 0:
              value += np.random.randint(0, delta)
            #end if delta > 0
          #end if is_incremental
          
          if is_alert[0] == 'N':
            alert = 0
          else:
            alert = int(value > alert_threshold)  
          if return_dict:          
            data['DATETIME'].append(final_timestamp)
            data['FUNCTIONALITY'].append(signature)
            data['LOCATION'].append(branch)
            data['ZONE'].append(zone)
            data['VALUE'].append(value)
            data['IS_ALERT'].append(alert)          
          else:
            data.append((final_timestamp, signature, branch, zone, value, alert))
        #end for all timestamps
      #end for each zone
    #end for each signature
  #end for each single_date
  return data


if __name__ == '__main__':
  import json
  from collections import OrderedDict
  timestamps = pd.date_range(start='2024-08-01 08:00:00', end='2024-08-01 18:00:00', freq='30T')
  factors = OrderedDict({str(x) : get_working_hours_factor(x) for x in timestamps})
  print(json.dumps(factors, indent=2))