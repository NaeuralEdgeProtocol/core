
if __name__ == '__main__':
  from naeural_core import Logger

  Logger.set_nice_prints()
  import pandas as pd
  import os

  root = '_local_cache/_output/comm_telemetry'
  session_id = '20220718_080622'
  event_type = ['distress_01', 'fallen_person_02']
  dfs = []
  parts = list(filter(lambda x: x.endswith('.csv'), os.listdir(os.path.join(root, session_id))))
  parts.sort()
  date_columns = ['t1_cap_time', 't2_plugin_time', 't3_comm_added_in_buff', 't4_comm_before_send', 't5_comm_after_send']
  for p in parts:
    dfs.append(pd.read_csv(os.path.join(root, session_id, p), parse_dates=date_columns))

  df = pd.concat(dfs).reset_index()
  df.drop(columns=['business_data', 'successful_send', 'message_uuid', 'demo_mode', 'stream_id', 'instance_id'],
          inplace=True)
  df_event = df[df.event_type.isin(event_type)].copy()

  df_event['t2mt1'] = df_event.t2_plugin_time - df_event.t1_cap_time
  df_event['t3mt2'] = df_event.t3_comm_added_in_buff - df_event.t2_plugin_time
  df_event['t4mt3'] = df_event.t4_comm_before_send - df_event.t3_comm_added_in_buff
  df_event['t5mt4'] = df_event.t5_comm_after_send - df_event.t4_comm_before_send
  df_event['t5mt1'] = df_event.t5_comm_after_send - df_event.t1_cap_time
  df_event.drop(columns=date_columns, inplace=True)
  timedelta_columns = ['t2mt1', 't3mt2', 't4mt3', 't5mt4', 't5mt1']
  for col in timedelta_columns:
    df_event[col] = df_event[col].map(lambda x: x.total_seconds())
  df_e = df_event.copy()

  from naeural_core import Logger

  Logger.set_nice_prints()
  import pandas as pd
  import os
  
  if False:
    root = '_local_cache/_output/comm_telemetry'
    session_id = '20220624_170425'
    event_type = 'ai_ext_01'
    dfs = []
    parts = list(filter(lambda x: x.endswith('.csv'), os.listdir(os.path.join(root, session_id))))
    parts.sort()
    date_columns = ['t1_cap_time', 't2_plugin_added_in_buff', 't3_plugin_before_exec', 't4_plugin_time',
                    't5_comm_added_in_buff', 't6_comm_before_send', 't7_comm_after_send']
    for p in parts:
      dfs.append(pd.read_csv(os.path.join(root, session_id, p), parse_dates=date_columns))

    df = pd.concat(dfs).reset_index()
    df.drop(columns=['business_data', 'successful_send', 'message_uuid', 'demo_mode', 'stream_id', 'instance_id'],
            inplace=True)
    df_event = df[df.event_type == event_type].copy()

    df_event['t2mt1'] = df_event.t2_plugin_added_in_buff - df_event.t1_cap_time
    df_event['t3mt2'] = df_event.t3_plugin_before_exec - df_event.t2_plugin_added_in_buff
    df_event['t4mt3'] = df_event.t4_plugin_time - df_event.t3_plugin_before_exec
    df_event['t5mt4'] = df_event.t5_comm_added_in_buff - df_event.t4_plugin_time
    df_event['t6mt5'] = df_event.t6_comm_before_send - df_event.t5_comm_added_in_buff
    if False:
      df_event['t7mt6'] = df_event.t7_comm_after_send - df_event.t6_comm_before_send
    df_event['t7mt1'] = df_event.t7_comm_after_send - df_event.t1_cap_time
    df_event.drop(columns=date_columns, inplace=True)
    timedelta_columns = ['t2mt1', 't3mt2', 't4mt3', 't5mt4', 't6mt5', 't7mt1']
    for col in timedelta_columns:
      df_event[col] = df_event[col].map(lambda x: x.total_seconds())
    df_e = df_event.copy()