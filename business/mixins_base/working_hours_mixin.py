
class _WorkingHoursMixin(object):

  def __init__(self):
    self.__is_new_shift = False
    self.__shift_started = False
    super(_WorkingHoursMixin, self).__init__()
    return

  
  @property
  def working_hours(self):
    lst_working_hours = self.cfg_working_hours

    # adapted also in case of having just one interval and mistakenly the operator did not configured list of lists.
    if isinstance(lst_working_hours, list) and len(lst_working_hours) == 2 and isinstance(lst_working_hours[0], str):
      lst_working_hours = [lst_working_hours]

    if isinstance(lst_working_hours, dict):
      lst_working_hours = {
        key.upper(): value
        for key, value in lst_working_hours.items()
      }
    # endif dict

    return lst_working_hours
  
  def _on_shift_start(self, interval_idx, weekday_name=None):
    # TODO: in future implement dict params for instance config in a specific time interval
    hour_schedule = self.working_hours[weekday_name] if weekday_name is not None else self.working_hours
    hrs = 'NON-STOP' if interval_idx is None else hour_schedule[interval_idx]
    shift_str = f'{hrs}' if weekday_name is None else f'{weekday_name}: {hrs}'
    msg = f"Starting new working hours shift {shift_str}"
    info = f"Plugin {self} starting new shift {shift_str}"
    self.P(msg)
    self._create_notification(
      msg=msg,
      info=info,
      displayed=True,
      notif_code=self.ct.NOTIFICATION_CODES.PLUGIN_WORKING_HOURS_SHIFT_START, 
    )
    self.add_payload_by_fields(
      status="Working hours shift STARTED",
      working_hours=self.cfg_working_hours,
      forced_pause=self.cfg_forced_pause,
      ignore_working_hours=self.cfg_ignore_working_hours,
      img=None,
    )
    return

  def _on_shift_end(self):
    msg = "Ended current working hours shift for {}. The full schedule is: {}".format(self, self.working_hours)
    info = "Plugin {} ended its shift shift. The full schedule is: {}".format(self, self.working_hours)
    self.P(msg, color='r')
    self._create_notification(
      msg=msg,
      info=info,
      working_hours=self.cfg_working_hours,
      forced_pause=self.cfg_forced_pause,
      displayed=True,
      notif_code=self.ct.NOTIFICATION_CODES.PLUGIN_WORKING_HOURS_SHIFT_END, 
    )
    self.add_payload_by_fields(
      status="Working hours shift ENDED",
      working_hours=self.cfg_working_hours,
      ignore_working_hours=self.cfg_ignore_working_hours,
      forced_pause=self.cfg_forced_pause,
      img=None,
    )
    return
  
  
  def __get_outside_working_hours(self):
    interval_idx = None
    result = True
    weekday_name = None

 # if the plugin is configured to ignore working hours it will always be considered as inside working hours
    if self.cfg_ignore_working_hours:      
      return False, interval_idx, weekday_name
    
    # if the provided working_hours is None the plugin will always be outside working hours
    if self.working_hours is None:
      return True, interval_idx, weekday_name

    ts_now = self.datetime.now()
    # extracting both the weekday and the hour intervals if it's the case
    lst_hour_schedule, weekday_name = self.log.extract_weekday_schedule(
      ts=ts_now,
      schedule=self.working_hours,
      return_day_name=True
    )
    
    # in case we have schedule based on week days and the current day was not specified
    # it means we are outside the working hours
    
    if lst_hour_schedule is not None:
      # if hour_schedule is an empty list we have 2 cases:
      # 1. The plugin will work non-stop on the current day (if schedule is using week days)
      # 2. The plugin will work non-stop regardless of the week day
      if len(lst_hour_schedule) == 0:
        result = False

      interval_idx = self.log.extract_hour_interval_idx(
        ts=ts_now,
        lst_schedule=lst_hour_schedule
      )
    # endif hour_schedule is not None
    return result, interval_idx, weekday_name
  

  @property
  def outside_working_hours(self):
    result, interval_idx, weekday_name = self.__get_outside_working_hours()

    # interval found or non-stop functioning
    if interval_idx is not None or not result:
      self.__is_new_shift = False
      if not self.__shift_started:
        self.__is_new_shift = True      
        self._on_shift_start(
          weekday_name=weekday_name,
          interval_idx=interval_idx
        )
      # endif mark new shift
      self.__shift_started = True
      result = False
    elif self.__shift_started:
      # shift already started so we close it
      self._on_shift_end()
      self.__shift_started = False
    # endif current time in valid interval
    return result


  @property
  def working_hours_is_new_shift(self):
    return self.__is_new_shift




if __name__ == '__main__':
  from core import Logger

  log = Logger(
    'gigi',
    base_folder='.',
    app_folder='_local_cache'
  )
  class Base(object):
    def __init__(self, **kwargs):
      return

  class P(_WorkingHoursMixin, Base):
    def __init__(self, **kwargs):
      super(P, self).__init__(**kwargs)
      return


  working_hours_tests = [
    [
      ['21:10', '10:10'],
      ['11:10', '11:25']
    ],
    [['10:10', '21:10']]
  ]

  for hours in working_hours_tests:
    log.P(hours)
    p = P()
    p.cfg_working_hours = hours
    p.log = log
    log.P(f'`p.outside_working_hours`={p.outside_working_hours}')
    log.P(f'`p.working_hours_is_new_shift`={p.working_hours_is_new_shift}')
    # print(p.outside_working_hours)
    # print(p.working_hours_is_new_shift)

