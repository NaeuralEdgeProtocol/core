from naeural_core.business.base.drivers import Device
from naeural_core.business.base.drivers.utils.nut2 import PyNUTClient

_CONFIG = {
  **Device.CONFIG,

  'ALLOW_EMPTY_INPUTS': True,

  "PROCESS_DELAY": 10,

  'DEVICE_IP': None,

  'DEVICE_NUT_DRIVER': None,

  'DEVICE_NUT_PORT': 3493,

  'DEVICE_NUT_LOGIN': None,

  'DEVICE_NUT_PASSWORD': None,

  'DEVICE_NUT_TIMEOUT': 5,  # the number of seconds to wait for a response from the NUT server

  'DEVICE_NUT_NAME': None,

  'DEVICE_BATTERY_CHARGE_THRESHOLD_SOFT': 70,  # percentage

  'DEVICE_BATTERY_CHARGE_THRESHOLD_HARD': 50,  # percentage

  'DEVICE_BATTERY_VOLTAGE_DEVIATION': 0.10,    # the deviation in voltage, 10% deviation is allowed

  'VALIDATION_RULES': {
    **Device.CONFIG['VALIDATION_RULES'],
  },

  "DEVICE_DEBUG": True,
}


class UPSConstants:
  UPS_DEVICE_BATTERY_CHARGE = "battery.charge"
  UPS_DEVICE_BATTERY_VOLTAGE = "battery.voltage"
  UPS_DEVICE_BATTERY_VOLTAGE_HIGH = "battery.voltage.high"
  UPS_DEVICE_BATTERY_VOLTAGE_LOW = "battery.voltage.low"
  UPS_DEVICE_BATTERY_VOLTAGE_NOMINAL = "battery.voltage.nominal"
  UPS_DEVICE_BATTERY_CURRENT = "battery.current"
  UPS_STATUS = "ups.status"
  UPS_LOAD = "ups.load"


class UPSCustomDevice(Device):
  """
      A custom device class for managing UPS devices.

      Subclasses can implement the following methods to handle specific events:

      def _on_battery_charge_change(self, status=None, charge=None):
          '''
          Called when the battery charge changes.
          Parameters:
              status (str): The current UPS status.
              charge (int): The new battery charge percentage.
          '''

      def _on_ups_status_change(self, status=None, battery_charge=None):
          '''
          Called when the UPS status changes.
          Parameters:
              status (str): The new UPS status.
          '''

      def _on_ups_below_threshold(self, status=None, soft_threshold=True, charge=None):
          '''
          Called when the battery charge goes below a threshold.
          Parameters:
              status (str): The current UPS status.
              soft_threshold (bool): True if the soft threshold is reached, False otherwise.
              charge (int): The current battery charge percentage.
          '''

      def _on_battery_voltage_abnormal(self, voltage=None, nominal_voltage=None):
          '''
          Called when the battery voltage is outside the normal range.
          Parameters:
              voltage (float): The current battery voltage.
              nominal_voltage (float): The nominal battery voltage.
          '''
      """
  CONFIG = _CONFIG

  def __init__(self, **kwargs):
    super(UPSCustomDevice, self).__init__(**kwargs)
    self.__ups = None
    self.__ups_connected = False
    self._ups_battery_voltage_nominal = None
    self._ups_battery_voltage_high = None
    self._ups_battery_voltage_low = None
    return

  def __attempt_reconnect(self):
    """
    Attempt to reconnect to the UPS device
    Returns
    -------
    """
    try:
      self.__ups = PyNUTClient(
        host=self.cfg_device_ip, port=self.cfg_device_nut_port,
        login=self.cfg_device_nut_login, password=self.cfg_device_nut_password
      )

      ups_vars = self.__ups.list_vars(self.cfg_device_nut_name)
      self.__cache_initial_values(ups_vars)

      self.__ups_connected = True
    except Exception as e:
      self.__ups = None
      self.__ups_connected = False

      if self.cfg_device_debug:
        self.P("Failed to reconnect to UPS device {} : \n {}".format(str(e), self.trace_info()), color='red')
      # end if debug
      self._create_error_payload(message="Failed to reconnect to UPS device")
    return

  def __cache_initial_values(self, ups_vars):
    """
    Cache the initial values of the UPS variables
    Parameters
    ----------
    ups_vars

    Returns
    -------

    """
    # Cache the battery charge for 60 seconds
    self.device_set_cache(
      path=UPSConstants.UPS_DEVICE_BATTERY_CHARGE,
      data=ups_vars.get(UPSConstants.UPS_DEVICE_BATTERY_CHARGE), seconds=60
    )

    # Cache the UPS status for 60 seconds
    self.device_set_cache(
      path=UPSConstants.UPS_STATUS, data=ups_vars.get(UPSConstants.UPS_STATUS), seconds=60
    )

    ups_battery_voltage_nominal = ups_vars.get(UPSConstants.UPS_DEVICE_BATTERY_VOLTAGE_NOMINAL)
    if isinstance(ups_battery_voltage_nominal, str):
      self._ups_battery_voltage_nominal = float(ups_battery_voltage_nominal)
    # end if nominal voltage

    ups_battery_voltage_high = ups_vars.get(UPSConstants.UPS_DEVICE_BATTERY_VOLTAGE_HIGH)
    if isinstance(ups_battery_voltage_high, str):
      self._ups_battery_voltage_high = float(ups_battery_voltage_high)
    # end if high voltage

    ups_battery_voltage_low = ups_vars.get(UPSConstants.UPS_DEVICE_BATTERY_VOLTAGE_LOW)
    if isinstance(ups_battery_voltage_low, str):
      self._ups_battery_voltage_low = float(ups_battery_voltage_low)

    return

  def _device_battery_charge_below_soft_threshold_check(self, status=None, charge=None):
    if hasattr(self, "_on_ups_below_threshold"):
      self._on_ups_below_threshold(status=status, soft_threshold=True, charge=charge)
      return
    # end if charge below threshold callback

    if self.cfg_device_debug:
      self.P("UPS battery charge below soft threshold")
    # end if debug battery charge below threshold

    if status is not None and charge is not None:
      self._create_error_payload(message=f"UPS is currently {status} and below soft threshold: {charge}%")
    # end if status and charge are not None

    return

  def _device_battery_charge_below_hard_threshold_check(self, status=None, charge=None):
    if hasattr(self, "_on_ups_below_threshold"):
      self._on_ups_below_threshold(status=status, soft_threshold=False, charge=charge)
      return
    # end if charge below threshold callback

    if self.cfg_device_debug:
      self.P("UPS battery charge below hard threshold")
    # end if debug battery charge below threshold

    if status is not None and charge is not None:
      self._create_error_payload(message=f"UPS is currently {status} and below hard threshold: {charge}%")
    # end if status and charge are not None

    return

  def _device_battery_change(self):
    ups_charge = self.device_ups_get_variable(UPSConstants.UPS_DEVICE_BATTERY_CHARGE)
    if ups_charge is None:
      self.P("Failed to get UPS battery charge", color='red')
      return

    charge = int(ups_charge)
    cached_charge = self.device_get_cache(UPSConstants.UPS_DEVICE_BATTERY_CHARGE)
    status = self.device_get_cache(UPSConstants.UPS_STATUS)

    # if the cached value does not equal the current value, update the cache
    if cached_charge is None or cached_charge != charge:
      self.device_set_cache(UPSConstants.UPS_DEVICE_BATTERY_CHARGE, charge, 60)
      if hasattr(self, "_on_battery_charge_change"):
        self._on_battery_charge_change(status=status, charge=charge)
        return
    # end if cached_charge is None

    if charge is not None:
      if int(charge) < int(self.cfg_device_battery_charge_threshold_soft):
        self._device_battery_charge_below_soft_threshold_check(status=status, charge=charge)
      # end if charge below soft threshold

      if int(charge) < int(self.cfg_device_battery_charge_threshold_hard):
        self._device_battery_charge_below_hard_threshold_check(status=status, charge=charge)

      if self.cfg_device_debug:
        self.P("UPS battery charge: {}%".format(charge))
    # end if charge is not None
    return charge

  def _device_battery_voltage_check(self):
    """
    Check the battery voltage against the nominal voltage

    Returns
    -------

    """
    usp_current_voltage = self.device_ups_get_variable(UPSConstants.UPS_DEVICE_BATTERY_VOLTAGE)
    ups_on_load = self.device_ups_get_variable(UPSConstants.UPS_LOAD)

    if isinstance(usp_current_voltage, str) and isinstance(ups_on_load, str):
      current_voltage = float(usp_current_voltage)
      voltage_deviation = current_voltage * self.cfg_device_battery_voltage_deviation  # 10% deviation (default)
      num_of_loads = int(ups_on_load)

      # Calculate the acceptable voltage range considering both the specific thresholds and the 20% deviation
      acceptable_low = max(self._ups_battery_voltage_low, current_voltage - voltage_deviation)
      acceptable_high = min(self._ups_battery_voltage_high, current_voltage + voltage_deviation)

      if not acceptable_low <= current_voltage <= acceptable_high and num_of_loads > 0:
        # Voltage is outside the normal operating range
        if hasattr(self, "_on_battery_voltage_abnormal"):
          self._on_battery_voltage_abnormal(current_voltage, self._ups_battery_voltage_nominal, num_of_loads)
          return
        # end if abnormal voltage callback

        message = (f"UPS battery voltage is abnormal: {current_voltage}V (nominal: {self._ups_battery_voltage_nominal}V, "
                   f"range: {self._ups_battery_voltage_low}-{self._ups_battery_voltage_high}V)")
        if self.cfg_device_debug:
          self.P(message, color='yellow')
        # end if debug
        self._create_error_payload(message=message)
      # end if voltage is outside the normal operating range
    # end if current voltage is a string

  def _device_online_status(self):
    status = self.device_ups_get_variable(UPSConstants.UPS_STATUS)
    cached_status = self.device_get_cache(UPSConstants.UPS_STATUS)
    charge = self.device_get_cache(UPSConstants.UPS_DEVICE_BATTERY_CHARGE)

    # if the cached value does not equal the current value, update the cache
    if cached_status is None or cached_status != status:
      self.device_set_cache(UPSConstants.UPS_STATUS, status, 60)

      # handle callback for status change
      if hasattr(self, "_on_ups_status_change"):
        self._on_ups_status_change(status=status, battery_charge=charge)
        return
      # end if status change callback

      if self.cfg_device_debug:
        self.P(f"UPS status: {status}")
    return status

  def device_ups_get_variable(self, variable):
    """
    Get the UPS status
    Parameters
    ----------
    variable : str
    Returns
    -------
    """
    # check if variable is one of the constants defined in UPSConstants
    if variable not in UPSConstants.__dict__.values():
      message = "Invalid variable: {}".format(variable)
      if self.cfg_device_debug:
        self.P(message)
      self._create_error_payload(message=message)
      return None

    if self.__ups_connected:
      try:
        return self.__ups.get_var(self.cfg_device_nut_name, variable)
      except Exception as e:
        message = "Error getting variable: {}".format(str(e))
        if self.cfg_device_debug:
          self.P(message)
        self._create_error_payload(message=message)
    return None

  def on_init(self):
    """
    Initialize the UPS device
    Returns
    -------
    """
    self.__attempt_reconnect()
    return

  def process(self):
    """
    Process the UPS device
    Returns
    -------

    """
    charge = None
    status = None

    if not self.__ups_connected:
      self.__attempt_reconnect()
    # end if not ups connected

    if self.__ups_connected:
      charge = self._device_battery_change()
      status = self._device_online_status()
      self._device_battery_voltage_check()

      if self.cfg_device_debug:
        self.P("UPS status: {} with battery percentage of {}%".format(status, charge))
      # end if debug
    # end if ups connected

    if self.cfg_device_debug:
      payload = {"status": status, "charge": charge}
      self._create_device_state_payload(state=payload)

    return
