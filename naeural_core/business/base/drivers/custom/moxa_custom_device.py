from naeural_core.business.base.drivers import HttpDevice
from naeural_core.business.base.drivers.utils.json_util import JsonUtil


class MoxaCT:
  READ_INPUT_PINS = "/api/slot/0/io/di"
  READ_RELAY_PINS = "/api/slot/0/io/relay"
  READ_SYS_INFO = "/api/slot/0/sysInfo"
  SET_PULSE_RELAY_STATE = "/api/slot/0/io/relay/{}/relayPulseStatus"
  SET_RELAY_STATE = "/api/slot/0/io/relay/{}/relayStatus"


class MoxaCustomDevice(HttpDevice):
  def __init__(self, *args, **kwargs):
    super(MoxaCustomDevice, self).__init__(**kwargs)
    self._headers = {"Accept": "vdn.dac.v1", "Content-Type": "application/json"}
    self.__system_info = self.NestedDotDict({"timestamp": 0, "device_model": None, "uptime": None})
    self.__digital_pins = []
    self.__relay_pins = []
    self.__debouncer = {}
    return

  def _moxa_debounce_value(self, index, current_value, timeout):
    """
    Debounce the value of a pin to avoid false positives
    Parameters
    ----------
    index : int
    current_value : int
    timeout : int

    Returns
    -------
    The debounced value
      int
    """
    if index not in self.__debouncer:
      self.__debouncer[index] = {
        "value": current_value,
        "timestamp": self.time()
      }
      return current_value
    else:
      elapsed = self.time() - self.__debouncer[index]["timestamp"]
      if elapsed >= timeout:  # all the values that are inside the timeout interval are ignored
        self.__debouncer[index] = {
          "value": current_value,
          "timestamp": self.time()
        }
    # endif pin index not in debounce
    return self.__debouncer[index]["value"]

  def _moxa_update_system_info(self):
    """
    Update the system information and obtain the device model and uptime
    Returns
    -------

    """
    try:
      path = MoxaCT.READ_SYS_INFO
      result = self.device_get(path=path, headers=self._headers)
      if result is None or "sysInfo" not in result:
        return

      device = self.NestedDotDict(result["sysInfo"]["device"][0])
      self.__system_info.device_model = device.modelName
      self.__system_info.uptime = device.deviceUpTime
      self.__system_info.timestamp = self.time()
      self.P(f"System info updated {self.__system_info}", color="green")
    except Exception as e:
      if self.cfg_device_debug:
        self.P(f"Error updating system info {e}", color="red")
    return

  def _moxa_update_digital_pins(self):
    """
    Update the digital pins and debounce the values to avoid false positives
    Returns
    -------
    list
      The debounced digital pins
    None if there was an error
    """
    try:
      path = MoxaCT.READ_INPUT_PINS
      result = self.device_get(path=path, headers=self._headers)
      if result is None or "io" not in result and "di" not in result["io"]:
        return

      # convert the result to a nested dot dict
      result_dct = self.NestedDotDict(result)
      debounced = []  # Create a new list to store the debounced values

      for pin in result_dct.io.di:
        if pin.diMode == 0:
          pin["diStatus"] = self._moxa_debounce_value(index=pin.diIndex, current_value=pin.diStatus,
                                                      timeout=self.cfg_device_debounce)
        debounced.append(pin)

      return debounced
    except Exception as e:
      if self.cfg_device_debug:
        self.P(f"Error updating digital pins {e}", color="red")
    return

  def _moxa_update_relay_pins(self):
    """
    Update the relay pins, fetch the values from the device
    Returns
    -------
    list
      The relay pins
    None if there was an error
    """
    try:
      path = MoxaCT.READ_RELAY_PINS
      result = self.device_get(path=path, headers=self._headers)
      if result is None or "io" not in result or "relay" not in result["io"]:
        return

      pins = self.NestedDotDict(result)
      return pins.io.relay
    except Exception as e:
      if self.cfg_device_debug:
        self.P(f"Error updating relay pins {e}", color="red")
    return

  def __moxa_create_relay_payload(self, relay_mode: int, relay_index: int, relay_value: int):
    """
    Create the payload for the relay, this will depend on the mode of the relay
    Parameters
    ----------
    relay_mode : int
    relay_index : int
    relay_value : int

    Returns
    -------
    dict
    """
    key = "relayPulseStatus" if relay_mode == 1 else "relayStatus"
    return {
      "slot": 0,
      "io": {
        "relay": {
          str(relay_index): {
            str(key): int(relay_value),
          }
        }
      }
    }

  def _moxa_set_relay_pin(self, index, value):
    """
    Set the value of a relay pin
    Parameters
    ----------
    index : int
    value : int

    Returns
    -------

    """
    try:
      for pin in self.__relay_pins:
        if pin.relayIndex == index:
          path = MoxaCT.SET_PULSE_RELAY_STATE.format(
            pin.relayIndex) if pin.relayMode == 1 else MoxaCT.SET_RELAY_STATE.format(
            pin.relayIndex)
          payload = self.__moxa_create_relay_payload(relay_mode=pin.relayMode, relay_index=pin.relayIndex,
                                                     relay_value=value)
          self.device_put(path=path, body=payload, headers=None)
    except Exception as e:
      if self.cfg_device_debug:
        self.P(f"Error setting relay pin {e}", color="red")
      self._create_error_payload(message=f"Error setting relay pin {e}")
    return

  def _has_relays_changed(self, relays: list, keys_to_ignore: list = None) -> bool:
    """
    Check if the relays have changed, compare the relays with the cached relays
    Parameters
    ----------
    relays
    keys_to_ignore

    Returns
    -------
    bool
      True if the relays have changed, False otherwise
    """
    if isinstance(relays, list):
      if keys_to_ignore is None:
        keys_to_ignore = []
      cached_relays = self.__relay_pins
      if not JsonUtil.compare_json(relays, cached_relays, keys_to_ignore=keys_to_ignore):
        return True
    return False

  def _has_inputs_changed(self, inputs: list, keys_to_ignore: list = None) -> bool:
    """
    Check if the inputs have changed, compare the inputs with the cached inputs
    Parameters
    ----------
    inputs
    keys_to_ignore

    Returns
    -------
    bool
      True if the inputs have changed, False otherwise
    """
    if isinstance(inputs, list):
      if keys_to_ignore is None:
        keys_to_ignore = []
      cached_inputs = self.__digital_pins
      if not JsonUtil.compare_json(inputs, cached_inputs, keys_to_ignore=keys_to_ignore):
        return True
    return False

  def _create_status_message(self):
    """
    Create the status message
    Parameters
    ----------

    Returns
    -------
    dict
    """
    message = {
      "modelName": self.__system_info.device_model,
      "deviceUpTime": self.__system_info.uptime,
      "inputs": self.__digital_pins,
      "relays": self.__relay_pins
    }

    return message

  def on_init(self):
    try:
      self._moxa_update_system_info()
    except Exception as e:
      self.P(f"Error updating system info {e}")

  def moxa_get_digital_pins(self):
    """
    Get the digital pins for the device
    Returns
    -------
    list
    """
    return self.__digital_pins

  def moxa_get_relay_pins(self):
    """
    Get the relay pins for the device
    Returns
    -------
    list
    """
    return self.__relay_pins

  def process(self):
    """
    This method runs on an interval defined by the plugin configuration
    Queries the device for the digital and relay pins and sends the payload to the client
    if the pins have changed

    Returns
    -------

    """
    should_send = False

    updated_digital_pins = self._moxa_update_digital_pins()
    if isinstance(updated_digital_pins, list) and self._has_inputs_changed(updated_digital_pins):
      should_send = True
      self.__digital_pins = updated_digital_pins

    updated_relay_pins = self._moxa_update_relay_pins()
    if isinstance(updated_relay_pins, list) and self._has_relays_changed(updated_relay_pins):
      should_send = True
      self.__relay_pins = updated_relay_pins

    if should_send:
      moxa_state = self._create_status_message()
      payload = self._create_payload(moxa_state=moxa_state)
      return payload

    return
