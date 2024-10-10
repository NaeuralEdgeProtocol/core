"""
Documentation:
https://global-technical.atlassian.net/wiki/spaces/DOCUMENTAT/pages/527335425/Configurare
https://global-technical.atlassian.net/wiki/spaces/DOCUMENTAT/pages/526614579/Exemple
"""
#global dependencies
import uuid

from functools import partial

#local dependencies
from naeural_core import constants as ct

from naeural_core.business.base import CVPluginExecutor

_CONFIG = {
  **CVPluginExecutor.CONFIG,
  'VALIDATION_RULES': {
    **CVPluginExecutor.CONFIG['VALIDATION_RULES'],
  },
}

__VER__ = '0.1.0.0'


class REPC: # RuleEnginePluginConstans
  COND_ALERT = 'ALERT'
  COND_DETECT = 'DETECT'
  COND_INPUTS = 'INPUTS'
  INPUT_IMG = 'IMG'
  INPUT_STRUCT_DATA = 'STRUCT_DATA'
  VALID_STRUCT_DATA_OPERATORS = ['==', '!=', '>', '<', '>=', '<=']
  VALID_CONDITIONS = [COND_ALERT, COND_DETECT, COND_INPUTS]
  OPERATOR_AND = 'AND'
  OPERATOR_OR = 'OR'

  IMG_VALUE_NONE = 'NONE'
  IMG_VALUE_NOT_NONE = 'NOT-NONE'

  GLOBAL_PLUGIN_ALERTER = 'global_plugin_alerter'

  PAYLOAD_KEY_IS_ALERT = 'IS_ALERT'
  PAYLOAD_KEY_IMG = 'IMG'


class RuleEnginePlugin(CVPluginExecutor):
  CONFIG = _CONFIG
  def __init__(self, **kwargs):
    self._ops_tree = None
    self._handle_callbacks = [
      self._handle_alert,
      self._handle_detect,
      self._handle_inputs_img,
      self._handle_inputs_struct_data,
      # if a new condition should be developed, create a new handler, a new op and add the handler here
    ]
    super(RuleEnginePlugin, self).__init__(**kwargs)
    return

  def startup(self):
    super().startup()

    self.alerter_create(alerter=REPC.GLOBAL_PLUGIN_ALERTER, **self.cfg_global_alerter_params)
    try:
      self._ops_tree = self._parse_conditions(self.cfg_conditions)
    except:
      pass

    return

  @property
  def cfg_conditions(self):
    return self._instance_config.get('CONDITIONS', [])

  @property
  def cfg_alert_on_static_payload(self):
    return self._instance_config.get('ALERT_ON_STATIC_PAYLOAD', {})

  @property
  def cfg_alert_off_static_payload(self):
    return self._instance_config.get('ALERT_OFF_STATIC_PAYLOAD', {})

  @property
  def cfg_dynamic_payload_keys(self):
    return self._instance_config.get('DYNAMIC_PAYLOAD_KEYS', [])

  @property
  def cfg_global_alerter_params(self):
    return self._instance_config.get('GLOBAL_ALERTER_PARAMS', {'raise_time': 0, 'lower_time': 0, 'value_count': 1})

  def _handle_alert(self, condition):
    if condition[0] != REPC.COND_ALERT:
      return
    assert len(condition) == 3
    assert isinstance(condition[1], str)
    assert isinstance(condition[2], dict)
    assert 'raise_time' in condition[2]
    assert 'lower_time' in condition[2]
    assert 'value_count' in condition[2]

    obj_name = condition[1]
    alerter = obj_name + '_' + str(uuid.uuid4())
    self.alerter_create(alerter=alerter, **condition[2])
    op = partial(self._op_alert, alerter_name=alerter, obj_name=obj_name)
    return op

  def _handle_detect(self, condition):
    if condition[0] != REPC.COND_DETECT:
      return
    assert len(condition) == 2
    assert isinstance(condition[1], str)
    obj_name = condition[1]
    op = partial(self._op_detect, obj_name=obj_name)
    return op

  def _handle_inputs_img(self, condition):
    if condition[0] != REPC.COND_INPUTS:
      return

    if condition[1] != REPC.INPUT_IMG:
      return

    assert len(condition) == 3
    assert condition[2] in [REPC.IMG_VALUE_NONE, REPC.IMG_VALUE_NOT_NONE]
    value = condition[2]
    op = partial(self._op_inputs_img, value=value)
    return op

  def _handle_inputs_struct_data(self, condition):
    if condition[0] != REPC.COND_INPUTS:
      return

    if condition[1] != REPC.INPUT_STRUCT_DATA:
      return

    assert len(condition) == 5
    sub_key = condition[2]
    operator = condition[3]
    value = condition[4]
    assert operator in REPC.VALID_STRUCT_DATA_OPERATORS
    op = partial(self._op_inputs_struct_data, sub_key=sub_key, operator=operator, value=value)
    return op

  def _op_alert(self, alerter_name, obj_name):
    max_prc = 0
    for model, lst_inferences_per_substream in self.dataapi_images_instance_inferences().items():
      for inf in self.log.flatten_2d_list(lst_inferences_per_substream):
        if obj_name == inf.get(ct.TYPE, None):
          max_prc = max(max_prc, inf[ct.PROB_PRC])

    self.alerter_add_observation(max_prc, alerter=alerter_name)

    is_alert = self.alerter_is_alert(alerter=alerter_name)
    return is_alert

  def _op_detect(self, obj_name):
    for model, lst_inferences_per_substream in self.dataapi_images_instance_inferences().items():
      for inf in self.log.flatten_2d_list(lst_inferences_per_substream):
        if obj_name == inf.get(ct.TYPE, None):
          return True

    return False

  def _op_inputs_img(self, value):
    # a condition is true when any of the INPUTS complies;
    for _input in self.dataapi_inputs():
      if value == REPC.IMG_VALUE_NONE and _input['IMG'] is None:
        return True

      if value == REPC.IMG_VALUE_NOT_NONE and _input['IMG'] is not None:
        return True
    #endfor

    return False

  def _op_inputs_struct_data(self, sub_key, operator, value):
    # a condition is true when any of the INPUTS complies;
    for _input in self.dataapi_inputs():
      if not isinstance(_input['STRUCT_DATA'], dict):
        continue

      struct_data_evaluation = eval('{}{}{}'.format(
        _input['STRUCT_DATA'].get(sub_key, None),
        operator,
        value
      ))

      if struct_data_evaluation:
        return True
    #endfor

    return False

  def _parse_conditions(self, conditions):
    if not conditions:
      return []

    crt = conditions[0]

    if isinstance(crt, str):
      if crt.upper() == REPC.OPERATOR_AND:
        return [REPC.OPERATOR_AND.lower()] + self._parse_conditions(conditions[1:])
      elif crt.upper() == REPC.OPERATOR_OR:
        return [REPC.OPERATOR_OR.lower()] + self._parse_conditions(conditions[1:])
      else:
        err_msg = 'PLUGIN ERROR! Unknown condition operator {} in rule engine.'.format(crt)
        self.P(err_msg, color='r')
        self._create_notification(
          notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
          msg=err_msg
        )
        raise ValueError(err_msg)
      # endif
    # endif

    if isinstance(crt, list) and isinstance(crt[0], str):
      op = None
      for handler in self._handle_callbacks:
        op = handler(crt)
        if op is not None:
          assert callable(op)
          break

      if op is not None:
        return [op] + self._parse_conditions(conditions[1:])
      else:
        err_msg = 'PLUGIN ERROR! Unknown condition keyword {} in rule engine.'.format(crt[0])
        self._create_notification(
          notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
          msg=err_msg
        )
        raise ValueError(err_msg)
      # endif
    # endif

    if isinstance(crt, list) and isinstance(crt[0], list):
      return [self._parse_conditions(crt)] + self._parse_conditions(conditions[1:])

    err_msg = "PLUGIN ERROR! Something went wrong in rule engine's `_parse_conditions`. Needs investigation."
    self._create_notification(
      notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
      msg=err_msg
    )
    raise ValueError(err_msg)

  def _parse_ops(self, ops):
    if not ops:
      return ''

    crt = ops[0]

    if crt in [REPC.OPERATOR_AND.lower(), REPC.OPERATOR_OR.lower()]:
      return ' ' + crt + ' ' + self._parse_ops(ops[1:])

    if callable(crt):
      return str(crt()) + self._parse_ops(ops[1:])

    if isinstance(crt, list):
      return '(' + self._parse_ops(crt) + ')' + self._parse_ops(ops[1:])

    err_msg = "PLUGIN ERROR! Something went wrong in rule engine's `_parse_ops`. Needs investigation."
    self._create_notification(
      notif=ct.STATUS_TYPE.STATUS_EXCEPTION,
      msg=err_msg
    )
    raise ValueError(err_msg)

  def _get_value_for_payload_key(self, payload_key):
    value = None
    if payload_key == REPC.PAYLOAD_KEY_IMG:
      value = self.dataapi_image()
    else:
      pass

    return value

  def _draw_witness_image(self):
    return

  def _process(self):
    expression = self._parse_ops(self._ops_tree)
    evaluation = eval(expression)

    self.alerter_add_observation(value=evaluation, alerter=REPC.GLOBAL_PLUGIN_ALERTER)

    payload = None
    if self.alerter_is_new_raise(REPC.GLOBAL_PLUGIN_ALERTER):
      payload = self._create_payload(
        is_alert=True,
        is_new_raise=True,
        is_new_lower=False,
        plugin_category='alertable',
        **self.cfg_alert_on_static_payload
      )
      self.P("Raised new alert for {}".format(self.unique_identification), color='r')
    elif self.alerter_is_new_lower(REPC.GLOBAL_PLUGIN_ALERTER):
      payload = self._create_payload(
        is_alert=False,
        is_new_raise=False,
        is_new_lower=True,
        plugin_category='alertable',
        **self.cfg_alert_off_static_payload
      )
      self.P("Lower alert for {}".format(self.unique_identification), color='g')

    if payload is not None:
      for payload_key in self.cfg_dynamic_payload_keys:
        value = self._get_value_for_payload_key(payload_key)
        vars(payload)[payload_key] = value

    return payload

# if __name__ == '__main__':
#
#   from naeural_core import Logger
#   log = Logger(lib_name='RE', base_folder='.', app_folder='_local_cache', TF_KERAS=False)
#   p = RuleEnginePlugin(
#     log=log,
#     stream_id='NONE',
#     signature='rule_engine',
#     default_config=_CONFIG,
#     upstream_config={},
#     shared_data={},
#     config_upload=None,
#   )
#
#   p.process()
