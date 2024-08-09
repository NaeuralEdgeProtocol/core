from core.business.base import BasePluginExecutor as BasePlugin

__VER__ = '0.1.0.0'

_CONFIG = {

  # mandatory area
  **BasePlugin.CONFIG,

  # our overwritten props
  'AI_ENGINE': "llm",
  'OBJECT_TYPE': [],
  'PROCESS_DELAY': 1,
  'ALLOW_EMPTY_INPUTS': False,  # if this is set to true the on-idle will be triggered continously the process

  'VALIDATION_RULES': {
    **BasePlugin.CONFIG['VALIDATION_RULES'],
  },
}


class LlmAgentPlugin(BasePlugin):
  CONFIG = _CONFIG

  def _process(self):
    # we always receive input from the upstream due to the fact that _process
    # is called only when we have input based on ALLOW_EMPTY_INPUTS=False
    data = self.dataapi_struct_data()
    self.P(f"Received request:\n{self.json_dumps(data, indent=2)}")
    inferences = self.dataapi_struct_data_inferences()
    text_responses = [inf.get('text') for inf in inferences]
    model_name = inferences[0].get('MODEL_NAME', None) if len(inferences) > 0 else None
    payload = self._create_payload(
      data=data,
      inferences=inferences,
      request_id=data.get('request_id', None),
      text_responses=text_responses,
      model_name=model_name,
    )
    return payload
