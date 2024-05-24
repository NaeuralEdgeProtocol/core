

from transformers import AutoTokenizer as LlamaTokenizer


### LLM constants

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class LlamaCT:
  P_USER_START = B_INST
  P_USER_END = E_INST
  P_ROUND_START = '<s>'
  P_ROUND_END = '</s>'
  P_SYS_START = B_SYS
  P_SYS_END = E_SYS

  HIST = 'history'
  REQ = 'request'
  RES = 'response'
  SYS = 'system_info'

  PRED = 'prediction'
  TEXT = 'text'
  TKNS = 'tokens'
  PRMP = 'prompt'
  TPS  = 'tps'

  # Constants for encoding a prompt using chat templates
  REQUEST_ROLE = 'user'
  REPLY_ROLE = 'assistant'
  SYSTEM_ROLE = 'system'
  ROLE_KEY = 'role'
  DATA_KEY = 'content'

  EE_HF_TOKEN = 'EE_HF_TOKEN'
### END LLM constants


class LlamaTokenizerMixin(object):
  def __init__(self, *args, **kwargs):
    super(LlamaTokenizerMixin, self).__init__(*args, **kwargs)
    return


  def _add_round(self, prompt, request, response=None, system_info=None):
    """
    Manual prompt generation. This is a helper function to generate a prompt

    Parameters
    ----------
    prompt : str
        the initial plain text prompt
    request : str
        the initial request if any that will be aded to the prompt
    response : str, optional
        round response if any, by default None
    system_info : str, optional
        the system prompt if any, by default None

    Returns
    -------
    full prompt as  str
    """
    if prompt is None:
      prompt = ''
    prompt += LlamaCT.P_ROUND_START
    if prompt == LlamaCT.P_ROUND_START and system_info is not None:
      prompt += LlamaCT.P_USER_START
      # add system
      prompt += LlamaCT.P_SYS_START
      prompt += system_info
      prompt += LlamaCT.P_SYS_END
      #end system
    else:
      prompt += LlamaCT.P_USER_START + '\n'
    #endif system info or not
    prompt += request + '\n'
    prompt += LlamaCT.P_USER_END
    # now, if this is a last request we do not have a response and we do not end the round
    if response is not None:
      prompt += '\n' + response + '\n'
      # now end round if we have response
      prompt += LlamaCT.P_ROUND_END
    #endif we have response
    return prompt


  def _get_prompt(self, request, history, system_info):
    """
    Goes through the list history that includes requests and responses
    and using the final request will generate a prompt

    Parameters
    ----------
    request : str
        current request
    history : list[dict]
        full previous history in the same format as for `_get_prompt_from_template`
    system_info : str
        system prompt

    Returns
    -------
    str
        full prompt

    Raises
    ------
    ValueError
        raises error if history format is wrong
    """
    prompt = ''
    # 1. prepare history
    if history is not None and len(history) > 0:
      if not (isinstance(history, list) and isinstance(history[0], dict)):
        msg = "`history` must be a list of dicts. Received {}".format(type(history))
        raise ValueError(msg)
      #endif type check
      for chat_round in history:
        round_request = chat_round.get(LlamaCT.REQ, None)
        round_response = chat_round.get(LlamaCT.RES, None)
        assert round_request is not None, "Each round in `history` must have a `request`"
        assert round_response is not None, "Each round in `history` must have a `response`"
        prompt = self._add_round(
          prompt=prompt,
          request=round_request,
          response=round_response,
          system_info=system_info
        )
      #endfor each round
    #endif we have history
    # 2. prepare request
    assert isinstance(request, str), "`request` must be a string"
    prompt = self._add_round(
      prompt=prompt,
      request=request,
      response=None,
      system_info=system_info,
    )
    return prompt


  def _get_prompt_from_template(self, request, history, system_info):
    """
    Uses Jinja template to generate a prompt.

    Parameters
    ----------
    request : str
        the current request
    history : list[dict]
        the list of previous requests and responses in the same format as for `_get_prompt`
    system_info : str
        the system prompt

    Returns
    -------
    str
        full prompt

    Raises
    ------
    ValueError
        _description_
    """
    chat = []
    if system_info is not None:
      if isinstance(self.tokenizer.chat_template, str):
        template = self.tokenizer.chat_template
        if '\"system\"' in template or "'system'" in template:
          # Only add if the template actually uses the system role
          chat.append({LlamaCT.ROLE_KEY: LlamaCT.SYSTEM_ROLE, LlamaCT.DATA_KEY: system_info})

    #endif create system info

    if history is not None and len(history) > 0:
      if not (isinstance(history, list) and isinstance(history[0], dict)):
        msg = "`history` must be a list of dicts. Received {}".format(type(history))
        raise ValueError(msg)
      #endif type check
      for chat_round in history:
        round_request = chat_round.get(LlamaCT.REQ, None)
        round_response = chat_round.get(LlamaCT.RES, None)
        assert round_request is not None, "Each round in `history` must have a `request`"
        assert round_response is not None, "Each round in `history` must have a `response`"
        chat.append({LlamaCT.ROLE_KEY: LlamaCT.REQUEST_ROLE, LlamaCT.DATA_KEY: round_request})
        chat.append({LlamaCT.ROLE_KEY: LlamaCT.REPLY_ROLE, LlamaCT.DATA_KEY: round_response})
      #endfor chat rounds
    #endif history check

    assert isinstance(request, str), "`request` must be a string"
    chat.append({LlamaCT.ROLE_KEY: LlamaCT.REQUEST_ROLE, LlamaCT.DATA_KEY: request})
    from_template = self.tokenizer.apply_chat_template(chat, tokenize=False)
    return from_template
