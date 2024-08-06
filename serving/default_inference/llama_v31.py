"""
@misc{touvron2023llamaopenefficientfoundation,
      title={LLaMA: Open and Efficient Foundation Language Models},
      author={Hugo Touvron and Thibaut Lavril and Gautier Izacard and Xavier Martinet and Marie-Anne Lachaux and Timothée Lacroix and Baptiste Rozière and Naman Goyal and Eric Hambro and Faisal Azhar and Aurelien Rodriguez and Armand Joulin and Edouard Grave and Guillaume Lample},
      year={2023},
      eprint={2302.13971},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2302.13971},
}

models:
  meta-llama/Meta-Llama-3.1-8B
  meta-llama/Meta-Llama-3.1-8B-Instruct

  meta-llama/Meta-Llama-3.1-70B
  meta-llama/Meta-Llama-3.1-70B-Instruct


  meta-llama/Meta-Llama-3.1-405B
  meta-llama/Meta-Llama-3.1-405B-FP8
  meta-llama/Meta-Llama-3.1-405B-Instruct
  meta-llama/Meta-Llama-3.1-405B-Instruct-FP8


Testing:
  A. Launch OnDemandTextInput with Explorer
  B. Write custom command (see below)



for llama3.1 in-filling:
```json
{
  "ACTION" : "PIPELINE_COMMAND",
  "PAYLOAD" : {
    "NAME": "code-on-demand",
    "PIPELINE_COMMAND" : {
      "STRUCT_DATA" : {
        "request" : "def hello_world(<FILL>\n\nhello_world()",
        "history" : [
          {
            "request"   : "print('hello",
            "response"  : " world')"
          }
        ],
        "system_info" : "You are a funny python programmer assistant. your task is to complete the code you are given. return only the completion, not the whole program."
      }
    }
  }
}
```
"""

from core.serving.base.base_llm_serving import BaseLlmServing as BaseServingProcess

__VER__ = '0.1.0.0'

_CONFIG = {
  **BaseServingProcess.CONFIG,

  "MODEL_NAME": "meta-llama/Meta-Llama-3.1-8B-Instruct",

  "PICKED_INPUT": "STRUCT_DATA",
  "RUNS_ON_EMPTY_INPUT": False,

  'VALIDATION_RULES': {
    **BaseServingProcess.CONFIG['VALIDATION_RULES'],
  },

}


class Llama_v31(BaseServingProcess):
  def __init__(self, **kwargs):
    self._counter = 0
    super(Llama_v31, self).__init__(**kwargs)
    return

  def _setup_llm(self):
    # just override this method as the base class has a virtual method that raises an exception
    return
