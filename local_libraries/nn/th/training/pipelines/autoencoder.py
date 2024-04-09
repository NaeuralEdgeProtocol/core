import torch as th
from core.local_libraries.nn.th.training.pipelines.base import BaseTrainingPipeline

class AutoencoderTrainingPipeline(BaseTrainingPipeline):
  score_key = 'dev_loss'
  score_mode = 'min'
  model_loss = th.nn.BCELoss()
