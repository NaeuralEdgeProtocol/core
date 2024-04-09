import inspect
import os
import torch as th
import pandas as pd
import numpy as np
import traceback
import abc
from functools import partial
from typing import Any, Dict
from time import time
from core import DecentrAIObject
from core import Logger
from core.local_libraries.nn.th.trainer import ModelTrainer
from PyE2 import _PluginsManagerMixin
from core import constants as ct


class BaseTrainingPipeline(DecentrAIObject, _PluginsManagerMixin):
  score_key = None
  score_mode = None

  data_factory_loc = None
  model_factory_loc = None
  training_callbacks_loc = None

  def __init__(self, log : Logger, signature : str, config : dict, path_to_dataset : str, **kwargs):
    self.config = config
    self.signature = signature

    self._dct_data_factories = {'train' : None, 'dev' : None, 'test' : None}
    self._dct_data_dirs = {'train' : '', 'dev' : '', 'test' : ''}

    self._path_to_dataset = path_to_dataset
    self._dataset_name = os.path.split(self._path_to_dataset.rstrip(os.sep))[-1]# .split(os.sep)[-1]

    self._data_factory_ref = None
    self._model_factory_ref = None
    self._training_callbacks_ref = None

    self._trainer = None
    self._trainer_callbacks_obj = None

    self._status_per_grid_idx = {}
    self._top_k_best_model_idx = None
    self._time_start_run = None
    self._grid_loop_exception = False

    self.df_model_name_to_hiperparams : pd.DataFrame = None

    self._status : Dict[str, Any] = {
      'ELAPSED' : None,
      'REMAINING' : None,
      'GRID_ITER' : None,
      'NR_ALL_GRID_ITER' : None,
      'GRID_RESULTS' : None,
      'BEST' : None,
    }

    self._metadata : Dict[str, Any] = {
      'CLASSES' : None,
      'MODEL_ARCHITECTURE_PATH' : None,
      'MODEL_NAME' : None,
      'FIRST_STAGE_TARGET_CLASS' : None,
      'INFERENCE_PREPROCESS_DEFINITIONS' : None,
    }
    super(BaseTrainingPipeline, self).__init__(log=log, **kwargs)
    return

  def startup(self):
    def _append_if_not_void(lst, x):
      if x is not None:
        lst.append(x)
      return

    super().startup()

    if self.score_key is None or self.score_mode is None:
      raise ValueError("score_key/score_mode") # TODO

    self._find_data_directories(self._path_to_dataset)

    if len(self._dct_data_dirs['train']) == 0 or len(self._dct_data_dirs['dev']) == 0:
      raise ValueError("train/dev not existing") # TODO

    data_factory_loc = ['local_libraries.nn.th.training.data']
    _append_if_not_void(data_factory_loc, self.data_factory_loc)

    model_factory_loc = ['local_libraries.nn.th.training.models']
    _append_if_not_void(model_factory_loc, self.model_factory_loc)

    training_callbacks_loc = ['local_libraries.nn.th.training.callbacks']
    _append_if_not_void(training_callbacks_loc, self.training_callbacks_loc)

    self._data_factory_ref = self._get_module_name_and_class(
      locations=data_factory_loc,
      name=self.signature,
      suffix='DataLoaderFactory'
    )

    self._model_factory_ref = self._get_module_name_and_class(
      locations=model_factory_loc,
      name=self.signature,
      suffix='ModelFactory'
    )

    self._training_callbacks_ref = self._get_module_name_and_class(
      locations=training_callbacks_loc,
      name=self.signature,
      suffix='TrainingCallbacks'
    )

    self._metadata['MODEL_ARCHITECTURE_PATH'] = '{}.{}'.format(
      self._model_factory_ref[0].__name__,
      self._model_factory_ref[1],
    )

    self._metadata['MODEL_NAME'] = self.cfg_model_name
    self._metadata['FIRST_STAGE_TARGET_CLASS'] = self.cfg_first_stage_target_class

    self.experiment_subfolder = 'training_' + self.log.session_id + '_{}'.format(self.cfg_model_name)

    self.experiment_models_folder = os.path.join(
      self.log.get_models_folder(),
      self.experiment_subfolder
    )

    return

  @abc.abstractmethod
  def model_loss(self, **dct_grid_option):
    """
    Defines the model loss. Grid parameters can be used.

    Parameters:
    ----------
    **dct_grid_option
      Current grid option

    Returns:
    -------
    loss_obj
      A loss object that can be called with 2 parameters (y_pred, y) and returns the final loss.
      Most of the times, torch losses will be used (e.g. th.nn.CrossEntropyLoss() )
    """
    raise NotImplementedError

  @property
  def cfg_batch_index(self):
    return self.config.get('BATCH_INDEX', True)

  @property
  def cfg_num_workers(self):
    return self.config.get("NUM_WORKERS", None)

  @property
  def cfg_model_name(self):
    return self.config['MODEL_NAME']

  @property
  def cfg_preload_data(self):
    return self.config.get('PRELOAD_DATA', True)

  @property
  def cfg_device_load_data(self):
    return self.config.get('DEVICE_LOAD_DATA', 'cuda:0')

  @property
  def cfg_device_training(self):
    return self.config.get('DEVICE_TRAINING', 'cuda:0')

  @property
  def cfg_start_grid_idx(self):
    return self.config.get('START_GRID_IDX', None)

  @property
  def cfg_end_grid_idx(self):
    return self.config.get('END_GRID_IDX', None)

  @property
  def cfg_grid_search(self):
    """
    Property that lets the pipeline to use other grid search, not the one defined in the architecture.
    """
    return self.config.get('GRID_SEARCH', {})

  def get_grid_search(self):
    """
    Getter for the grid search dictionary.
    This is used in order to be able to override it in a child class.
    Returns
    -------
    dict
    """
    return self.cfg_grid_search

  @property
  def cfg_batch_size(self):
    return self.config['BATCH_SIZE']

  @property
  def cfg_epochs(self):
    return self.config['EPOCHS']

  @property
  def cfg_first_stage_target_class(self):
    return self.config.get('FIRST_STAGE_TARGET_CLASS', None)

  @property
  def cfg_keep_top_k_iterations(self):
    """
    Property that tells the pipeline to save only the best K models in a grid search in order to save disk space.
    """
    return self.config.get('KEEP_TOP_K_ITERATIONS', None)

  @property
  def default_grid_search(self):
    dct = self._model_factory_ref[3] or {}
    return dct.get('GRID_SEARCH', {})

  @property
  def model_factory_class_def(self):
    return self._model_factory_ref[2]

  @property
  def training_callbacks_class_def(self):
    return self._training_callbacks_ref[2]

  @property
  def status(self):
    if self._time_start_run is not None:
      self._status['ELAPSED'] = time() - self._time_start_run

    if self.df_model_name_to_hiperparams is not None:
      self._status['GRID_RESULTS'] = self.df_model_name_to_hiperparams.to_dict()

    if self._top_k_best_model_idx is not None:
      self._status['BEST'] = self._status_per_grid_idx[self._top_k_best_model_idx[0]]

    return self._status

  @property
  def metadata(self):
    if self._dct_data_factories['train'] is not None:
      train_data_factory = self._dct_data_factories['train']
      if self._metadata.get('CLASSES', None) is None:
        self._metadata['CLASSES'] = train_data_factory.dataset_info['class_to_id']
      # endif

      preprocess_definitions = train_data_factory.preprocess_definitions
      self._metadata['INFERENCE_PREPROCESS_DEFINITIONS'] = [
        [x[0].__module__ + '.' + x[0].__name__, x[1]]
        for x in preprocess_definitions
      ]
    #endif

    return self._metadata

  @property
  def grid_has_finished(self):
    grid_has_finished = False
    if self._status['NR_ALL_GRID_ITER'] is not None:
      grid_has_finished = (self._status['GRID_ITER'] == self._status['NR_ALL_GRID_ITER'])
    return grid_has_finished or self._grid_loop_exception

  def _find_data_directories(self, path_to_dataset):
    def _recursive_search_dirs(path_to_dataset, base_path=''):
      dirs = list(filter(
        lambda x: os.path.isdir(os.path.join(path_to_dataset, x)),
        os.listdir(path_to_dataset)
      ))

      subdirs = [os.path.join(base_path, d) for d in dirs]
      if len(dirs) > 0:
        for d in dirs:
          subdirs += _recursive_search_dirs(os.path.join(path_to_dataset, d), os.path.join(base_path, d))

      return subdirs

    dirs = _recursive_search_dirs(path_to_dataset)

    for k in self._dct_data_dirs.keys():
      for d in dirs:
        if k in d.lower():
          self._dct_data_dirs[k] = os.path.join(path_to_dataset, d)
          # now we break early because we found the first apparition of the 'k' dataset
          break
        # endif
      # endfor
    # endfor

    for k, v in self._dct_data_dirs.items():
      if len(v) > 0:
        self.P("'{}' dataset is found at '{}'".format(k, v), color='g')
      else:
        self.P("'{}' dataset is not available".format(k), color='r')

    return

  def _create_data_factories(self, image_height=None, image_width=None, **kwargs):
    class_def = self._data_factory_ref[2]

    for data_subset, path in self._dct_data_dirs.items():
      if len(path) == 0:
        continue

      obj = class_def(
        log=self.log,
        load_observations=self.cfg_preload_data,
        batch_size=self.cfg_batch_size,
        path_to_dataset=path,
        data_subset_name=data_subset,
        image_height=image_height,
        image_width=image_width,
        load_device=self.cfg_device_load_data,
        training_device=self.cfg_device_training,
        num_workers=self.cfg_num_workers,
        **kwargs,
      )
      obj.create()
      self._dct_data_factories[data_subset] = obj
    #endfor
    return

  def _release_data_factories(self):
    for factory in self._dct_data_factories.values():
      if factory is None:
        continue

      factory.release_device()
    #endfor
    return

  def _update_data_factories(self, **kwargs):
    for factory in self._dct_data_factories.values():
      if factory is None:
        continue

      factory.update_data_factory(**kwargs)
    # endfor


  def _maybe_test(self):
    res = None
    if self._dct_data_factories['test'] is not None:
      res = self._trainer_callbacks_obj.test_callback(
        data_generator=self._dct_data_factories['test'].data_loader,
        dataset_info=self._dct_data_factories['test'].dataset_info,
      )
    #endif

    return res

  def _maybe_dev(self):
    res = None
    if self._dct_data_factories['dev'] is not None:
      res = self._trainer_callbacks_obj.dev_callback(
        data_generator=self._dct_data_factories['dev'].data_loader,
        dataset_info=self._dct_data_factories['dev'].dataset_info,
        key='dev',
        epoch=0
      )

    # endif

    return res

  def _collect_dev_results(self, dct_grid_option, model_name, res=None):
    if res is None:
      res = self._trainer.epochs_data.values()
    else:
      columns = res.keys()
      res = res.values()
    model_results = pd.DataFrame([res])
    model_results.columns = columns
    row = model_results[model_results[self.score_key] == self._trainer.score_eval_func(model_results[self.score_key])]
    # subfolder_path = 'training_' + self.log.session_id + '_{}'.format(self.cfg_model_name)

    self.log.save_dataframe(
      df=model_results,
      fn=f'{model_name}',
      folder='output',
      subfolder_path=self.experiment_subfolder
    )

    if self.df_model_name_to_hiperparams is None:
      self.df_model_name_to_hiperparams = pd.DataFrame(
        columns=['model_name', 'model_idx', *list(dct_grid_option.keys()), *row.columns]
      )
    # endif

    self.df_model_name_to_hiperparams = pd.concat([self.df_model_name_to_hiperparams,
      pd.DataFrame([{
        'model_name': f'{model_name}',
        **dct_grid_option,
        **row.to_dict('records')[0]
      }])],
      ignore_index=True
    )

    self.df_model_name_to_hiperparams.sort_values(by=self.score_key, ascending=(self.score_mode == 'min'), inplace=True)
    self._top_k_best_model_idx = self.df_model_name_to_hiperparams['model_idx'].values[:self.cfg_keep_top_k_iterations]
    self.log.save_dataframe(
      df=self.df_model_name_to_hiperparams,
      fn='00_grid_{}_{}'.format(self._dataset_name, self.cfg_model_name),
      folder='output',
      subfolder_path=self.experiment_subfolder
    )

    self.P("Results so far:\n{}".format(self.df_model_name_to_hiperparams))
    return

  def _collect_results(self, dct_grid_option, i):
    model_results = pd.DataFrame(self._trainer.epochs_data.values())
    row = model_results[model_results[self.score_key] == self._trainer.score_eval_func(model_results[self.score_key])]
    # subfolder_path = 'training_' + self.log.session_id + '_{}'.format(self.cfg_model_name)

    self.log.save_dataframe(
      df=model_results,
      fn=f'{self.cfg_model_name}_{i}',
      folder='output',
      subfolder_path=self.experiment_subfolder
    )

    if self.df_model_name_to_hiperparams is None:
      self.df_model_name_to_hiperparams = pd.DataFrame(
        columns=['model_name', 'model_idx', *list(dct_grid_option.keys()), *row.columns]
      )
    # endif

    self.df_model_name_to_hiperparams = pd.concat([self.df_model_name_to_hiperparams,
      pd.DataFrame([{
        'model_name': f'{self.cfg_model_name}_{i}',
        'model_idx': i,
        **dct_grid_option,
        **row.to_dict('records')[0]
      }])],
      ignore_index=True
    )

    self.df_model_name_to_hiperparams.sort_values(by=self.score_key, ascending=(self.score_mode == 'min'), inplace=True)
    self._top_k_best_model_idx = self.df_model_name_to_hiperparams['model_idx'].values[:self.cfg_keep_top_k_iterations]
    self.log.save_dataframe(
      df=self.df_model_name_to_hiperparams,
      fn='00_grid_{}_{}'.format(self._dataset_name, self.cfg_model_name),
      folder='output',
      subfolder_path=self.experiment_subfolder
    )

    self.P("Results so far:\n{}".format(self.df_model_name_to_hiperparams))
    return

  def _fit(self, model, model_name, model_loss, trainer_kwargs, callbacks_kwargs):

    self._trainer = ModelTrainer(
      log=self.log,
      model=model,
      model_name=model_name,
      losses=model_loss,
      batch_size=self.cfg_batch_size,
      device=self.cfg_device_training,
      score_key=self.score_key,
      score_mode=self.score_mode,
      base_folder=self.experiment_models_folder,
      **trainer_kwargs
    )

    self._trainer_callbacks_obj = self.training_callbacks_class_def(
      log=self.log,
      owner=self._trainer,
      preprocess_before_fw_callback=self._dct_data_factories['train'].preprocess_before_forward,
      training_device=self.cfg_device_training,
      **callbacks_kwargs
    )

    self._trainer.train_on_batch = self._trainer_callbacks_obj.train_on_batch_callback

    self._trainer.evaluate = partial(
      self._trainer_callbacks_obj.evaluate_callback,
      data_generator=self._dct_data_factories['dev'].data_loader,
      dataset_info=self._dct_data_factories['dev'].dataset_info,
      key='dev'
    )

    self._trainer.evaluate_train = partial(
      self._trainer_callbacks_obj.evaluate_callback,
      data_generator=self._dct_data_factories['train'].data_loader,
      dataset_info=self._dct_data_factories['train'].dataset_info,
      key='train'
    )

    self._trainer.fit(
      th_dl=self._dct_data_factories['train'].data_loader,
      epochs=self.cfg_epochs,
      batch_index=self.cfg_batch_index
    )

    return

  def _disk_cleanup(self):
    for idx, status in self._status_per_grid_idx.items():
      if idx not in self._top_k_best_model_idx:
        try:
          os.remove(status['best_file'])
          self.P("Model files saved at iteration {} were deleted because they are not in top k best anymore.".format(idx), color='y')
        except FileNotFoundError:
          pass

        try:
          os.remove(status['best_file'] + '.optim')
        except FileNotFoundError:
          pass

    return

  def _save_model_def(self, model_kwargs, model_name):
    if not os.path.exists(self.experiment_models_folder):
      os.makedirs(self.experiment_models_folder)
    model_def_fn = "{}_def.json".format(model_name)
    self.log.save_json(
      model_kwargs,
      os.path.join(self.experiment_models_folder, model_def_fn)
    )
    return

  def get_pretrained_weights_path(self, pretrained_weights_url):
    if os.path.isfile(pretrained_weights_url):
      pretrained_weights_path = pretrained_weights_url
    else:
      # TODO: review this in order to be able to use minio
      try:
        saved_files, _ = self.log.maybe_download(
          url=pretrained_weights_url,
          fn=f'{self.cfg_model_name}_pretrained_weights.th',
          target='models',
          unzip=False
        )
        pretrained_weights_path = saved_files[0]
      except Exception as e:
        self.log.P(f'Could not download pretrained weights from {pretrained_weights_url}', color='r')
        pretrained_weights_path = None
      # endtry download
    # endif pretrained_weights_url is a local file
    return pretrained_weights_path

  def run(self, start_iter=0, end_iter=None):
    self._time_start_run = time()
    self._grid_loop_exception = False
    dct_grid_search = self.get_grid_search() or self.default_grid_search
    lst_data_params = dct_grid_search.get('DATA_PARAMS', [])
    lst_callbacks_params = dct_grid_search.get('CALLBACKS_PARAMS', [])
    lst_trainer_params = dct_grid_search.get('TRAINER_PARAMS', [])
    lst_model_params = dct_grid_search.get('MODEL_PARAMS', None)
    lst_exceptions = dct_grid_search.get('EXCEPTIONS', None)
    lst_fixed = dct_grid_search.get('FIXED', None)
    lst_keep_non_model_params = dct_grid_search.get('KEEP_NON_MODEL_PARAMS', [])
    all_grid_options = self.log.get_grid_iterations(
      params_grid=dct_grid_search['GRID'],
      priority_keys=lst_data_params,
      exceptions=lst_exceptions,
      fixed=lst_fixed,
      verbose=1
    )

    if self.cfg_end_grid_idx is not None:
      all_grid_options = all_grid_options[:self.cfg_end_grid_idx]
    if self.cfg_start_grid_idx is not None:
      all_grid_options = all_grid_options[self.cfg_start_grid_idx:]

    self._status['NR_ALL_GRID_ITER'] = len(all_grid_options)
    self.P("Grid search for model base name: {}".format(self.cfg_model_name), color='b')

    lst_non_model_params = lst_callbacks_params + lst_trainer_params + lst_data_params
    if lst_model_params is None:
      lst_model_params = list((set(list(dct_grid_search['GRID'].keys())) - set(lst_non_model_params)).union(set(lst_keep_non_model_params)))

    iter_times = []
    iteration = 0
    last_data_params = None
    if end_iter is None:
      end_iter = len(all_grid_options)
    try:
      for i, dct_grid_option in enumerate(all_grid_options):
        if i < start_iter:
          continue
        if i >= end_iter:
          continue
        iteration = i+1
        time_start_iter = time()
        self.log.reset_seeds(seed=42)
        model_name = self.cfg_model_name + '_{}'.format(i)
        lst_grid_dict = ["{}:{}".format(k, v) for k, v in dct_grid_option.items()]
        str_grid_dict = "  ".join(lst_grid_dict)
        self.P("Training grid option {}/{}: {}".format(iteration, len(all_grid_options), str_grid_dict), color='g')

        data_kwargs = {k: dct_grid_option[k] for k in lst_data_params}
        if last_data_params != data_kwargs:
          if last_data_params is None:
            self._create_data_factories(**data_kwargs)
          else:
            try:
              # Try to update data factories
              self._update_data_factories(**data_kwargs)
            except NotImplementedError:
              # If update method not implemented -> reload them
              self._release_data_factories()
              self._create_data_factories(**data_kwargs)
            #end try/except
          #endif
          last_data_params = data_kwargs
        #endif

        trainer_kwargs = {k : dct_grid_option[k] for k in lst_trainer_params}
        callbacks_kwargs = {k : dct_grid_option[k] for k in lst_callbacks_params}
        model_kwargs = {k: dct_grid_option[k] for k in lst_model_params}

        self._save_model_def(model_kwargs, model_name)
        if inspect.getfullargspec(self.model_factory_class_def.__init__).varkw is not None:
          model_kwargs = dct_grid_option

        model = self.model_factory_class_def(**model_kwargs).to(self.cfg_device_training)
        model_loss = self.model_loss(**dct_grid_option)
        skip_iteration = False
        pretrained_weights_url = trainer_kwargs.get('pretrained_weights', None)
        if pretrained_weights_url is not None:
          pretrained_weights_path = self.get_pretrained_weights_path(pretrained_weights_url)
          if pretrained_weights_path is not None:
            try:
              model.load_state_dict(th.load(pretrained_weights_path, map_location=self.cfg_device_training))
              self.log.P(f'Successfully loaded pretrained weights from {pretrained_weights_path}', color='g')
            except Exception as e:
              skip_iteration = True
              self.log.P(
                f'Could not load pretrained weights from {pretrained_weights_path}. Skipping current iteration',
                color='r')
            # endtry load pretrained weights
          # endif pretrained weights successfully located
        # endif pretrained_weights_url

        if not skip_iteration:
          self._fit(
            model=model,
            model_name=model_name,
            model_loss=model_loss,
            trainer_kwargs=trainer_kwargs,
            callbacks_kwargs=callbacks_kwargs
          )

          self._status_per_grid_idx[i] = self._trainer.training_status.copy()
          self._status_per_grid_idx[i]['model_kwargs'] = model_kwargs
          self._status_per_grid_idx[i]['dct_grid_option'] = dct_grid_option

          dct_test_results = self._maybe_test()
          if dct_test_results is not None:
            self._trainer.epochs_data[self._trainer.training_status['best_epoch']] = {
              **self._trainer.epochs_data[self._trainer.training_status['best_epoch']],
              **dct_test_results
            }
          self._collect_results(dct_grid_option, i)
          iter_times.append(time()-time_start_iter)
          self._status['GRID_ITER'] = iteration
          self._status['REMAINING'] = np.mean(iter_times) * (self._status['NR_ALL_GRID_ITER'] - self._status['GRID_ITER'])
        # endif skip_iteration
        self._disk_cleanup()
        model.cpu()
        del model
      #endfor
    except ct.ForceStopException as e:
      self.P("Grid search stopped at iteration {}/{}: {}".format(
        iteration,
        self._status['NR_ALL_GRID_ITER'],
        e
      ))
    except Exception as e:
      msg = "Grid Exception at iteration {}/{}:\n{}\n{}".format(
        iteration,
        self._status['NR_ALL_GRID_ITER'],
        traceback.format_exc(),
        self.log.get_error_info(return_err_val=True)
      )
      self.P(msg, color='r')
      self._grid_loop_exception = True
    #end try-except

    self._release_data_factories()

    return
