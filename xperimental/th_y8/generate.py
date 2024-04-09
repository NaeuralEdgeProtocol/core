"""
[Y8][23-03-31 20:50:43] Timing results at 2023-03-31 20:50:43:
[Y8][23-03-31 20:50:43] Section 'main' last seen 0.0s ago
[Y8][23-03-31 20:50:43]  y5l6 = 0.0429s/q:0.0429s/nz:0.0429s, max: 0.0468s, lst: 0.0429s, c: 100/L:6%
[Y8][23-03-31 20:50:43]    y5l6_nms = 0.0245s/q:0.0245s/nz:0.0245s, max: 0.0296s, lst: 0.0270s, c: 100/L:14%
[Y8][23-03-31 20:50:43]      y5l6_nms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0004s, lst: 0.0001s, c: 100/L:100%
[Y8][23-03-31 20:50:43]  y5l6_inclnms = 0.0433s/q:0.0433s/nz:0.0433s, max: 0.0471s, lst: 0.0426s, c: 100/L:9%
[Y8][23-03-31 20:50:43]    y5l6_inclnms_cpu = 0.0002s/q:0.0002s/nz:0.0199s, max: 0.0006s, lst: 0.0001s, c: 100/L:99%
[Y8][23-03-31 20:50:43]  y5s6 = 0.0126s/q:0.0126s/nz:0.0126s, max: 0.0307s, lst: 0.0283s, c: 100/L:0%
[Y8][23-03-31 20:50:43]    y5s6_nms = 0.0017s/q:0.0017s/nz:0.0017s, max: 0.0037s, lst: 0.0023s, c: 100/L:0%
[Y8][23-03-31 20:50:43]      y5s6_nms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0004s, lst: 0.0002s, c: 100/L:100%
[Y8][23-03-31 20:50:43]  y5s6_inclnms = 0.0123s/q:0.0123s/nz:0.0123s, max: 0.0271s, lst: 0.0234s, c: 100/L:0%
[Y8][23-03-31 20:50:43]    y5s6_inclnms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0003s, lst: 0.0002s, c: 100/L:100%
[Y8][23-03-31 20:50:43]  y8l = 0.0329s/q:0.0329s/nz:0.0329s, max: 0.0356s, lst: 0.0326s, c: 100/L:11%
[Y8][23-03-31 20:50:43]    y8l_nms = 0.0187s/q:0.0187s/nz:0.0187s, max: 0.0223s, lst: 0.0203s, c: 100/L:12%
[Y8][23-03-31 20:50:43]      y8l_nms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0004s, lst: 0.0002s, c: 100/L:100%
[Y8][23-03-31 20:50:43]  y8l_inclnms = 0.0329s/q:0.0329s/nz:0.0329s, max: 0.0371s, lst: 0.0329s, c: 100/L:4%
[Y8][23-03-31 20:50:43]    y8l_inclnms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0005s, lst: 0.0002s, c: 100/L:100%
[Y8][23-03-31 20:50:43]  y8s = 0.0110s/q:0.0110s/nz:0.0110s, max: 0.0268s, lst: 0.0094s, c: 100/L:0%
[Y8][23-03-31 20:50:43]    y8s_nms = 0.0018s/q:0.0018s/nz:0.0018s, max: 0.0035s, lst: 0.0016s, c: 100/L:0%
[Y8][23-03-31 20:50:43]      y8s_nms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0003s, lst: 0.0002s, c: 100/L:100%
[Y8][23-03-31 20:50:43]  y8s_inclnms = 0.0100s/q:0.0100s/nz:0.0100s, max: 0.0254s, lst: 0.0089s, c: 100/L:0%
[Y8][23-03-31 20:50:43]    y8s_inclnms_cpu = 0.0002s/q:0.0002s/nz:-1.0000s, max: 0.0003s, lst: 0.0001s, c: 100/L:100%
[Y8][23-03-31 20:50:43] Section 'LOGGER_internal' last seen 0.0s ago
[Y8][23-03-31 20:50:43]  _logger = 0.0030s/q:0.0032s/nz:0.0032s, max: 0.0047s, lst: 0.0020s, c: 122/L:17%
[Y8][23-03-31 20:50:43]    _logger_add_log = 0.0010s/q:0.0011s/nz:0.0011s, max: 0.0019s, lst: 0.0006s, c: 122/L:18%
[Y8][23-03-31 20:50:43]    _logger_save_log = 0.0019s/q:0.0021s/nz:0.0021s, max: 0.0035s, lst: 0.0014s, c: 122/L:17%
[Y8][23-03-31 20:50:43] Model y5l6 gain -0.00047s, equal: True
[Y8][23-03-31 20:50:43] Model y5s6 gain 0.00031s, equal: True
[Y8][23-03-31 20:50:43] Model y8l gain -0.00007s, equal: True
[Y8][23-03-31 20:50:43] Model y8s gain 0.00097s, equal: True


pip install ultralytics[export]


TODO:
    - finish comparision nms y5 vs y8
    - add nms in y5 DONE
    - add nms in y8 DONE
    - optimize nms5
    - optimize nms8
    - resave, load and send to testing!
    
"""
import numpy as np
import os
import sys
import torch as th
import torchvision as tv
import json
import cv2

import time 

try:
  from ultralytics import YOLO
except:
  print("WARNING: Only TESTS and GENERATE_FULL will work !", flush=True)

from core import Logger
from decentra_vision.draw_utils import DrawUtils
from core.local_libraries.nn.th.utils import th_resize_with_pad
from plugins.serving.architectures.y5.general import scale_coords

from core.xperimental.th_y8.utils import predict
from core.xperimental.th_y8.utils import Y5, Y8
import gc
  


if __name__ == "__main__":
  
  GENERATE = False
  GENERATE_FULL = True
  GENERATE_TOPK = True
  TOP_K_VALUES = [False, True] if GENERATE_TOPK else [False]
  generated_models = []
  TRANSFER_CLASSNAMES = False
  CLASSNAMES = None
  CLASSNAMES_DONATE_FROM = r'C:\repos\edge-node\_local_cache\_models\20230723_y8l_nms.ths'
  TEST = True
  TEST_FP16 = False
  FP_VALUES = [False, True] if TEST_FP16 else [False]
  SHOW = True
  SHOW_LABELS = True
  DEVICE = 'cuda'
  dev = th.device(DEVICE)  
  N_TESTS = 100 if 'cuda' in DEVICE.lower() else 10

  top_candidates = 6

  if CLASSNAMES is None:
    extra_files = {'config.txt': ''}
    model = th.jit.load(CLASSNAMES_DONATE_FROM, map_location='cpu', _extra_files=extra_files)
    CLASSNAMES = json.loads(extra_files['config.txt'])['names']
  # end if CLASSNAMES is None

  log = Logger('Y8', base_folder='.', app_folder='_local_cache')
  log.P("Using {}".format(dev))
  models_folder = log.get_models_folder()
  model_date = time.strftime('%Y%m%d', time.localtime(time.time()))
  pyver = sys.version.split()[0].replace('.','')
  thver = th.__version__.replace('.','')
  # MODELS = [
  #   {
  #     'model'   : 'yolov8l',
  #     'imgsz'   : [640, 896],
  #   },
  #   {
  #     'model'   : 'yolov8s',
  #     'imgsz'   : [448, 640],
  #   },
  # ]
  folder = os.path.split(__file__)[0]
  # os.chdir(folder)
  # img = cv2.imread(os.path.join(folder,'bus.jpg'))
  img_names = [
    'bus.jpg',
    'faces9.jpg',
    'faces21.jpg',
    'img.png',
    '688220.png',
    'bmw_man3.png',
    'faces3.jpg',
    'LP3.jpg',
    'pic1_crop.jpg'
  ]

  # assert img is not None
  origs = [
    cv2.imread(os.path.join(folder, im_name))
    for im_name in img_names
  ]
  images = origs
  # images = [
  #   np.ascontiguousarray(img[:,:,::-1])
  #   for img in origs
  # ]
  
  # if GENERATE:
  #   for dct_model in MODELS:
  #     model = YOLO(dct_model['model'] + '.pt')
  #     success = model.export(format="torchscript", imgsz=dct_model['imgsz'])
  #     fn = "{}_{}_{}x{}_th{}_py{}.ths".format(
  #       model_date,
  #       dct_model['model'].replace('yolov','y'),
  #       dct_model['imgsz'][0],dct_model['imgsz'][1],
  #       thver, pyver,
  #     )
  #     fn = os.path.join(models_folder, fn)
  #     os.rename(success, fn)
  #     os.remove(dct_model['model'] + '.pt')
  #     dct_model['pts'] = fn
      
  # prepare models either for .thes or/and testing
  models_for_nms_prep = [
    {
      'pts': os.path.join(models_folder, x),
      'model': (
        x[x.find('lpd'):] if 'lpd' in x else x[x.find('y'):]
      ).split('.ths')[0].split('.torchscript')[0]
    } for x in os.listdir(models_folder) if ('.ths' in x or '.torchscript' in x) and ('_nms' not in x) and ('y' in x)
  ]


  
  if GENERATE_FULL:
    log.P("Generating for:\n   {}".format("\n   ".join([m['pts'] for m in models_for_nms_prep])))
    for m in models_for_nms_prep:
      for topk in TOP_K_VALUES:
        fn_path = m['pts']
        model_name = os.path.splitext(os.path.split(fn_path)[1])[0]
        is_y5 = '_y5' in model_name
        # if '_y5s' not in model_name:
        #   continue
        if is_y5:
          model = Y5(fn_path, dev=dev, topk=topk)
        else:
          model = Y8(fn_path, dev=dev, topk=topk)

        model.eval()
        # fn_model_name = model_name[model_name.find('_y')+1:].split('_')[0]
        fn_model_name = model_name[model_name.find('y'):].split('.ths')[0]
        imgsz = model.config.get('imgsz')
        if imgsz is None:
          imgsz = model.config.get('shape')
        config = {
          **model.config,
          'input_shape': (*imgsz, 3),
          'python': sys.version.split()[0],
          'torch': th.__version__,
          'torchvision': tv.__version__,
          'device': DEVICE,
          'optimize': False,
          'n_candidates': top_candidates,
          'date': model_date,
          'model': fn_model_name,
          'names': CLASSNAMES if TRANSFER_CLASSNAMES else model.config['names'],
        }
        log.P("  Model {}: {},{}".format(model_name, fn_path, list(model.config.keys())), color='m')

        h, w = imgsz[-2:]
        log.P("  Resizing from {} to {}".format([x.shape for x in images],(h,w)))
        results = th_resize_with_pad(
          img=images,
          h=h,
          w=w,
          device=dev,
          normalize=True,
          return_original=False
        )
        if len(results) < 3:
          prep_inputs, lst_original_shapes = results
        else:
          prep_inputs, lst_original_shapes, lst_original_images = results


        log.P("  Scripting...")
        model.to(dev)
        with th.no_grad():
          traced_model = th.jit.trace(model, prep_inputs[:1], strict=False)

          log.P("  Forwarding using traced...")
          output = traced_model(prep_inputs)
        # endwith no_grad

        config['includes_nms'] = True
        config['includes_topk'] = topk
        extra_files = {'config.txt' : json.dumps(config)}
        model_name_suffix = f'nms_top{top_candidates}' if topk else 'nms'
        fn = os.path.join(models_folder, f'{model_date}_{fn_model_name}_{model_name_suffix}.ths')
        log.P(f"  Saving '{fn}'...")
        traced_model.save(fn, _extra_files=extra_files)

        extra_files['config.txt'] = ''
        loaded_model = th.jit.load(fn, _extra_files=extra_files)
        config = json.loads(extra_files['config.txt' ].decode('utf-8'))
        log.P("  Loaded config with {}".format(list(config.keys())))
        prep_inputs_test = th.cat([prep_inputs[:1], prep_inputs[:1]])
        log.P("  Running forward...")
        res, n_det = loaded_model(prep_inputs_test)
        log.P("  Done running forward. Ouput:/n{}".format(res.shape))
        generated_models.append({
          'pts': fn,
          'model': m['model'],
        })
        del loaded_model
        del traced_model
        gc.collect()
        th.cuda.empty_cache()
      # endfor include topk or not
    # endfor m in models
  # endif GENERATE_FULL

  dev = th.device('cuda')  
  if TEST:
    models_for_test = [
      {
        'pts': os.path.join(models_folder, x),
        'model': (
          x[x.find('lpd'):] if 'lpd' in x else x[x.find('_y') + 1:].split('_')[0]
        ).split('.ths')[0].split('.torchscript')[0].split('_nms')[0]
      } for x in os.listdir(models_folder) if ('.ths' in x or '.torchscript' in x)
    ]
    # now lets test WITH nms from libaries
    models_for_test = [
      *models_for_nms_prep,
      *generated_models
    ]
   
    painter = DrawUtils(log=log)

    log.P(f"Testing {len(models_for_test)} models{' with FP16' if TEST_FP16 else ''} with {N_TESTS} runs (+{20} for warmup) each on {DEVICE}")
    log.P("Testing on :\n   {}".format("\n   ".join([m['pts'] for m in models_for_test])))
    dct_results = {}
    for m in models_for_test:
      for use_fp16 in FP_VALUES:
        fn_path = m['pts']
        extra_files = {'config.txt' : ''}
        model = th.jit.load(
          f=fn_path,
          map_location=dev,
          _extra_files=extra_files,
        )
        if use_fp16:
          model.half()
        # endif use_fp16
        log.P("Done loading model on device {}".format(dev))
        config = json.loads(extra_files['config.txt' ].decode('utf-8'))
        imgsz = config.get('imgsz', config.get('shape'))
        includes_nms = config.get('includes_nms')
        includes_topk = config.get('includes_topk')
        model_name = m['model']
        model_result_name = (model_name + '_fp16') if use_fp16 else model_name
        if model_result_name not in dct_results:
          dct_results[model_result_name] = {}
        is_y5 = 'y5' in model_name
        if includes_topk:
          model_name = model_result_name + '_topk'
        else:
          model_name = model_result_name + ('_inclnms' if includes_nms else '')
        log.P("Model {}: {}, {}:".format(model_name, imgsz, fn_path,), color='m')
        maxl = max([len(k) for k in config])
        for k,v in config.items():
          if not (isinstance(v, dict) and len(v) > 5):
            log.P("  {}{}".format(k + ':' + " " * (maxl - len(k) + 1), v), color='m')

        h, w = imgsz[-2:]
        log.P("  Resizing from {} to {}".format([x.shape for x in images],(h,w)))
        class_names = config['names']
        results = th_resize_with_pad(
          img=images + images,
          h=h,
          w=w,
          device=dev,
          normalize=True,
          return_original=False,
          half=use_fp16
        )
        if len(results) < 3:
          prep_inputs, lst_original_shapes = results
        else:
          prep_inputs, lst_original_shapes, lst_original_images = results

        # warmup
        log.P("  Warming up...")
        for _ in range(20):
          print('.', flush=True, end='')
          pred_nms_cpu = predict(model, prep_inputs, model_name, config, log=log, timing=False)
        print('')

        # timing
        log.P("  Predicting...")
        for _ in range(N_TESTS):
          print('.', flush=True, end='')
          pred_nms_cpu = predict(model, prep_inputs, model_name, config, log=log, timing=True)
        print('')

        log.P("  Last preds:\n{}".format(pred_nms_cpu))

        mkey = 'includes_topk' if includes_topk else 'includes_nms' if includes_nms else 'normal'

        dct_results[model_result_name][mkey] = {
            'res'  : pred_nms_cpu,
            'time' : log.get_timer_mean(model_name),
            'name' : model_name
        }

        if SHOW:
          log.P("  Showing...")
          for i in range(len(images)):
            # now we have each individual image and we generate all objects
            # what we need to do is to match `second_preds` to image id & then
            # match second clf with each box
            img_bgr = origs[i].copy()
            np_pred_nms_cpu = pred_nms_cpu[i]
            original_shape = lst_original_shapes[i]
            np_pred_nms_cpu[:, :4] = scale_coords(
              img1_shape=(h, w),
              coords=np_pred_nms_cpu[:, :4],
              img0_shape=original_shape,
            ).round()
            lst_inf = []
            for det in np_pred_nms_cpu:
              det = [float(x) for x in det]
              # order is [left, top, right, bottom, proba, class] => [L, T, R, B, P, C, RP1, RC1, RP2, RC2, RP3, RC3]
              L, T, R, B, P, C = det[:6]  # order is [left, top, right, bottom, proba, class]
              label = class_names[str(int(C))] if SHOW_LABELS else ''
              img_bgr = painter.draw_detection_box(image=img_bgr, top=int(T), left=int(L), bottom=int(B), right=int(R), label=label, prc=P)
            painter.show(fn_path, img_bgr, orig=(0, 0))
          #endfor plot images
        #endif show
      # endfor fp16
    # endfor each model
  # endif test models
  log.show_timers()
  for mn in dct_results:
    if 'normal' not in dct_results[mn] or 'includes_nms' not in dct_results[mn]:
      continue
    a1 = dct_results[mn]['normal']['res']
    t1 = dct_results[mn]['normal']['time']
    a2 = dct_results[mn]['includes_nms']['res']
    t2 = dct_results[mn]['includes_nms']['time']
    # a3 = dct_results[mn]['includes_topk']['res']
    # t3 = dct_results[mn]['includes_topk']['time']
    # a3 = [x[:, :6] for x in a3]
    
    ok1 = all([np.allclose(a1[i], a2[i]) for i in range(len(a1))])
    # ok2 = all([np.allclose(a1[i], a3[i]) for i in range(len(a1))])

    gain1 = t1-t2
    rel_gain1 = gain1 / t1
    # gain2 = t1-t3
    log.P(f'Model with NMS {mn} {t1} => {t2}[gain {gain1:.5f}s({rel_gain1:.2f}%)], equal: {ok1}', color='r' if not ok1 else 'g')
    # log.P("Model with NMS {} gain {:.5f}s, equal: {}".format(mn, gain1, ok1), color='r' if not ok1 else 'g')
    # log.P("Model with TopK {} gain {:.5f}s, equal: {}".format(mn, gain2, ok2), color='r' if not ok2 else 'g')
      
    