from ultralytics import YOLO
# from naeural_core.xperimental.torchscript.base_torch_scripter import BaseTorchScripter
# from naeural_core import Logger
from naeural_core.local_libraries.nn.th.utils import th_resize_with_pad

import cv2 as cv
import yaml
import os


from ultralytics.utils.benchmarks import benchmark



def size_to_str(size):
  if isinstance(size, int):
    return f'{size}x{size}'
  else:
    return f'{size[0]}x{size[1]}'


model_paths = [
  "C:/repos/edge-node/core/xperimental/lpd/runs/detect/train11/weights/best.pt",
  "C:/repos/edge-node/core/xperimental/lpd/runs/detect/train12/weights/best.pt",
  "C:/repos/edge-node/core/xperimental/lpd/runs/detect/train13/weights/best.pt"
]


def get_model_and_args(model_path):
  model = YOLO("yolov8n.yaml").load(model_path)
  args_path = os.path.join(os.path.split(os.path.split(model_path)[0])[0], 'args.yaml')
  with open(args_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
  imgsz = data_yaml['imgsz']
  ds_name = os.path.splitext(data_yaml['data'])[0]
  kwargs = {
    'imgsz': imgsz,
    'data': ds_name,
    'device': 'cuda:0'
  }

  return model, kwargs


def get_model_name(kwargs):
  return f'y8n_{kwargs.get("data", "unk")}_{size_to_str(kwargs["imgsz"])}.ths'


def compare_models(m1, m2):
  m1_state, m2_state = m1.state_dict(), m2.state_dict()
  n_found, n_diff = 0, 0

  for key, value in m1_state.items():
    if key in m2_state.keys():
      if not (m2_state[key] == value).all():
        # print(f'  {key} is different')
        n_diff += 1
    else:
      # print(f'  {key} not found in {model_paths[j]}')
      n_found += 1
    # endif key in m2_state.keys()
  # endfor key, value in m1_state.items()
  for key, value in m2_state.items():
    if key in m1_state.keys():
      if not (m1_state[key] == value).all():
        # print(f'  {key} is different')
        n_diff += 1
    else:
      # print(f'  {key} not found in {model_paths[i]}')
      n_found += 1
    # endif key in m1_state.keys()
  # endfor key, value in m2_state.items()
  print(f'  {n_found} keys not found, {n_diff} keys are different')
  print(f'{"not " if n_diff + n_found > 0 else ""}matching')
  return (n_diff + n_found) == 0


def check_coreml():
  from ultralytics import YOLO

  model = YOLO('yolov8n.yaml').load('yolov8n.pt')
  model.export(format='coreml', imgsz=640, half=False, verbose=True)


if __name__ == '__main__':
  CHECK_COREML = False
  if CHECK_COREML:
    check_coreml()
    exit(-1)
  BENCHMARK = False
  if BENCHMARK:
    x = benchmark(
      model='yolov8n.pt',
      imgsz=320,
      data='coco128.yaml',
      device='cpu',
      half=False,
    )
    exit(-1)
  TRACE_GENERAL_DETECTOR = True
  if TRACE_GENERAL_DETECTOR:
    # imgsz = [448, 640]
    imgsz = 640

    MODELS = {
      'y8n': {
        'yaml': 'yolov8n.yaml',
        'weights': 'yolov8n.pt',
      },
      'y8s': {
        'yaml': 'yolov8s.yaml',
        'weights': 'yolov8s.pt',
      },
      'y8l': {
        'yaml': 'yolov8l.yaml',
        'weights': 'yolov8l.pt',
      },
    }

    for model_name in ['y8n', 'y8s', 'y8l']:
      model = YOLO(MODELS[model_name]['yaml']).load(MODELS[model_name]['weights'])
      device = 'cuda:0'
      model = model.to(device)
      format = "torchscript"

      for use_int8 in [
        False,
        # True
      ]:
        print(f'Exporting {model_name} with use_int8={use_int8}')
        export_kwargs = {
          'format': format,
          'imgsz': imgsz,
          'int8': use_int8,
        }
        pt_path = f'{model_name}_{"int8" if use_int8 else ""}.torchscript'
        setattr(model.model, 'pt_path', pt_path)
        model.export(
          **export_kwargs
        )
      # endfor use_int8 in [True, False]
    # endfor model_name in ['y8n', 'y8s', 'y8l']
    exit(-1)
  # endif True
  TRAIN = False
  TEST = False
  COMPARE_MODELS = False
  TRACE_MODELS = False
  device = 'cuda:0'
  if TRACE_MODELS:
    for model_path in model_paths:
      model, kwargs = get_model_and_args(model_path)
      model = model.to(device)
      model.export(format="torchscript", **kwargs)
      model_name = get_model_name(kwargs)
      os.rename('yolov8n.torchscript', model_name)
    # endfor model_path in model_paths
    exit(-1)
  # endif TRACE_MODELS
  if COMPARE_MODELS:
    models = [
      YOLO("yolov8n.yaml").load(model_path)
      for model_path in model_paths
    ]
    for i in range(len(models)):
      for j in range(i + 1, len(models)):
        m1, m2 = models[i], models[j]
        print(f'Comparing {model_paths[i]} with {model_paths[j]}')
        m1_state, m2_state = m1.state_dict(), m2.state_dict()
        n_found, n_diff = 0, 0

        for key, value in m1_state.items():
          if key in m2_state.keys():
            if not (m2_state[key] == value).all():
              # print(f'  {key} is different')
              n_diff += 1
          else:
            # print(f'  {key} not found in {model_paths[j]}')
            n_found += 1
          # endif key in m2_state.keys()
        # endfor key, value in m1_state.items()
        for key, value in m2_state.items():
          if key in m1_state.keys():
            if not (m1_state[key] == value).all():
              # print(f'  {key} is different')
              n_diff += 1
          else:
            # print(f'  {key} not found in {model_paths[i]}')
            n_found += 1
          # endif key in m1_state.keys()
        # endfor key, value in m2_state.items()
        print(f'  {n_found} keys not found, {n_diff} keys are different')
        print(f'{"not " if n_diff + n_found > 0 else ""}matching')
      # endfor j in range(i + 1, len(models))
    # endfor i in range(len(models))
    exit(-1)
  # endif COMPARE_MODELS
  if TRAIN:
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    data_yaml = ""
    imgsz = 640
    # model.train(data='lpd.yaml', epochs=20, imgsz=640, device='cuda:0')
    # model.train(data='lpd_no_diff.yaml', epochs=20, imgsz=640, device='cuda:0')
    model.train(data='lpd_no_diff.yaml', epochs=20, imgsz=[448, 640], device='cuda:0')
    model_path = model.trainer.best
  else:
    model_path = "C:/repos/edge-node/core/xperimental/lpd/runs/detect/train15/weights/best.pt"
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    data_yaml = "lpd_no_diff"
    imgsz = [448, 640]

    # model_path = "C:/repos/edge-node/core/xperimental/lpd/runs/detect/train12/weights/best.pt"
    # model = YOLO("yolov8n.yaml")
    # data_yaml = "lpd_no_diff"
    # imgsz = 640

    # model_path = "C:/repos/edge-node/core/xperimental/lpd/runs/detect/train11/weights/best.pt"
    # model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    # data_yaml = "lpd"
    # imgsz = 640
  # endif TRAIN

  model = model.load(model_path).to(device)
  if TEST:
    img_paths = [
      "Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/25mai/bd225a82-54e7-4bfa-8776-636a519cf907_O_0.jpg",
      "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo/images/2.Dev/Cars_with_LP/Complexity_2/4.GTS/dataset_builder_1__DATASET_BUILDER_01__DATASET_BUILDER_603.jpg"
    ]

    origs = [
      # cv2.imread(os.path.join(folder, im_name))
      cv.imread(im_name)
      for im_name in img_paths
    ]
    # images = [
    #   np.ascontiguousarray(img[:, :, ::-1])
    #   for img in origs
    # ]
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])
    results = th_resize_with_pad(
      img=origs,
      h=h,
      w=w,
      device=device,
      normalize=True,
      return_original=False
    )

    if len(results) < 3:
      prep_inputs, lst_original_shapes = results
    else:
      prep_inputs, lst_original_shapes, lst_original_images = results

    results = model(prep_inputs)

  # endif TEST

  model.export(format="torchscript", imgsz=imgsz)
  # model_name = f'y8n_{data_yaml}_{size_to_str(imgsz)}.ths'
  # os.rename('yolov8n.torchscript', model_name)

  # input_shape = (*imgsz, 3) if not isinstance(imgsz, int) else (imgsz, imgsz, 3)
  #
  # log = Logger(
  #   lib_name='EE',
  #   base_folder='./core/xperimental/lpd',
  #   app_folder='_local_cache',
  #   max_lines=3000,
  #   TF_KERAS=False
  # )
  # tracer = BaseTorchScripter(
  #   log=log,
  #   model=model,
  #   model_name=model_name,
  #   input_shape=input_shape,
  #   use_fp16=False
  # )
  #
  # img_paths = [
  #   "C:/Users/Workstation/Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/25mai/6cf0d612-617a-4411-97ae-40910733f5ff_O_0.jpg",
  #   "C:/Users/Workstation/Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/25mai/dbd5e154-47fd-404b-8b21-871937d5c7db_O_0.jpg",
  #   "C:/Users/Workstation/Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/25mai/c3fc83c9-005f-45c3-bfb5-837637ec47e4_O_0.jpg",
  #   "C:/Users/Workstation/Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/25mai/bd225a82-54e7-4bfa-8776-636a519cf907_O_0.jpg"
  # ]
  #
  # inputs = [
  #   cv.imread(img_path)
  #   for img_path in img_paths
  # ]
  #
  # ts_path = tracer.generate(
  #   inputs=inputs, batch_size=4,
  #   device='cuda:0', to_test=True,
  #   nr_warmups=20, nr_tests=100
  # )
  # log.P(f'The saved model is at: {ts_path}')

