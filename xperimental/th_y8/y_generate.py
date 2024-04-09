from ultralytics import YOLO
import torch as th
import json


if __name__ == '__main__':
  model_cfgs = {
    's': {
      'yaml': 'yolov8s.yaml',
      'weights': 'yolov8s.pt',
    },
    'l': {
      'yaml': 'yolov8l.yaml',
      'weights': 'yolov8l.pt',
    },
  }

  sizes = {
    's': [
      [576, 1024],
      [640, 1152],
      [768, 768]
    ],
    'l': [
      [640, 1152],
      [768, 768],
      [896, 896]
    ],
  }

  classnames_model_path = r'C:\repos\edge-node\_local_cache\_models\20230723_y8l_nms.ths'
  extra_files = {'config.txt': ''}
  model = th.jit.load(classnames_model_path, map_location='cpu', _extra_files=extra_files)
  CLASSNAMES = json.loads(extra_files['config.txt'])['names']

  for model_name in ['s', 'l']:
    model = YOLO(model_cfgs[model_name]['yaml']).load(model_cfgs[model_name]['weights'])
    device = 'cuda:0'
    model = model.to(device)
    format = "torchscript"

    for imgsz in sizes[model_name]:
      print(f'Exporting y8{model_name} with imgsz={imgsz}')
      export_kwargs = {
        'format': format,
        'imgsz': imgsz,
      }
      pt_path = f'y8{model_name}_{imgsz[0]}x{imgsz[1]}.torchscript'
      setattr(model.model, 'pt_path', pt_path)
      setattr(model.model, 'names', CLASSNAMES)
      model.export(**export_kwargs)
    # endfor imgsz
  # endfor model_name

