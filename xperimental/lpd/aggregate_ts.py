import os
import yaml
import shutil


def decode_key(key, value):
  if key == 'data':
    return os.path.splitext(value)[0]
  elif key == 'imgsz':
    return max(value)
  elif key == 'model':
    return value[0] + os.path.splitext(value)[0][-2:]
  return value


if __name__ == '__main__':
  start_iter = 62
  end_iter = 68

  relevant_keys = [
    'model',
    'data',
    'imgsz',
    'epochs'
  ]
  root_dir = os.path.dirname(os.path.abspath(__file__))
  dest_dir = os.path.join(root_dir, 'traces')
  os.makedirs(dest_dir, exist_ok=True)
  for i in range(start_iter, end_iter):
    curr_dir = os.path.join(root_dir, 'runs', 'detect', f'train{i if i > 1 else ""}')
    args_path = os.path.join(curr_dir, 'args.yaml')
    ts_path = os.path.join(curr_dir, 'weights', 'best.torchscript')

    yaml_data = None
    with open(args_path, 'r') as f:
      yaml_data = yaml.safe_load(f)
    fn = '_'.join([f'{decode_key(k, yaml_data[k])}' for k in relevant_keys])
    dest_path = os.path.join(dest_dir, f'{fn}.ths')
    shutil.copyfile(ts_path, dest_path)
  # endfor
