import xml.etree.ElementTree as ET
import os
import json
import cv2 as cv
import shutil
from tqdm import tqdm
from label_testing import get_dets


IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'bmp')


def valid_type(val):
  if not isinstance(val, str):
    return True
  if val == '_':
    return True
  is_valid = True
  for c in val:
    if c.islower():
      is_valid = False
      break
  return is_valid


def clear_dets(dets):
  return [det for det in dets if valid_type(det[0])]


def save_label(label_save_path, file_path, real_dets, additional, ext):
  if ext == 'txt':
    detection_list = [
      [0, (l + r) / 2 / img_w, (t + b) / 2 / img_h, (r - l) / img_w, (b - t) / img_h]
      for dtype, l, t, r, b in real_dets
    ]
    label_txt = '\n'.join([
      ' '.join([str(i) for i in detection])
      for detection in detection_list
    ])
    with open(os.path.join(label_save_path, os.path.basename(file_path) + '.txt'), 'w') as f:
      f.write(label_txt)
    # endwith open
  elif ext == 'xml':
    tree = additional
    root = tree.getroot()
    to_delete = root.findall('object')
    for elem in to_delete:
      root.remove(elem)
    for dtype, l, t, r, b in real_dets:
      obj = ET.SubElement(root, 'object')
      ET.SubElement(obj, 'name').text = dtype
      bbox = ET.SubElement(obj, 'bndbox')
      ET.SubElement(bbox, 'xmin').text = str(l)
      ET.SubElement(bbox, 'ymin').text = str(t)
      ET.SubElement(bbox, 'xmax').text = str(r)
      ET.SubElement(bbox, 'ymax').text = str(b)
    # endfor dtype, l, t, r, b in real_dets
    tree.write(os.path.join(label_save_path, os.path.basename(file_path) + '.xml'))
  # endif ext

  return


def convert_datapoint2(
    img_path, base_dir, dest_dir, no_subfolders=False, pre_subdirs="",
    convert=True, input_label_ext='xml', output_label_ext='txt'
):
  file_path = os.path.splitext(img_path)[0]
  xml_path = file_path + f'.{input_label_ext}'
  if input_label_ext == 'txt':
    xml_path = xml_path.replace('images', 'labels')
  img = cv.imread(img_path)
  img_h, img_w, _ = img.shape
  subdirs = os.path.split(file_path[len(base_dir) + 1:])[:-1]
  # maybe this is not necessary
  # if no_subfolders:
  #   subdir_prefix = '___'.join(['__'.join(subdir.split(os.path.sep)) for subdir in subdirs])

  if output_label_ext == 'txt':
    img_save_path = os.path.join(dest_dir, 'images', pre_subdirs, *subdirs)
    label_save_path = os.path.join(dest_dir, 'labels', pre_subdirs, *subdirs)
  elif output_label_ext == 'xml':
    img_save_path = os.path.join(dest_dir, pre_subdirs, *subdirs)
    label_save_path = os.path.join(dest_dir, pre_subdirs, *subdirs)
  os.makedirs(img_save_path, exist_ok=True)
  stats = {}
  n_dets = 0
  if os.path.exists(xml_path):
    dets, additional = get_dets(path=xml_path, img_w=img_w, img_h=img_h, ext=input_label_ext)
    for dtype, l, t, r, b in dets:
      type_key = dtype if isinstance(dtype, str) else int(dtype)
      stats[type_key] = stats.get(type_key, 0) + 1
      if dtype == 'unknown':
        print(f'Unknown detection in {file_path}')
    real_dets = clear_dets(dets)
    n_dets = len(real_dets)
    if convert:
      label_save_path = os.path.join(label_save_path, str(n_dets))
      os.makedirs(label_save_path, exist_ok=True)
      save_label(
        label_save_path, file_path, real_dets, additional, output_label_ext
      )
    # endif convert
  # endif os.path.exists(xml_path)
  if convert:
    img_save_path = os.path.join(img_save_path, str(n_dets))
    os.makedirs(img_save_path, exist_ok=True)
    shutil.copy(img_path, os.path.join(img_save_path, os.path.basename(file_path) + '.jpg'))
  # endif convert
  return stats


# TODO: remove this and rename convert_datapoint2 to
#  convert_datapoint after further testing
def convert_datapoint(img_path, base_dir, dest_dir, no_subfolders=False, pre_subdirs="", convert=True):
  file_path = os.path.splitext(img_path)[0]
  xml_path = file_path + '.xml'
  img = cv.imread(img_path)
  img_h, img_w, _ = img.shape
  subdirs = os.path.split(file_path[len(base_dir) + 1:])[:-1]
  # maybe this is not necessary
  # if no_subfolders:
  #   subdir_prefix = '___'.join(['__'.join(subdir.split(os.path.sep)) for subdir in subdirs])

  img_save_path = os.path.join(dest_dir, 'images', pre_subdirs, *subdirs)
  label_save_path = os.path.join(dest_dir, 'labels', pre_subdirs, *subdirs)
  os.makedirs(img_save_path, exist_ok=True)
  stats = {}
  n_dets = 0
  if os.path.exists(xml_path):
    dets = get_dets(path=xml_path, img_w=img_w, img_h=img_h, ext='xml')
    for dtype, l, t, r, b in dets:
      stats[dtype] = stats.get(dtype, 0) + 1
      if dtype == 'unknown':
        print(f'Unknown detection in {file_path}')
    real_dets = clear_dets(dets)
    n_dets = len(real_dets)
    if convert:
      detection_list = [
        [0, (l + r) / 2 / img_w, (t + b) / 2 / img_h, (r - l) / img_w, (b - t) / img_h]
        for dtype, l, t, r, b in real_dets
      ]

      label_txt = '\n'.join([
        ' '.join([str(i) for i in detection])
        for detection in detection_list
      ])
      label_save_path = os.path.join(label_save_path, str(n_dets))
      os.makedirs(label_save_path, exist_ok=True)
      with open(os.path.join(label_save_path, os.path.basename(file_path) + '.txt'), 'w') as f:
        f.write(label_txt)
      # endwith open
    # endif convert
  # endif os.path.exists(xml_path)
  if convert:
    img_save_path = os.path.join(img_save_path, str(n_dets))
    os.makedirs(img_save_path, exist_ok=True)
    shutil.copy(img_path, os.path.join(img_save_path, os.path.basename(file_path) + '.jpg'))
  # endif convert
  return stats


def ds_size(base_dir):
  it = 0
  for root, dirs, files in os.walk(base_dir):
    for file in files:
      if file.lower().endswith(IMG_EXTENSIONS):
        it += 1
      # endif img_path.endswith('.jpg')
    # endfor img_path in files
  # endfor root, dirs, files in os.walk(base_dir)
  return it


def convert_dataset(base_dir, dest_dir, period=1000, convert=True, **kwargs):
  it = 0
  total_stats = {}
  for root, dirs, files in os.walk(base_dir):
    for file in files:
      if file.lower().endswith(IMG_EXTENSIONS):
        it += 1
        if it % period == 0:
          print(f'Processing {it}th image')
        # endif it % period == 0
        img_path = os.path.join(root, file)
        # curr_stats = convert_datapoint(img_path, base_dir, dest_dir, convert=convert)
        curr_stats = convert_datapoint2(img_path, base_dir, dest_dir, convert=convert, **kwargs)
        for k, v in curr_stats.items():
          total_stats[k] = total_stats.get(k, 0) + v
        # endfor k, v in curr_stats.items()
      # endif img_path.endswith('.jpg')
    # endfor img_path in files
  # endfor root, dirs, files in os.walk(base_dir)

  return total_stats


def random_to_train_validate(base_dir, dest_dir, period=1000, train_ratio=0.8):
  it = 0
  images = []
  for root, dirs, files in os.walk(base_dir):
    for file in files:
      if file.lower().endswith(IMG_EXTENSIONS):
        it += 1
        if it % period == 0:
          print(f'Processing {it}th/{len(files)} image')
        # endif it % period == 0
        img_path = os.path.join(root, file)
        images.append(img_path)
      # endif img_path.endswith('.jpg')
    # endfor img_path in files
  # endfor root, dirs, files in os.walk(base_dir)
  print(f'Processed {it} images in total')

  import random
  random.shuffle(images)
  delimiter_idx = int(len(images) * train_ratio)
  train_images = images[:delimiter_idx]
  val_images = images[delimiter_idx:]
  print(f'Number of train images: {len(train_images)}')
  print(f'Number of val images: {len(val_images)}')
  print(f'Starting conversion for train images({len(train_images)})')
  for img_path in tqdm(train_images):
    convert_datapoint(img_path, base_dir, dest_dir, pre_subdirs='1.Train')
  # endfor img_path in train_images
  print(f'Starting conversion for val images({len(val_images)})')
  for img_path in tqdm(val_images):
    convert_datapoint(img_path, base_dir, dest_dir, pre_subdirs='2.Dev')
  # endfor img_path in val_images
  return


def sort_by_value(dct, decreasing=True):
  res = sorted(dct.items(), key=lambda x: x[1], reverse=decreasing)
  return {x[0]: x[1] for x in res}


def aggregate_stats(stats):
  agg_stats = {}
  for dk, dv in stats.items():
    for k, v in dv.items():
      agg_stats[k] = agg_stats.get(k, 0) + v
  return agg_stats


if __name__ == '__main__':
  if False:
    json_path = "stats.json"
    with open(json_path, 'r') as f:
      stats = json.load(f)
    # endwith open
    a = 1
    print(stats)
  if True:
    # base_dir = "C:/training/LPD_datasets/V2/LPD_V2"
    # dest_dir = "C:/training/LPD_datasets/V2/converted"
    # random_to_train_validate(base_dir, dest_dir)

    # base_dir = "C:/training/LPD_datasets/LPDR_v5"
    # dest_dir = "C:/training/LPD_datasets/LPDR_v5_converted"

    # base_dir = r"C:\training\LPD_datasets\LPD_v4.1\initial"
    # dest_dir = r"C:\training\LPD_datasets\LPD_v4.1\converted"

    base_dir = r"C:\training\LPD_datasets\LPD_v5\initial"
    dest_dir = r"C:\training\LPD_datasets\LPD_v5\converted"
    filtered_dir = r"C:\training\LPD_datasets\LPD_v5\filtered"
    convert = False
    ds_stats = {}
    task_str = f'analyzing{" and converting" if convert else ""}'
    for subdir in ['1.train', '2.dev', '3.test']:
      print(f'Started {task_str} {subdir}...')
      ds_stats[subdir] = convert_dataset(
        # os.path.join(base_dir, subdir),
        os.path.join(filtered_dir, subdir),
        os.path.join(filtered_dir, subdir),
        convert=convert,
        input_label_ext='xml',
        output_label_ext='xml'
      )
      print(f'Finished {task_str} {subdir}')
    # endfor subdir
    final_stats = aggregate_stats(ds_stats)
    sorted_stats = sort_by_value(final_stats)
    json.dump(sorted_stats, open('sorted_stats.json', 'w'))
    json.dump(ds_stats, open('stats.json', 'w'))
    a = 1
    # print('Total stats:')
    # for k, v in ds_stats.items():
    #   print(f'{k}: {v}')
    exit(-1)
  # endif True
  if False:
    base_dir = "C:/resources/sample_ds"
    dest_dir = "C:/resources/sample_ds_converted"
  else:
    base_dir = "_local_cache/_data/Dataset_LPLR_v5.2.2"
    dest_dir = "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo"
  convert_dataset(base_dir, dest_dir, period=1000)
  exit(-1)

  img_path = "C:/resources/sample_ds/1/2/abc abc/CV 1 LPR 1 (2021.08.04 08-38-54.150)_car 0.75_5.jpeg"
  convert_datapoint(img_path, base_dir, dest_dir)

  exit(-1)
  file_path = os.path.splitext(img_path)[0]
  xml_path = file_path + '.xml'
  img = cv.imread(img_path)
  img_h, img_w, _ = img.shape
  subdirs = os.path.split(file_path[len(base_dir) + 1:])[:-1]
  save_path = os.path.join(dest_dir, *subdirs)
  os.makedirs(save_path, exist_ok=True)


  if os.path.exists(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    detection_list = []
    for obj in root.findall('object'):
      bbox = obj.find('bndbox')
      t, l, b, r = (
        int(bbox.find('ymin').text),
        int(bbox.find('xmin').text),
        int(bbox.find('ymax').text),
        int(bbox.find('xmax').text),
      )
      x, y, w, h = l / img_w, t / img_h, (r - l) / img_w, (b - t) / img_h
      detection_list.append([0, x, y, w, h])
    # endfor obj

    label_txt = '\n'.join([
      ' '.join([str(i) for i in detection])
      for detection in detection_list
    ])
    with open(os.path.join(save_path, os.path.basename(file_path) + '.txt'), 'w') as f:
      f.write(label_txt)
    # endwith open

  # endif os.path.exists(xml_path)
  shutil.copy(img_path, os.path.join(save_path, os.path.basename(file_path) + '.jpg'))



