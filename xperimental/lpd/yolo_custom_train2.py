# TODO Bleo: WIP: save metrics for LPD training
import cv2
import os
import torch as th
from ultralytics import YOLO


test_images = {
  'train': [
    "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo-no_diff/images/1.Train/Cars_with_LP/Complexity_2/9.OTHER/000014_car 0.70_0.jpg",
    "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo-no_diff/images/1.Train/Cars_with_LP/Complexity_2/9.OTHER/000122_car 0.94_2.jpg"
  ],
  'dev': [
    "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo/images/2.Dev/Cars_with_LP/Complexity_2/4.GTS/dataset_builder_1__DATASET_BUILDER_01__DATASET_BUILDER_603.jpg",
    "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo-no_diff/images/2.Dev/Cars_with_LP/Complexity_3/4.GTS/dataset_builder_2__DATASET_BUILDER_01__DATASET_BUILDER_28.jpg",
    "_local_cache/_data/Dataset_LPLR_v5.2.2_for_yolo-no_diff/images/2.Dev/Cars_without_LP/Obs_TEST/1.GOC/CV 1 LPR 1 (2021.08.04 13-00-29.853)_truck 0.75_2.jpg"
  ],
  'test': [
    "C:/Users/bleot/Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/25mai/bd225a82-54e7-4bfa-8776-636a519cf907_O_0.jpg",
    "C:/Users/bleot/Dropbox/DATA/_vapor_data/__tests/LPLR/test_lpr/TEST_masini_grele/dataset_builder_3__DATASET_BUILDER_01__DATASET_BUILDER_5470.jpg"
  ]
}


def test(model, img_ds, prefix='', show=False):
  for ds_name, ds in img_ds.items():
    test_ds(model=model, img_paths=ds, prefix=f'{prefix}_{ds_name}', show=show)
  # endfor
  return


def test_ds(model, img_paths, prefix='', show=False):
  for img_path in img_paths:
    print(f'img_path: {img_path}')
    with th.no_grad():
      results = model(img_path)[0]
    np_results = results.boxes.data.cpu().numpy()
    if show:
      img = cv2.imread(img_path)
    # endif show
    objects = []
    for np_result in np_results:
      l, t, r, b = np_result[:4]
      conf, cls = np_result[4:]
      objects.append((conf, cls))
      if show:
        img = cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
      # endif show
    # endfor
    print(f'Found: {objects} @{prefix}_{os.path.basename(img_path)}')
    if show:
      cv2.imshow(f'{prefix}_{os.path.basename(img_path)}', img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    # endif show
  # endfor
  return


grid = {
  "data": [
    # "lpd",
    # "lpd_no_diff",
    # "v1_only",
    # "v1+lpl_no_diff",
    # "lpd_v2",
    # 'lpdr_v5',
    # 'lpd_v4.1',
    'lpd_v5',
  ],
  "imgsz": [
    [448, 640],
    [640, 896]
  ],
  "epochs": [
    1,
    10,
    20,
    40
  ],
}


"""
20-23: [ld_no_diff]x[[448, 640],[640, 896]]x[10, 20]
24-25: [ld_no_diff]x[[448, 640],[640, 896]]x[40]

"""


if __name__ == '__main__':
  total_iterations = len(grid['epochs']) * len(grid['imgsz']) * len(grid['data'])
  it = 0
  for epochs in grid['epochs']:
    for imgsz in grid['imgsz']:
      for data_yaml in grid['data']:
        it += 1
        print(f'Iteration: {it}/{total_iterations}: epochs: {epochs}, imgsz: {imgsz}, data_yaml: {data_yaml}')
        # 1. Initialize model
        model = YOLO("yolov8n.yaml").load("yolov8n.pt")
        # data_yaml = "lpd_no_diff"
        # imgsz = [448, 640]
        device = 'cuda:0'

        # # 2. Test before train
        # test(model, test_images, prefix='pre')

        # 3. Train
        model.train(data=f'{data_yaml}.yaml', epochs=epochs, imgsz=imgsz, device=device)

        # # 4.1. Test after train
        # test(model, test_images, prefix='post')
        # 4.2. Test after train with best loaded
        model = model.load(model.trainer.best).to(device)
        test(model, test_images, prefix='best')

        # 5. Save model
        model.export(format="torchscript", imgsz=imgsz, data=data_yaml, epochs=epochs, device=device)

        # 6. Test after save
        # TODO: test this correctly after save
        # model = YOLO("yolov8n.yaml").load("runs/detect/train15/weights/best.torchscript")
        # test(model, test_images, prefix='ts')
      # endfor data
    # endfor imgsz
  # endfor epochs
