import os
import pickle
import test

import torch as th
from generate_dataset import show_one_image

save_path = "C:\\Users\\Stefan.saraev\\Downloads\\LP frontal\\Placute rotite"

if __name__ == "__main__":
  dev = th.device("cuda")
  train_path = os.path.join(save_path, "train")
  theta_pickle_file_name = "i_theta.pickle"
  print(os.listdir(train_path))
  
  if theta_pickle_file_name in os.listdir(train_path):
    with open(os.path.join(train_path, theta_pickle_file_name), 'rb') as handle:
      thetas_read = pickle.load(handle)

  assert len(thetas_read) == len(os.listdir(os.path.join(train_path, "images"))), "Difference in number of elements indexed in the pickle file and images in folder"
  print(len(thetas_read))
  print(len(os.listdir(os.path.join(train_path, "images"))))

  checking = {k:v for k, v in list(thetas_read.items())[:50]}
  
  images_path = os.path.join(train_path, "images")
  
  for img_name, theta in checking.items():
    img_path = os.path.join(images_path, img_name + ".jpg")
    np_img = test.read_image(img_path)
    th_img = test.np2th(np_img).unsqueeze(0).to(dev)
    
    theta = th.Tensor(theta).to(dev)
    
    th_transformed = test.transform(th_img, theta)
    show_one_image(th_img)
    show_one_image(th_transformed)
    
    # todo