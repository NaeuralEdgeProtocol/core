import os
import random
from typing import Any

import cv2
import numpy as np
import torch as th
import torchvision.transforms as T

from naeural_core.local_libraries.nn.th.utils import th_resize_with_pad
from plugins.serving.pipelines.data.lpl import read_labeling_xml


def add_theta_to_dict(th_theta, dict, label, iteration):
  key = label + "__{}".format(iteration)
  dict[key] = th_theta.cpu().numpy()
  return

def add_image_to_dict(th_image, dict, label, iteration):
  np_image = self.th2np(th_image.squeeze(0))
  key = label + "__{}".format(iteration)
  dict[key] = np_image
  return

def _save_np_image(np_image, save_path, label, extension):
  from PIL import Image
  img_path = os.path.join(save_path, label + ".{}".format(extension))
  np_image = Image.fromarray(np_image)
  np_image.save(img_path)

def merge_dicts(images, thetas, i_thetas):
  res = {}
  keys = images.keys()
  for key in keys:
    res[key] = (images[key], thetas[key], i_thetas[key],)
  return res   

def _save_data(list, path, folder, extension):
  theta_file_name = "theta.pickle"
  i_theta_file_name = "i_theta.pickle"
  import pickle

  
  folder_path = os.path.join(path, folder)
  images_folder_path = os.path.join(folder_path, "images")
  thetas = {}
  i_thetas = {}
  print(len(list))
  for label, (img, theta, i_theta) in list:
    thetas[label] = theta
    i_thetas[label] = i_theta
    _save_np_image(img, images_folder_path, label, extension)
  
  theta_path = os.path.join(folder_path, theta_file_name)
  with open(theta_path, 'wb') as handle:
    pickle.dump(thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  i_theta_path = os.path.join(folder_path, i_theta_file_name)
  with open(i_theta_path, 'wb') as handle:
    pickle.dump(i_thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_data(dict, path, train_dev_ratio, extension="jpg"):
  data = sorted(list(dict.items()))
  N = len(data)
  no_dev = int(np.ceil(N / (train_dev_ratio + 1)))
  
  dev = data[:no_dev]
  train = data[no_dev:]
  
  _save_data(train, path, "train", extension)
  _save_data(dev, path, "dev", extension)
  return

class DS_Builder():
  def __init__(self, dataset_path, *, dataset='theta', input_dim, batch_size=1, output_dim=None, target_value='original', dev=th.device("cuda")) -> None:
    """Class that generates input for training process

    Args:
        dataset_path (str): path to dataset
        dataset (str, optional): Dataset to be returned. Can be 'theta' or 'encoder'. Defaults to 'theta'.
        input_dim (tuple, optional): Shape of input image.
        output_dim (tuple, optional): Shape of output image, if dataset is 'encoder'. Defaults to 'None'
        target_value (str, optional): The image to be selected as output for 'encoder' dataset. Can be 'original' or 'processed'. Defaults to 'original'
        dev (th.device, optional): Torch device. Defaults to 'th.device("cuda")'
    """
    lst_np_images, _, lst_offsets = self._read_dataset_with_offsets(dataset_path)
    
    self.dataset = dataset
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.dev = dev
    self.target_value = target_value
    self.batch_size = batch_size
    
    self.lst_th_images_orig_lp_offset = []
    
    self.index = 0
    for np_image, offset in zip(lst_np_images, lst_offsets):
      th_image_orig = self._np2th(np_image).unsqueeze(0).to(self.dev)
      th_img_shape = offset[-2:]
      offset = offset[:2]
      th_image_lp = self._get_lp_from_image(th_image_orig, offset, th_img_shape)

      self.lst_th_images_orig_lp_offset.append((th_image_orig, th_image_lp, offset))
      
    self.index = 0
    
  def get_generator(self):
    while True:
      lst_elements = [self() for _ in range(self.batch_size)]
      fst = [e[0] for e in lst_elements]
      snd = [e[1] for e in lst_elements]
      
      yield th.cat(fst), th.cat(snd)
        
  def __call__(self) -> Any:
    # Shuffle the data if we reached the end
    if self.index == len(self.lst_th_images_orig_lp_offset):
      random.shuffle(self.lst_th_images_orig_lp_offset)
      self.index = 0
    # Extract random image from dataset
    th_image_orig, th_image_lp, offset = self.lst_th_images_orig_lp_offset[self.index]
    self.index += 1
    
    # Compute transform parameters
    transform_kwargs = self._generate_random_kwargs(
      translate_range = [(0,0)],
      uniform_scale_range = [(1, 1)],
      shear_range = [(-30, -12), (12, 30), (-30, 30)],
    )
    
    # Cut image to extract a padded one
    th_image_lp_padded = self._cut_pad_th_image(th_image_orig, th_image_lp, offset, transform_kwargs)   
    
    # Add image shape to transform parameters
    transform_kwargs['image_shape'] = th_image_lp_padded.shape[2:]
    
    # Compute theta
    th_theta = self._compute_matrix(**transform_kwargs).unsqueeze(0)
    
    # Apply Transform
    th_image_transformed = self._transform(th_image_lp_padded, th_theta)
    
    # Apply resize with pad
    th_image_transformed_padded, orig_size = th_resize_with_pad(th_image_transformed, self.input_dim[0], self.input_dim[1], device=dev)
    
    # Compute Inverse Theta
    th_inverse_theta = self._compute_inverse(th_theta, (th_image_transformed_padded.shape[2:], orig_size))
  
    if self.dataset == 'theta':
      return th_image_transformed_padded, th_inverse_theta
  
    if self.target_value == 'original':
      th_final = T.Resize(size=self.output_dim)(th_image_lp)
    
    if self.target_value == 'processed':
      th_i_transformed = self._transform(th_image_transformed_padded, th_inverse_theta)
      th_bands = self._cut_bands(th_i_transformed, (th_image_transformed_padded.shape[2:], orig_size))
      
      th_final = T.Resize(size=self.output_dim)(th_bands)
    
    if self.dataset == 'encoder':
      return th_image_transformed_padded, th_final
    
    return None 

  def _generate_random_kwargs(self, *, translate_range, rotation_degrees_range=None, uniform_scale_range, shear_range):
    """Generate kwargs for the self.compute_matrix() method

    Args:
        translate_range (list[tuple[float]]): min, max values for translate in Ox, Oy. Values in pixels
        uniform_scale_range (list[tuple[float]]): min, max values for scaling
        shear_range (list[tuple[float]]): min, max values for shearing. Values in degrees for both Ox and Oy
        rotation_degrees_range (list[tuple[float]], optional): min, max for rotation angle. Values in degrees. Defaults self, to None. If None, compute rotation based on shear

    Returns:
        _type_: _description_
    """
    tx = self._generate_random_value(translate_range)
    ty = self._generate_random_value(translate_range)
    u  = self._generate_random_value(uniform_scale_range)
    sx = self._generate_random_value(shear_range)
    sy = sx #0 #-sx / 2 # self._generate_random_value(shear_range)
    if rotation_degrees_range is not None:
      r = self._generate_random_value(rotation_degrees_range)
    else:
      r = -sx
    
    return {
      't': (tx, ty),
      'u': u, 
      's': (sx, sy),
      'r': r, 
    }

  def _read_xml(self, dataset_path, file_name):
    return read_labeling_xml(os.path.join(dataset_path, file_name))

  def _proces_xml(self, data):
    return data[0][0], data[0][1], data[0][2] - data[0][0], data[0][3] - data[0][1]

  def _read_labels(self, dataset_path, image_extension='.jpg', xml_extension=None):
    file_names = [file_name for file_name in os.listdir(dataset_path) if os.path.splitext(file_name)[1] != xml_extension]
    labels = [os.path.splitext(file_name)[0] for file_name in file_names]
    
    if xml_extension is None:
      return_value = file_names, labels
    else:
      metadata = [self._proces_xml(self._read_xml(dataset_path, label+xml_extension)) for label in labels]
      return_value = file_names, labels, metadata
    
    return return_value

  def _read_dataset_as_list(self, dataset_path):
    file_names, labels = read_labels(dataset_path)
    lst_np_images = [self._read_image(os.path.join(dataset_path, file_name)) for file_name in file_names]
    return lst_np_images, labels

  def _read_dataset_with_offsets(self, dataset_path):
    file_names, labels, metadata = self._read_labels(dataset_path, '.jpg', '.xml')
    lst_np_images = [self._read_image(os.path.join(dataset_path, file_name)) for file_name in file_names]
    return lst_np_images, labels, metadata

  def show_images(self, *th_images):
    # images = [self.th2np(th_image) for th_image in th_images]
    no_lines = 3
    width = int(max([max(th_img.shape) for th_img in th_images]) * 1.2)
    mat = th.ones((3, no_lines * width, no_lines * width))
    
    for i in range(no_lines):
      for j in range(no_lines):
        index = i * no_lines + j
        if index >= len(th_images):
          break
        th_image = th_images[index]
        h, w = th_image.shape[2:]
        mat[:, i*width : i*width+h, j*width : j * width + w] = th_image[0]
      if index >= len(th_images):
        break
    mat = self._th2np(mat)
    cv2.imshow("original", mat[:,:,::-1])
    cv2.waitKey(0)

  def _show_one_image(self, th_images, index=0):
    sample_image = self._th2np(th_images[index])
    cv2.imshow("original", sample_image)
    cv2.waitKey(0)

  def _simple_pad_th_image(self, th_img):
    u, d, l, r = 250,250,250,250
    
    th_image_new = th.zeros((th_img.shape[0], th_img.shape[1], th_img.shape[2] + u + d, th_img.shape[3] + l + r)).to(device="cuda")

    th_image_new[:, :, u : th_img.shape[2] + u, l : th_img.shape[3] + l] = th_img
    return th_image_new, (u, l)

  def _pad_th_image(self, th_img, transform_kwargs):
    translate = transform_kwargs['t']
    scale     = transform_kwargs['u']
    shear     = transform_kwargs['s']
    shape     = th_img.shape[2:]
    
    u,d,l,r = 0,0,0,0
    
    max_dim = max(shape[0], shape[1])
    
    shear = np.deg2rad(shear[0]), np.deg2rad(shear[1])
    shear = np.abs(np.tan(shear[0])), np.abs(np.tan(shear[1]))  
    # add shear
    diag = np.linalg.norm(shape)
    
    u += np.abs(shear[0] * shape[1])
    d += np.abs(shear[0] * shape[1])
    l += np.abs(shear[1] * shape[0])
    r += np.abs(shear[1] * shape[0])
    
    u = int(u) + random.randint(0,4)
    d = int(d) + random.randint(0,4)
    l = int(l) + random.randint(0,4)
    r = int(r) + random.randint(0,4)
    # u,d = int(shape[0] * 1.1), int(shape[0] * 1.1)
    # l,r = int(shape[1] * 0.1), int(shape[1] * 0.1)
    return u,d,l,r

  def _cut_pad_th_image(self, th_img_padded, th_img, offset, transform_kwargs):
    u,d,l,r = self._pad_th_image(th_img, transform_kwargs)
    
    
    ou, ol = offset
    h,w = th_img.shape[-2:]
    
    ph, pw = th_img_padded.shape[-2:]
    
    sh, eh = max(ou - u, 0), min(ou + h + d, ph)
    sw, ew = max(ol - l, 0), min(ol + w + r, pw)
    
    return th_img_padded[:, :, sh:eh, sw:ew]

  def _get_lp_from_image(self, th_image_padded, offset, th_img_shape):
    h,w = th_img_shape
    ou, ol = offset
    return th_image_padded[:, :, ou:ou+h, ol:ol+w]

  def _simulate_pad(self, th_img, h, w):
    shape = th_img.shape[-2:]
    des = (h,w)
    
    scale = min(des[0] / shape[0], des[1] / shape[1])
    pad_h = -abs(des[0] - shape[0] * scale) / des[0]
    pad_w = -abs(des[1] - shape[1] * scale) / des[1]
    
    theta = self._compute_matrix((pad_h //2, pad_w//2), 0, scale, (0,0)).unsqueeze(0)
    return self._transform(th_img, theta)

  def _cut_bands(self, th_image, size_before_after):
    size_after, (size_before, ) = size_before_after
    scale = min(size_after[0] / size_before[0], size_after[1] / size_before[1])
    pad = max(size_after[0] - size_before[0] * scale, size_after[1] - size_before[1] * scale) / 2

    h, w = th_image.shape[-2:]
    new_w = int(w - pad)
    new_h = int((new_w + pad) / 4)
    
    pad_w = int(pad / 2)
    pad_h = int((h - new_h) / 2)
    
    return th_image[:,:,pad_h:new_h+pad_h, pad_w:new_w+pad_w]

  def _change_background(self, th_image):
    th_image = th_image.permute(0,2,3,1)
    blacks = (th_image == th.zeros(3).to(device="cuda")).all(-1)
    th_image[blacks] =  th.Tensor([0,0,1]).to(device="cuda")
    th_image = th_image.permute(0,3,1,2)

  def _get_shear_mat(self, theta):
    tx, ty = theta
    mx = th.tan(th.tensor(tx))
    my = th.tan(th.tensor(ty))
    shear_mat = th.tensor(
      [[1, mx, 0],
      [my, 1, 0],
      [0, 0, 1]]).type(th.float32)
    return shear_mat

  def _compute_matrix(self, t, r, u, s, image_shape):
    """Compute the transform theta from a transform matrix, adapted for pythorch affine_grid

    Args:
        t (tuple[float]): Translate distance (in pixels)
        r (float): rotation angle (in degrees)
        u (float): uniform scale factor
        s (float): Shear angle (in degrees)
        image_shape (tuple): The size of the original image, `(height, width)`.

    Returns:
        th.Tensor: the theta used for th..nn.functional.affine_grid()
    """
    Mt = th.Tensor([
      [1, 0, 2 * t[1] / image_shape[1]],
      [0, 1, 2 * t[0] / image_shape[0]],
      [0, 0, 1]
    ]).type(th.float32)

    r = np.deg2rad(r)
    Mr = th.Tensor([
      [np.cos(r), np.sin(r), 0],
      [-np.sin(r), np.cos(r), 0],
      [0, 0, 1]
    ]).type(th.float32)
    Mu = th.Tensor([
      [1/u, 0, 0],
      [0, 1/u, 0],
      [0, 0, 1]
    ]).type(th.float32)
    s = np.deg2rad(s[0]), np.deg2rad(s[1])
    Ms = self._get_shear_mat(s)

    M = Mt @ Mu @ Ms @ Mr # Maybe add translate to move the image in the center
    return th.Tensor(M[:2]).to(device="cuda")

  def _generate_random_value(self, list_range):
    index = random.randint(0, len(list_range)-1)
    range = list_range[index]
    min, max = range
    return random.random() * (max - min) + min

  def _compute_random_matrix(self, *, translate_range, rotation_degrees_range=None, uniform_scale_range, shear_range):
    tx = self._generate_random_value(translate_range)
    ty = self._generate_random_value(translate_range) * 20
    u  = self._generate_random_value(uniform_scale_range)
    sx = self._generate_random_value(shear_range)
    sy = self._generate_random_value(shear_range)
    if rotation_degrees_range is not None:
      rd = self._generate_random_value(rotation_degrees_range)
    else:
      rd = 50 * sx
    return self._compute_matrix((tx, ty), rd, u, (sx, sy))

  def _read_image(self, path):
    np_original = cv2.imread(path, cv2.IMREAD_COLOR)
    if np_original is not None:
      # if this line is causing a bug it will be replaced with: np_original = cv2.cvtColor(np_original, cv2.COLOR_BGR2RGB)
      # until further testing this will remain
      np_original = np_original[:, :, ::-1]
    return np_original
    
  def _add_pad(self, th_img, delta : tuple = None):
    if delta is not None:
      delta_h, delta_w = delta
    else:
      delta_h, delta_w = 200, 200
      
    th_img_new = th.zeros(th_img.shape[0], th_img.shape[1], th_img.shape[2] + delta_h, th_img.shape[3] + delta_w).to(dev)

    th_img_new[:, :, delta_h//2 : th_img.shape[2] + delta_h // 2, delta_w // 2 : th_img.shape[3] + delta_w // 2] = th_img
    return th_img_new

  def _transform(self, th_img, th_theta):
    shape = th_img.shape

    transform_matrix = th.nn.functional.affine_grid(th_theta, shape).to(dev)

    th_transformed = th.nn.functional.grid_sample(th_img, transform_matrix)
    return th_transformed

  def _compute_inverse(self, th_theta, size_before_after=None):
    global dev
    pad = th.Tensor([[0, 0, 1]]).to(dev).repeat(th_theta.shape[0], 1, 1)
    M = th.cat([th_theta, pad], dim=1)
    
    scale_h, scale_w = 1, 1
    if size_before_after is not None:
      size_after, (size_before, ) = size_before_after
      scale_h, scale_w = size_after[0] / size_before[0], size_after[1] / size_before[1]
      
    Ms = th.Tensor([
      [1/scale_w, 0, 0],
      [0, 1/scale_h, 0],
      [0, 0, 1]
    ]).to(device="cuda").unsqueeze(0)
    
    M = Ms @ M
        
    
    inv = Ms @ th.inverse(M)
    return inv[:,:2]

  def _th2np(self, th_img):
    if len(th_img.shape) == 3:
      return (th_img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    elif len(th_img.shape) == 4:
      return (th_img * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

  def _np2th(self, *args):
    return T.ToTensor()(*args)


if __name__ == "__main__":
  # dataset_path = "C:\\Users\\Stefan.saraev\\Downloads\\LP frontal\\Doar placute frontale"
  # save_path = "C:\\Users\\Stefan.saraev\\Downloads\\LP frontal\\Placute rotite"

  offset_dataset_path = "C:\\Users\\Stefan.saraev\\Downloads\\Frontal cars 200 img\\Frontal cars 200 img"
  dev = th.device("cuda")
  
  # Read dataset
  dsBuilder = DS_Builder(
    offset_dataset_path,
    dataset='theta', 
    input_dim=(120, 200),
    dev=dev,
    batch_size=50
  )
  
  generator = dsBuilder.get_generator()
  
  for i in range(10):
    th_image, th_theta = next(generator)
    dsBuilder.show_images(th_image)

  dsBuilder = DS_Builder(
    offset_dataset_path,
    dataset='encoder', 
    input_dim=(120, 200),
    output_dim=(24,94),
    target_value="processed",
    dev=dev
  )
  
  generator = dsBuilder.get_generator()
  
  for i in range(10):
    th_image, th_res = next(generator)
    dsBuilder.show_images(th_image, th_res)
  
  
  dsBuilder = DS_Builder(
    offset_dataset_path,
    dataset='encoder', 
    input_dim=(120, 200),
    output_dim=(24,94),
    target_value="original",
    dev=dev
  )
  
  generator = dsBuilder.get_generator()
  
  for i in range(10):
    th_image, th_res = next(generator)
    dsBuilder.show_images(th_image, th_res)
  # for iteration in range(ITERATIONS):
  #   for i, (label, np_image) in enumerate(zip(labels, lst_np_images)):
  #     # Generate Theta2
  #     transform_kwargs = generate_random_kwargs(
  #       translate_range = [(-7, 7)],
  #       uniform_scale_range = [(1, 1)],
  #       shear_range = [(-30, -12), (12, 30), (-30, 30)],
  #     )
      
  #     # injura-ma, si asa trebuie sa incarc fiecare poza individual :P
  #     th_image = self.np2th(np_image).unsqueeze(0).to(dev)
      
  #     if not READ_PADS:
  #       # Add pad
  #       th_image_padded, offset = simple_pad_th_image(th_image)
  #     else:
  #       th_image_padded = th_image
  #       offset = lst_offsets[i][:2]
  #       th_img_shape = lst_offsets[i][-2:]
  #       th_image = get_lp_from_image(th_image_padded, offset, th_img_shape)
  #     # Change Background
  #     # change_background(th_image_padded)
      
  #     # Cut padded
  #     th_image_padded = cut_pad_th_image(th_image_padded, th_image, offset, transform_kwargs)
      
  #     # Add image shape to transform parameters
  #     transform_kwargs['image_shape'] = th_image_padded.shape[2:]
      
  #     # Compute theta2
  #     th_theta2 = self.compute_matrix(**transform_kwargs).unsqueeze(0)
  #     add_theta_to_dict(th_theta2, thetas, label, iteration)
      
  #     # Apply Transform
  #     th_transformed = self.transform(th_image_padded, th_theta2)
      
  #     # Change Background
  #     # th_bk_changed = change_background(th_transformed)
      
  #     # Apply resize with pad
  #     th_transformed_padded, orig_size = th_resize_with_pad(th_transformed, 120, 200, device=dev, fill_value=0)
  #     add_image_to_dict(th_transformed_padded, images, label, iteration)
      
  #     # Compute Theta1
  #     th_theta1 = self.compute_inverse(th_theta2, (th_transformed_padded.shape[2:], orig_size))
  #     add_theta_to_dict(th_theta1, i_thetas, label, iteration)

  #     th_i_transformed = self.transform(th_transformed_padded, th_theta1)
      
  #     th_bands = cut_bands(th_i_transformed, (th_transformed_padded.shape[2:], orig_size))
      
  #     shape = (24, 94)
  #     th_final = Resize(size=shape)(th_bands)
      
  #     show_images(th_image, th_image_padded, th_transformed, th_transformed_padded, th_i_transformed, th_bands, th_final)
  #     pad_th_image(th_image_padded, transform_kwargs)
      
  # Save the data
  # data_to_save = merge_dicts(images, thetas, i_thetas) 
  # save_data(data_to_save, save_path, 10/3)
    
    # Apply inverse transform
    # th_i_transformed_padded = self.transform(th_transformed_padded, th_theta1)
        
    # show_one_image(th_image_padded)
    # show_one_image(th_transformed)
    # show_one_image(th_transformed_padded)
    # show_one_image(th_i_transformed_padded)
    # show_one_image(th_resize_with_pad(th_image_padded, 120, 200, device=dev, fill_value=0)[0])
  
    # print("Distance between padded and inverse transformed is {}".format(th.dist(th_transformed_padded, th_i_transformed_padded).cpu().numpy()))
    # break