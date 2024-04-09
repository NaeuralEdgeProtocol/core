"""
1. incarcam datele in 
"""

from generate_dataset import *
dataset_path = "C:\\Users\\Stefan.saraev\\Downloads\\LP frontal\\Placute rotite\\train\\images"
labels, th_images = read_dataset(dataset_path, add_pad=False)

show_one_image(th_images)
  