import os
from decord import VideoReader, cpu

if __name__ == '__main__':
  path_fld = r'C:\Users\ETA\Dropbox\MKT\Blur\20220125_Blur Alex\BLUR 25.01.2022\Originals'
  files = [
    'Camera bariere access Kaufland.avi',
    'output_500mb_part_000.mkv',
    'output_500mb_part_001.mkv',
    'output_500mb_part_002.mkv',
    'output_500mb_part_003.mkv',
    'output_500mb_part_004.mkv',
    'output_500mb_part_005.mkv',
    'output_500mb_part_006.mkv',
  ]

  for name in files:
    path_file = os.path.join(path_fld, name)
    vr = VideoReader(path_file, ctx=cpu(0))
    print('{}: {}'.format(name, len(vr)))
  #endfor


