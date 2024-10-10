import numpy as np
import cv2

from naeural_core import Logger
from decentra_vision.draw_utils import DrawUtils


def cvshow(img):
  cv2.imshow("test", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return


def blur1(img, top, left, bottom, right, painter):
  return painter.blur_person(
    frame=img,
    top=top,
    left=left,
    bottom=bottom,
    right=right,
    method='gaussian',
  )


def blur2(img, top, left, bottom, right, painter):
  return painter.blur_person(
    frame=img,
    top=top,
    left=left,
    bottom=bottom,
    right=right,
    # method='scale',
  )


if __name__ == '__main__':
  N_TESTS = 100

  l = Logger('BLRS', base_folder='.', app_folder='_cache')
  painter = DrawUtils(log=l)

  img = cv2.imread('xperimental/_images/H2160_W3840/faces3.jpg')
  ref = img[100:800, 100:1300].copy()  # img[720:895, 742:872].copy()
  ref_blurred = blur2(ref.copy(), 0, 0, ref.shape[0], ref.shape[1], painter)

  for func in [blur1, blur2]:
    for i in range(N_TESTS):
      h = np.random.randint(200, 400)
      w = np.random.randint(75, 150)
      x = np.random.randint(0, img.shape[1] - (w + 1))
      y = np.random.randint(0, (img.shape[0]) - (h + 1))
      res = func(
        img=img,
        top=y,
        left=x,
        bottom=y + h,
        right=x + w,
        painter=painter
      )

    ref_blurred = func(ref.copy(), 0, 0, ref.shape[0], ref.shape[1], painter)
    cvshow(ref)
    cvshow(ref_blurred)

  l.show_timers()
