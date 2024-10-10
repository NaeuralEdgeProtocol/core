import cv2
from naeural_core.local_libraries.vision.watermark import apply_watermark, apply_watermark_from_file, prepare_watermark
from naeural_core import Logger

if __name__ == '__main__':
  l = Logger('WMRK', base_folder='.', app_folder='_local_cache')
  # img = cv2.imread('core/xperimental/_images/H1080_W1920/faces7.jpg')
  img = cv2.imread('core/xperimental/_images/H480_W720/covid1.jpg')
  wm = prepare_watermark(
    fn_watermark="plugins/images/watermark.png", 
    target_shape_HW=(480, 720)
  )
  cv2.imshow("wm", wm)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  for _ in range(100): 
    test = img.copy()
    l.start_timer('apply_wm')
    test = apply_watermark(
      np_bgr=test,
      np_watermark=wm,
      )
    l.stop_timer('apply_wm')
  cv2.imshow("test", test)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  l.show_timers()