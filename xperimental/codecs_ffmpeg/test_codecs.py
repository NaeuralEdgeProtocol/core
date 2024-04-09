import cv2
import os

CODECS = ["mp4v", 'MP4V', "XVID" , "MJPG", "DIVX", "FMP4", "avc1", "AVC1", "H264", "h264"]
EXTS = ['.avi', '.mp4']

if __name__:
  cnt = 0
  for ext in EXTS:
    for codec in CODECS:
      cnt += 1
      print("Trying {}/{}:  ".format(ext, codec), flush=True, end=' ')
      fn = 'test{}.avi'.format(cnt)
      out = cv2.VideoWriter(
        filename=fn, 
        fourcc=cv2.VideoWriter_fourcc(*codec), 
        fps=10, 
        frameSize=(1280,720)
      )
      if out.isOpened():
        print("{} with {} is working OK".format(ext, codec), flush=True)
      else:
        print("{} with {} is NOT working!".format(ext, codec), flush=True)
      out.release()
      os.remove(fn)
      