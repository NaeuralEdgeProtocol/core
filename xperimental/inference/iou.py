import numpy as np

def nms(dets, thresh):
  """Non-maximum suppression."""
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    intersection = w * h
    overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

    inds = np.where(overlap <= thresh)[0]
    order = order[inds + 1]
  return keep
  

if __name__ == '__main__':
  # top1, left1, bottom1, right1 = [98, 1100, 265, 1252]
  # top2, left2, bottom2, right2 = [115, 1092, 377, 1253]
  
  top1, left1, bottom1, right1 = [116, 967, 238, 1103]
  top2, left2, bottom2, right2 = [126, 965, 383, 1104]

  
  top = max(top1, top2)
  left = max(left1, left2)
  right = min(right1, right2)
  bottom = min(bottom1, bottom2)
  
  h_overlap = bottom - top + 1
  w_overlap = right - left + 1
  
  area_overlap = h_overlap * w_overlap
  
  area_union = (right1 - left1 + 1) * (bottom1 - top1 + 1) + \
          (right2 - left2 + 1) * (bottom2 - top2 + 1) - \
          area_overlap
  iou = area_overlap / area_union
  print(area_overlap, area_union, area_overlap / area_union)
  
  
  arr1 = np.array([top1, left1, bottom1, right1, 69])
  arr11 = np.array([top1+1, left1+2, bottom1+2, right1+2, 50])
  arr2 = np.array([top2, left2, bottom2, right2, 22])
  keep = nms(np.vstack([arr1, arr11, arr2]), 0.5)
  print(keep)
