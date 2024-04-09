import numpy as np
import torch as th
import torchvision as tv

def box_iou(box1, box2):
  # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
  """
  Return intersection-over-union (Jaccard index) of boxes.
  Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
  Arguments:
      box1 (Tensor[N, 4])
      box2 (Tensor[M, 4])
  Returns:
      iou (Tensor[N, M]): the NxM matrix containing the pairwise
          IoU values for every element in boxes1 and boxes2
  """

  def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])

  area1 = box_area(box1.T)
  area2 = box_area(box2.T)

  # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
  inter = (th.min(box1[:, None, 2:], box2[:, 2:]) - th.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
  return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
  """
  Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
  top-left corner and (x2, y2) is the bottom-right corner.

  Args:
      x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
  Returns:
      y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
  """
  y = x.clone() 
  y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
  y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
  y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
  y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
  return y

@th.jit.script
def y5_nms_topk(prediction,
           # conf_thres:float=0.25, 
           # iou_thres:float=0.45, 
           # classes=None, 
           # agnostic=False, 
           # multi_label:bool=False,
           # max_det:int=300
           ):
  """Runs Non-Maximum Suppression (NMS) on inference results

  Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
  """

  nc = prediction.shape[2] - 5  # number of classes
  xc = prediction[..., 4] > 0.25  # candidates #conf_thres
  # Settings
  max_wh = 7680  # (pixels) minimum and maximum box width and height
  max_nms = 30000  # maximum number of boxes into tv.ops.nms()
  n_candidates = 6
  bs = prediction.shape[0]
  
  # output = [th.zeros((0, 6), device=prediction.device)] * 256 # max batch size 256
  th_output = th.zeros((bs, 300, 4 + 2 * n_candidates), device=prediction.device)
  th_n_det = th.zeros(bs, device=prediction.device, dtype=th.int32)
  for xi in range(bs):  # image index, image inference
    x = prediction[xi]
    x = x[xc[xi]]  # confidence

    # If none remain process next image
    if x.shape[0] > 0:
      # Compute conf
      x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

      # Box (center x, center y, width, height) to (x1, y1, x2, y2)
      box = xywh2xyxy(x[:, :4])

      # Detections matrix nx(4+2*n_candidates) (xyxy, conf1, cls1, .., confi, clsi)
      # conf, j = x[:, 5:].max(1, keepdim=True) # confidente & class-id
      # x = th.cat((box, conf, j.float()), 1)[conf.view(-1) > 0.25] #conf_thres

      top_conf, top_idxs = x[:, 5:].sort(1, descending=True)
      mask = top_conf[:, 0] > 0.25
      runners = th.stack([top_conf[:, :n_candidates], top_idxs[:, :n_candidates]], dim=2).view(top_conf.shape[0],
                                                                                                   2 * n_candidates)
      x = th.cat((box, runners), 1)[mask]

      # Check shape
      n = x.shape[0]  # number of boxes
      if n > 0:  # no boxes  
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes preparred as offsets
        boxes = x[:, :4] + c # add offeset to put each class in "different scene"
        scores = x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.nms(boxes, scores, 0.45)  # NMS #iou_thres
        i = i[:300] #max_det

        th_n_det[xi] = i.shape[0]
        th_output[xi, :th_n_det[xi], :] = x[i]

  return th_output, th_n_det


@th.jit.script
def y5_nms(
    prediction,
   # conf_thres:float=0.25,
   # iou_thres:float=0.45,
   # classes=None,
   # agnostic=False,
   # multi_label:bool=False,
   # max_det:int=300
):
  """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
          list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

  nc = prediction.shape[2] - 5  # number of classes
  xc = prediction[..., 4] > 0.25  # candidates #conf_thres
  # Settings
  max_wh = 7680  # (pixels) minimum and maximum box width and height
  max_nms = 30000  # maximum number of boxes into tv.ops.nms()
  bs = prediction.shape[0]

  # output = [th.zeros((0, 6), device=prediction.device)] * 256 # max batch size 256
  th_output = th.zeros((bs, 300, 6), device=prediction.device)
  th_n_det = th.zeros(bs, device=prediction.device, dtype=th.int32)
  for xi in range(bs):  # image index, image inference
    x = prediction[xi]
    x = x[xc[xi]]  # confidence

    # If none remain process next image
    if x.shape[0] > 0:
      # Compute conf
      x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

      # Box (center x, center y, width, height) to (x1, y1, x2, y2)
      box = xywh2xyxy(x[:, :4])

      # Detections matrix nx(4+2*n_candidates) (xyxy, conf1, cls1, .., confi, clsi)
      conf, j = x[:, 5:].max(1, keepdim=True) # confidente & class-id
      x = th.cat((box, conf, j.float()), 1)[conf.view(-1) > 0.25] #conf_thres

      # Check shape
      n = x.shape[0]  # number of boxes
      if n > 0:  # no boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes preparred as offsets
        boxes = x[:, :4] + c  # add offeset to put each class in "different scene"
        scores = x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.nms(boxes, scores, 0.45)  # NMS #iou_thres
        i = i[:300]  # max_det

        th_n_det[xi] = i.shape[0]
        th_output[xi, :th_n_det[xi], :] = x[i]

  return th_output, th_n_det


@th.jit.script
def y8_nms_topk(
        prediction,
        # conf_thres:float=0.25,
        # iou_thres:float=0.45,
        # classes=None,
        # agnostic=False,
        # multi_label:bool=False,
        # max_det:int=300,
        # max_time_img:float=0.05,
        # max_nms:int=30000,
        # max_wh:int=7680,
):
  """
  Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

  Arguments:
      prediction (th.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
          containing the predicted boxes, classes, and masks. The tensor should be in the format
          output by a model, such as YOLO.
      conf_thres (float): The confidence threshold below which boxes will be filtered out.
          Valid values are between 0.0 and 1.0.
      iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
          Valid values are between 0.0 and 1.0.
      classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
      agnostic (bool): If True, the model is agnostic to the number of classes, and all
          classes will be considered as one.
      multi_label (bool): If True, each box may have multiple labels.
      labels (List[List[Union[int, float, th.Tensor]]]): A list of lists, where each inner
          list contains the apriori labels for a given image. The list should be in the format
          output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
      max_det (int): The maximum number of boxes to keep after NMS.
      nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
      max_time_img (float): The maximum time (seconds) for processing one image.
      max_nms (int): The maximum number of boxes into tv.ops.nms().
      max_wh (int): The maximum box width and height in pixels

  Returns:
      (List[th.Tensor]): A list of length batch_size, where each element is a tensor of
          shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
          (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
  """

  bs = prediction.shape[0]  # batch size
  nc = prediction.shape[1] - 4  # number of classes
  nm = 0 # prediction.shape[1] - nc - 4
  mi = 4 + nc  # mask start index
  xc = prediction[:, 4:mi].amax(1) > 0.25  # candidates  #conf_thres
  n_candidates = 6

  th_output = th.zeros((bs, 300, 4 + 2 * n_candidates + nm), device=prediction.device)
  th_n_det = th.zeros(bs, device=prediction.device, dtype=th.int32)

  for xi in range(prediction.shape[0]):  # image index, image inference
    x = prediction[xi]
    x = x.transpose(0, -1)[xc[xi]]  # confidence

    # If none remain process next image
    if x.shape[0] > 0:

      # Detections matrix nx6 (xyxy, conf, cls)
      box, cls, mask = x.split((4, nc, nm), 1)
      box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

      # conf, j = cls.max(1, keepdim=True)
      # x = th.cat((box, conf, j.float(), mask), 1)

      top_conf, top_idxs = x[:, 4:].sort(1, descending=True)
      runners = th.stack([top_conf[:, :n_candidates], top_idxs[:, :n_candidates]], dim=2).view(top_conf.shape[0],
                                                                                               2 * n_candidates)
      x = th.cat((box, runners, mask), 1)

      # Check shape
      n = x.shape[0]  # number of boxes
      if n > 0:  # no boxes
        x = x[x[:, 4].argsort(descending=True)[:30000]]  # sort by confidence and remove excess boxes #max_nms

        # Batched NMS
        c = x[:, 5:6] * 7680  # classes #max_wh
        boxes = x[:, :4] + c
        scores = x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.nms(boxes, scores, 0.45)  # NMS #iou_thres
        i = i[:300]  # limit detections #max_det

        th_n_det[xi] = i.shape[0]
        th_output[xi, :th_n_det[xi], :] = x[i]
          
  return th_output, th_n_det


@th.jit.script
def y8_nms(
    prediction,
    # conf_thres:float=0.25,
    # iou_thres:float=0.45,
    # classes=None,
    # agnostic=False,
    # multi_label:bool=False,
    # max_det:int=300,
    # max_time_img:float=0.05,
    # max_nms:int=30000,
    # max_wh:int=7680,
):
  """
  Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

  Arguments:
      prediction (th.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
          containing the predicted boxes, classes, and masks. The tensor should be in the format
          output by a model, such as YOLO.
      conf_thres (float): The confidence threshold below which boxes will be filtered out.
          Valid values are between 0.0 and 1.0.
      iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
          Valid values are between 0.0 and 1.0.
      classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
      agnostic (bool): If True, the model is agnostic to the number of classes, and all
          classes will be considered as one.
      multi_label (bool): If True, each box may have multiple labels.
      labels (List[List[Union[int, float, th.Tensor]]]): A list of lists, where each inner
          list contains the apriori labels for a given image. The list should be in the format
          output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
      max_det (int): The maximum number of boxes to keep after NMS.
      nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
      max_time_img (float): The maximum time (seconds) for processing one image.
      max_nms (int): The maximum number of boxes into tv.ops.nms().
      max_wh (int): The maximum box width and height in pixels

  Returns:
      (List[th.Tensor]): A list of length batch_size, where each element is a tensor of
          shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
          (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
  """

  bs = prediction.shape[0]  # batch size
  nc = prediction.shape[1] - 4  # number of classes
  nm = 0  # prediction.shape[1] - nc - 4
  mi = 4 + nc  # mask start index
  xc = prediction[:, 4:mi].amax(1) > 0.25  # candidates  #conf_thres

  th_output = th.zeros((bs, 300, 6 + nm), device=prediction.device)
  th_n_det = th.zeros(bs, device=prediction.device, dtype=th.int32)

  for xi in range(prediction.shape[0]):  # image index, image inference
    x = prediction[xi]
    x = x.transpose(0, -1)[xc[xi]]  # confidence

    # If none remain process next image
    if x.shape[0] > 0:

      # Detections matrix nx6 (xyxy, conf, cls)
      box, cls, mask = x.split((4, nc, nm), 1)
      box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

      conf, j = cls.max(1, keepdim=True)
      x = th.cat((box, conf, j.float(), mask), 1)

      # Check shape
      n = x.shape[0]  # number of boxes
      if n > 0:  # no boxes
        x = x[x[:, 4].argsort(descending=True)[:30000]]  # sort by confidence and remove excess boxes #max_nms

        # Batched NMS
        c = x[:, 5:6] * 7680  # classes #max_wh
        boxes = x[:, :4] + c
        scores = x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.nms(boxes, scores, 0.45)  # NMS #iou_thres
        i = i[:300]  # limit detections #max_det

        th_n_det[xi] = i.shape[0]
        th_output[xi, :th_n_det[xi], :] = x[i]

  return th_output, th_n_det

