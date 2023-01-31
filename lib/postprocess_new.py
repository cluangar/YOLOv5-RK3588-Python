import numpy as np
import lib.config as config
#from lib.cython_post import sigmoid
from lib.cython_post import sigmoid, xywh2xyxy, nms_boxes, process, filter_boxes, letterbox_reverse_box

OBJ_THRESH = config.OBJ_THRESH
NMS_THRESH = config.NMS_THRESH
IMG_SIZE = config.IMG_SIZE

def letterbox_reverse_box_in(x1, y1, x2, y2, width, height, new_width, new_height, dw, dh):

    w_scale = width / new_width
    h_scale = height / new_height

    x1 = (x1-dw)*w_scale
    x2 = (x2+dw)*w_scale
    y1 = (y1-dh)*h_scale
    y2 = (y2+dh)*h_scale
    
    return [x1, y1, x2, y2]

#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
    
#def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
#    y = np.copy(x)
#    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#    return y

def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
#**-**
#        arr_mask = np.array(mask, dtype=int, ndmin=1)
#        arr_anchors = np.array(anchors, dtype=int, ndmin=2)
        arr_mask = np.array(mask, dtype=np.int32)
        arr_anchors = np.array(anchors, dtype=np.int32)
#**-**
#        b, c, s = process_in(input,mask , anchors)
#        b, c, s = process_in(input,arr_mask , arr_anchors)
        b, c, s = process(input, arr_mask, arr_anchors)
 
#        b, c, s = filter_boxes_in(b, c, s)
        b, c, s = filter_boxes(b, c, s)
                
#        print("[Debug] Shape b = {}".format(np.shape(b)))
#        print("[Debug] Shape c = {}".format(np.shape(c)))
#        print("[Debug] Value c = {}".format(c))
#        print("[Debug] Shape s = {}".format(np.shape(s)))
        
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

#        print(np.shape(b))
#        print(s.dtype)
        keep = nms_boxes(b, s)
#        keep = nms_boxes_in(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def process_in(input, mask, anchors):
## input 4D, mask 1D, anchors 2D
## input np.array, list, list
## input float, int, int (2Darray)
##    print(type(anchors))
##    print(input.dtype)
##    print(np.shape(anchors))
##    print(anchors)
    
#    anchors = [anchors[i] for i in mask]
    anchors = np.array([anchors[i] for i in mask])
#    grid_h, grid_w = map(int, input.shape[0:2])
    grid_h, grid_w = [int(x) for x in input.shape[0:2]]
    
    box_confidence = sigmoid(input[..., 4])	#3D

    box_confidence = np.expand_dims(box_confidence, axis=-1)	#4D

    box_class_probs = sigmoid(input[..., 5:])		#4D

    box_xy = sigmoid(input[..., :2])*2 - 0.5		#4D

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)	#4D

#    print("[Debug] before multiply box_wh ={}".format(box_wh))
    box_wh = box_wh * anchors
#    print("[Debug] after multiply box_wh ={}".format(box_wh))

    box = np.concatenate((box_xy, box_wh), axis=-1)	#4D
#    print(np.shape(box))
    return box, box_confidence, box_class_probs
    
def nms_boxes_in(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep    
    
def filter_boxes_in(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)	#2D
    box_confidences = box_confidences.reshape(-1)	#1D
#    print("[Debug] box_confidences = {}".format(np.shape(box_confidences)))
#    print("[Debug] box_confidences value = {}".format(box_confidences))
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])	#2D
    _box_pos = np.where(box_confidences >= OBJ_THRESH)	#2D
    boxes = boxes[_box_pos]	#2D

#    print("[Debug] shape _box_pos = {}".format(np.shape(_box_pos)))
#    print("[Debug] value _box_pos = {}".format(_box_pos)) 
#    print("[Debug] shape result_boxes = {}".format(np.shape(boxes)))
#    print("[Debug] value result_boxes = {}".format(boxes))

    box_confidences = box_confidences[_box_pos]	#1D
 
#    print("[Debug] shape result_box_confidence = {}".format(np.shape(box_confidences)))
#    print("[Debug] value result_box_confidence = {}".format(box_confidences))
 
    box_class_probs = box_class_probs[_box_pos]	#2D

#    print("[Debug] shape result_box_class_probs = {}".format(np.shape(box_class_probs)))
#    print("[Debug] value result_box_class_probs = {}".format(box_class_probs))

    
    class_max_score = np.max(box_class_probs, axis=-1)	#1D
    
#    print("[Debug] shape class_max_score = {}".format(np.shape(class_max_score)))
#    print("[Debug] value class_max_score = {}".format(class_max_score))
    
    classes = np.argmax(box_class_probs, axis=-1)	#1D
    
#    print("[Debug] shape classes = {}".format(np.shape(classes)))
#    print("[Debug] value classes = {}".format(classes))
        
    _class_pos = np.where(class_max_score >= OBJ_THRESH)	#1D

#    print("[Debug] shape _class_pos = {}".format(np.shape(_class_pos)))
#    print("[Debug] value _class_pos = {}".format(_class_pos))
    
    boxes = boxes[_class_pos]
    
#    print("[Debug] shape result2_boxes = {}".format(np.shape(boxes)))
#    print("[Debug] value result2_boxes = {}".format(boxes))    
    
    classes = classes[_class_pos]

#    print("[Debug] shape result_classes = {}".format(np.shape(classes)))
#    print("[Debug] value result_classes = {}".format(classes))
    
    scores = (class_max_score* box_confidences)[_class_pos]

#    print("[Debug] shape scores = {}".format(np.shape(scores)))
#    print("[Debug] value scores = {}".format(scores))
    
    return boxes, classes, scores    



