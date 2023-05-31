cimport cython
cimport numpy as np
from cython cimport floating
from numpy cimport ndarray

import numpy as np
import lib.config as config

cdef float OBJ_THRESH = config.OBJ_THRESH
cdef float NMS_THRESH = config.NMS_THRESH
cdef int IMG_SIZE = config.IMG_SIZE



#@cython.boundscheck(False)
#@cython.wraparound(False)

cdef sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def xywh2xyxy(np.ndarray[floating, ndim=2] x):
    cdef np.ndarray[floating, ndim=2] y
    cdef np.int32_t i, j
    cdef np.float64_t val
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            val = y[i, j]
            if j == 0:
                val -= y[i, 2] / 2  # top left x
            elif j == 1:
                val -= y[i, 3] / 2  # top left y
            elif j == 2:
                val += y[i, 2] / 2  # bottom right x
            elif j == 3:
                val += y[i, 3] / 2  # bottom right y
            y[i, j] = val
    return y

#def yolov5_post_process(input_data):
def yolov5_post_process(np.ndarray[floating, ndim=3] input_data):
    cdef list boxes, classes, scores
    cdef np.float64_t[:, :] b, c, s
    cdef np.int32_t i, j	
	
	
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
#        b, c, s = process(input, mask, anchors)
#        b, c, s = filter_boxes(b, c, s)
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

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

#def process(input, mask, anchors):
def process(np.ndarray[floating, ndim=4] input, np.ndarray[np.int32_t, ndim=1] mask, list[float] anchors):
    cdef int grid_h, grid_w
    cdef np.ndarray[floating, ndim=3] box_confidence
    cdef np.ndarray[floating, ndim=3] box_class_probs
    cdef np.ndarray[floating, ndim=3] box_xy
    cdef np.ndarray[floating, ndim=3] box_wh
    cdef np.ndarray[floating, ndim=4] box


    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs
    

def nms_boxes(np.ndarray[np.float_t, ndim=2] boxes, np.ndarray[np.float_t, ndim=1] scores):
    cdef int i
    cdef np.ndarray[np.float_t, ndim=1] x, y, w, h, areas, keep
    cdef np.ndarray[np.int32_t, ndim=1] order
    
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


    

def filter_boxes(np.ndarray[np.float_t, ndim=2] boxes, np.ndarray[np.float_t, ndim=1] box_confidences, np.ndarray[np.float_t, ndim=2] box_class_probs):
    cdef np.ndarray[np.float_t, ndim=2] _boxes = boxes.reshape(-1, 4)
    cdef np.ndarray[np.float_t, ndim=1] _box_confidences = box_confidences.reshape(-1)
    cdef np.ndarray[np.float_t, ndim=2] _box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    cdef np.ndarray[np.int32_t, ndim=1] _box_pos = np.where(_box_confidences >= OBJ_THRESH)
    boxes = _boxes[_box_pos]
    box_confidences = _box_confidences[_box_pos]
    box_class_probs = _box_class_probs[_box_pos]

    cdef np.ndarray[np.float_t, ndim=1] class_max_score = np.max(box_class_probs, axis=-1)
    cdef np.ndarray[np.int32_t, ndim=1] classes = np.argmax(box_class_probs, axis=-1)
    cdef np.ndarray[np.int32_t, ndim=1] _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores
   



