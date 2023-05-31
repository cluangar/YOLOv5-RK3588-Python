cimport cython
cimport numpy as np
from cython cimport floating
from numpy cimport ndarray

import numpy as np
import lib.config as config

#-from libc.stdlib cimport malloc, free

OBJ_THRESH = config.OBJ_THRESH
NMS_THRESH = config.NMS_THRESH
IMG_SIZE = config.IMG_SIZE

#-def ListArray(a, int len):

#-    cdef int *my_ints

#-    my_ints = <int *>malloc(len(a)*cython.sizeof(int))
#-    if my_ints is NULL:
#-        raise MemoryError()

#-    for i in xrange(len(a)):
#-        my_ints[i] = a[i]

#-    with nogil:
        #Once you convert all of your Python types to C types, then you can release the GIL and do the real work
#-        ...
#-        free(my_ints)

#-    #convert back to python return type
#-    return value

cpdef ListArrayInt(a, int len):
    cdef int y
    for i in range(len):
        y = a[i]
        a[i] = y
    return a

#cpdef sigmoid(np.ndarray[float, ndim=3] x):
#    cpdef float y
#    y = 1 / (1 + np.exp(-x))
#    return y
#    return 1 / (1 + np.exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


cpdef xywh2xyxy(np.ndarray[floating, ndim=2] x):
    cdef np.ndarray[floating, ndim=2] y
#    cdef np.int32_t i, j
#    cdef np.float64_t val
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
#    for i in range(y.shape[0]):
#        for j in range(y.shape[1]):
#            val = y[i, j]
#            if j == 0:
#                val -= y[i, 2] / 2  # top left x
#            elif j == 1:
#                val -= y[i, 3] / 2  # top left y
#            elif j == 2:
#                val += y[i, 2] / 2  # bottom right x
#            elif j == 3:
#                val += y[i, 3] / 2  # bottom right y
#            y[i, j] = val
    return y


#cpdef nms_boxes(np.ndarray[double, ndim=2] boxes, np.ndarray[float, ndim=1] scores):
#cpdef nms_boxes(np.ndarray[double, ndim=2] boxes, np.ndarray[double, ndim=1] scores):
cpdef nms_boxes(np.ndarray[floating, ndim=2] boxes, np.ndarray[floating, ndim=1] scores):
    cdef int i
#    cdef np.ndarray[np.float_t, ndim=1] x, y, w, h, areas, keep
    cdef np.ndarray[floating, ndim=1] x, y, w, h, areas
#    cdef list keep
    cdef np.ndarray[long, ndim=1] order

#def nms_boxes(boxes, scores):
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
    keep = np.array(keep, dtype=np.int32)
#    keep = np.empty(0, dtype=np.float)
    
    while order.size > 0:
        i = order[0]
#        keep.append(i)
#        keep += [i]
        keep.resize((i,))

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

#def process(input, mask, anchors):
#def process(np.ndarray[floating, ndim=4] input, np.ndarray[np.int32_t, ndim=1] mask, list[float] anchors):
#def process(np.ndarray[floating, ndim=4] input, list[int] mask, list[int] anchors):
#cpdef process(float[:,:,:,:] input, int[:] mask, int[:,:] anchors):
#cpdef process(float[:,:,:,:] input, int[:] list_mask, int[:,:] list_anchors):
cpdef process(np.ndarray[floating, ndim=4] input, np.ndarray[np.int32_t, ndim=1] mask, np.ndarray[np.int32_t, ndim=2] anchors):
#def process(np.ndarray[floating, ndim=4] input, char[:] list_mask, char[:] list_anchors):
    cdef int grid_h, grid_w
    cdef np.ndarray[floating, ndim=3] box_confidence
    cdef np.ndarray[floating, ndim=4] d4_box_confidence
    cdef np.ndarray[floating, ndim=4] box_class_probs
    cdef np.ndarray[floating, ndim=4] box_xy
    cdef np.ndarray[floating, ndim=4] box_wh
    cdef np.ndarray[floating, ndim=4] box

#    cdef float[:,:,:] box_confidence
#    cdef float[:,:,:,:] box_class_probs
#    cdef float[:,:,:,:] box_xy
#    cdef float[:,:,:,:] box_wh
#    cdef float[:,:,:,:] box

#    cdef np.ndarray[int, ndim=1] mask
#    cdef np.ndarray[int, ndim=2] anchors


###    cdef int[[:]] anchors
#    mask = 	np.array(list_mask, dtype=np.int)		    
#    anchors = 	np.array(list_anchors, dtype=np.int)		    
#    mask = np.frombuffer(list_mask, dtype=np.int)
#    anchors = np.frombuffer(list_anchors, dtype=np.int)


#    anchors = [anchors[i] for i in mask]
    anchors = np.array([anchors[i] for i in mask])
#    grid_h, grid_w = map(int, input.shape[0:2])
    grid_h, grid_w = [int(x) for x in input.shape[0:2]]

#    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.array(sigmoid(input[..., 4])).astype(np.float32)
#    box_confidence = np.expand_dims(box_confidence, axis=-1)
    d4_box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = np.array(sigmoid(input[..., 5:])).astype(np.float32)

    box_xy = np.array(sigmoid(input[..., :2])*2 - 0.5).astype(np.float32)

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = np.array(pow(sigmoid(input[..., 2:4])*2, 2)).astype(np.float32)

    box_wh = np.array(box_wh * anchors).astype(np.float32)
    box = np.concatenate((box_xy, box_wh), axis=-1)

#   box_class_probs = np.empty(0, dtype=np.float32)    
#   box = np.empty(0, dtype=np.float32)

    return box.astype(np.float32), d4_box_confidence.astype(np.float32), box_class_probs.astype(np.float32)

    
    
