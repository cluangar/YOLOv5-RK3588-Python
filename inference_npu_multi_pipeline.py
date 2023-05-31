import logging as log
import time
import numpy as np
import cv2
import sys
import os
import signal
#from rknn.api import RKNN
from multiprocessing import Process, Queue, Lock
import multiprocessing

import platform
from rknnlite.api import RKNNLite
from lib.postprocess import yolov5_post_process
import lib.config as config

IMG_SIZE = config.IMG_SIZE

CLASSES = config.CLASSES

# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = config.DEVICE_COMPATIBLE_NODE

RK356X_RKNN_MODEL = config.RK356X_RKNN_MODEL
RK3588_RKNN_MODEL = config.RK3588_RKNN_MODEL

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                else:
                    host = 'RK356x'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def load_model():
    host_name = get_host()
    if host_name == 'RK356x':
        rknn_model = RK356X_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')	
    return rknn_lite
	
	

#def draw(image, boxes, scores, classes):
def draw(image, boxes, scores, classes, ratio, dw, dh):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        
        #reverse letterbox to original input cam
#        top, left, right, bottom = reverse_letterbox(top, left, right, bottom, ratio, dw, dh)
        
        
#        print('class: {}, score: {}'.format(CLASSES[cl], score))
#        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def reverse_letterbox(top, left, right, bottom, ratio, dw, dh):
    hr, wr = ratio
    print("[Debug] pad  dw = {}, dh = {}".format(dw, dh))
    print("[Debug] Check Ratio hr = {}, wr = {}".format(hr, wr))
    print("[Debug] ori top = {}, left = {}, right = {}, bottom = {}".format(top, left, right, bottom))
    
#    top = (top - (dh))/hr
#    bottom =  (bottom - (dh))/hr
#    left = (left - (dw))/wr
#    right -= (right - (dw))/wr
    x_scale = ( 1920 / (640-(dw*2)))
    y_scale = ( 1080 / (640-(dh*2)))
    
    left = ((left-dw)/wr)*x_scale
    top = ((top-dh)/hr)*y_scale	#ok
    right = ((right-dw)/wr)*x_scale		#ok
    bottom = ((bottom-dh)/hr)*y_scale

    print("[Debug] after top = {}, left = {}, right = {}, bottom = {}".format(top, left, right, bottom))

    
    return top, left, right, bottom
	



def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ("uvch264src device=/dev/video{} ! "
               "image/jpeg, width={}, height={}, framerate=30/1 ! "
               "jpegdec ! "
               "video/x-raw, format=BGR ! "
               "appsink").format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)





def video_capture(q_frame:Queue, q_image:Queue, flag, cam):
#    video = cv2.VideoCapture(0)
    video = open_cam_usb(cam, config.CAM_WIDTH, config.CAM_HEIGHT)    
    print("video.isOpened()={}", video.isOpened())
    try:
        while True:
            if flag.value == 20:
                if video.isOpened():
                    video.release()
                    print("video release!")
                print("exit video_capture!")
                break
            s = time.time()
            ret, frame = video.read()
            assert ret, 'read video frame failed.'
            #print('capture read used {} ms.'.format((time.time() - s) * 1000))

            s = time.time()
#            image = cv2.resize(frame, (416, 416))
            image, ratio, (dw, dh) = letterbox(frame, new_shape=(config.IMG_SIZE, config.IMG_SIZE))
            ori_image = image
#            image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #print('capture resize used {} ms.'.format((time.time() - s) * 1000))

            s = time.time()
            if q_frame.empty():
#                q_frame.put(frame)
#                q_frame.put([frame, ratio, (dw, dh)])
                q_frame.put([ori_image, ratio, (dw, dh)])
            if q_image.full():
                continue
            else:
                q_image.put(image)
            #print("capture put to queue used {} ms".format((time.time()-s)*1000))
    except KeyboardInterrupt:
        video.release()
        print("exit video_capture!")

def infer_rknn(q_image:Queue, q_infer:Queue, flag):
    rknn = load_model()
    try:
        while True:
            if flag.value == 10:
                print("befor exit infer rknn")
                rknn.release()
                print("exit infer_rknn!")
                flag.value = 20
                break
            s = time.time()
            if q_image.empty():
                continue
            else:
                image = q_image.get()
            #print('Infer get, used time {} ms. '.format((time.time() - s) * 1000))

            s = time.time()
            
#            out_boxes, out_boxes2 = rknn.inference(inputs=[image])
#            out_boxes = out_boxes.reshape(SPAN, LISTSIZE, GRID0, GRID0)
#            out_boxes2 = out_boxes2.reshape(SPAN, LISTSIZE, GRID1, GRID1)
#            input_data = []
#            input_data.append(np.transpose(out_boxes, (2, 3, 0, 1)))
#            input_data.append(np.transpose(out_boxes2, (2, 3, 0, 1)))

            # Inference
            outputs = rknn.inference(inputs=[image])

            # post process
            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]

            input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
            input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
            input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

            input_data = list()
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
            
            
            
            #print('Infer done, used time {} ms. '.format((time.time() - s) * 1000))

            s = time.time()
            if q_infer.full():
                continue
            else:
                q_infer.put(input_data)
            #print('Infer put, used time {} ms. '.format((time.time() - s) * 1000))
    except KeyboardInterrupt:
        print("befor exit infer rknn")
        rknn.release()
        print("exit infer_rknn!")

def post_process(q_infer, q_objs, flag):
    while True:
        if flag.value == 20:
            break
        s = time.time()
        if q_infer.empty():
            continue
        else:
            input_data = q_infer.get()
        #print('Post process get, used time {} ms. '.format((time.time() - s) * 1000))

        s = time.time()
        
#        boxes, classes, scores = yolov3_post_process(input_data)
        boxes, classes, scores = yolov5_post_process(input_data)
        
        #print('Post process done, used time {} ms. '.format((time.time() - s) * 1000))

        s = time.time()
        if q_objs.full():
            continue
        else:
            q_objs.put((boxes, classes, scores))
        #print('Post process put, used time {} ms. '.format((time.time()-s)*1000))



if __name__ == '__main__':
	
    #log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG)
    
#    rknn = load_model()

    q_frame = Queue(maxsize=1)
    q_image = Queue(maxsize=3)
    q_infer = Queue(maxsize=3)
    q_objs = Queue(maxsize=3)
    flag = multiprocessing.Value("d", 0)

    p_cap1 = Process(target=video_capture, args=(q_frame, q_image, flag, config.CAM_DEV))
    #p_cap2 = Process(target=video_capture, args=(q_frame, q_image, flag, config.CAM_DEV2))
    p_infer1 = Process(target=infer_rknn, args=(q_image, q_infer, flag))
    p_infer2 = Process(target=infer_rknn, args=(q_image, q_infer, flag))
    p_infer3 = Process(target=infer_rknn, args=(q_image, q_infer, flag))
    p_post1 = Process(target=post_process, args=(q_infer, q_objs, flag))
    p_post2 = Process(target=post_process, args=(q_infer, q_objs, flag))
    p_post3 = Process(target=post_process, args=(q_infer, q_objs, flag))


    p_cap1.start()
    #p_cap2.start()
    p_infer1.start()
    p_infer2.start()
    p_infer3.start()
    p_post1.start()
    p_post2.start()
    p_post3.start()

    fps = 0
    l_used_time = []

    try:
        while True:
            s = time.time()
#            frame = q_frame.get()
            frame, ratio, (dw, dh) = q_frame.get()
            boxes, classes, scores = q_objs.get()
            #print('main func, get objs use {} ms. '.format((time.time() - s) * 1000))

            if boxes is not None:
#                draw(frame, boxes, scores, classes)
                draw(frame, boxes, scores, classes, ratio, dw, dh)
            cv2.putText(frame, text='FPS: {}'.format(fps), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.imshow("results", frame)

            c = cv2.waitKey(5) & 0xff
            if c == 27:
                flag.value = 10
                time.sleep(5)
                cv2.destroyAllWindows()
                print("ESC, exit main!")
                break

            used_time = time.time() - s
            l_used_time.append(used_time)
            if len(l_used_time) > 20:
                l_used_time.pop(0)
            fps = int(1/np.mean(l_used_time))
            #print('main func, used time {} ms. '.format(used_time*1000))
    except KeyboardInterrupt:
        time.sleep(5)
        cv2.destroyAllWindows()
        print("ctrl + c, exit main!")

    p_cap1.terminate()
    #p_cap2.terminate()
    p_infer1.terminate()
    p_infer2.terminate()
    p_infer3.terminate()
    p_post1.terminate()
    p_post2.terminate()
    p_post3.terminate()
    sys.exit()




