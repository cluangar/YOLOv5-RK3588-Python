import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite
from imutils.video import FPS
import time
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

def draw(image, boxes, scores, classes):
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

def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ("uvch264src device=/dev/video{} ! "
               "image/jpeg, width={}, height={}, framerate=30/1 ! "
               "jpegdec ! "
               "video/x-raw, format=BGR ! "
               "appsink").format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if __name__ == '__main__':

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


    #Create Stream from Webcam
    vs = open_cam_usb(config.CAM_DEV, config.CAM_WIDTH, config.CAM_HEIGHT)


    time.sleep(2.0)
    fps = FPS().start()

    if not vs.isOpened():
	    print("Cannot capture from camera. Exiting.")
	    quit()
	
    prev_frame_time = 0
    new_frame_time = 0

  
# loop over the frames from the video stream
    while True:

        ret, frame = vs.read()
        
        if not ret:
            break
#Debug Webcam Input
#        cv2.imshow("Original", frame)
#Show FPS in Pic
        new_frame_time = time.time()
        show_fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        show_fps = int(show_fps)
        show_fps = str("{} FPS".format(show_fps))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, ratio, (dw, dh) = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
#        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # Inference
        outputs = rknn_lite.inference(inputs=[frame])

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

#Disable Enable YOLO Post process
        boxes, classes, scores = yolov5_post_process(input_data)

#        boxes = None

        img_1 = frame
        if boxes is not None:

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


            img_1 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            draw(img_1, boxes, scores, classes)
            
            # show FPS in Frame
            cv2.putText(img_1, show_fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
            
            #rescale to equal original
#            old_h, old_w = img_1.shape[:2]
#            ratio_h, ratio_w = ratio
#            print(old_h/ratio_h, old_w/ratio_w)
#            img_1 = cv2.resize(img_1, (int(old_h/ratio_h), int(old_w/ratio_w)), interpolation=cv2.INTER_LINEAR)
#            img_1 = cv2.resize(img_1, (800, 800), interpolation=cv2.INTER_LINEAR)
            
            # show output
            cv2.imshow("yolov5 post process result", img_1)
            
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        

        # update the FPS counter
        fps.update()


    rknn_lite.release()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()

