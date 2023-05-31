import threading
import cv2


class VideoCaptureThreading:
    def __init__(self, src=0, width=640, height=480):
		
        self.gst_str = ("v4l2src device=/dev/video{} io-mode=0 ! "
               "image/jpeg, width={}, height={}, framerate=30/1 ! "
               "queue ! mppjpegdec ! "
               "video/x-raw ! "
#               "videoconvert ! "
#               "video/x-raw, format=RGB ! "
#               "queue ! videoconvert ! appsink drop=1 sync=false").format(src, width, height)		
               "queue ! videoconvert ! appsink drop=1 sync=false").format(0, 640, 480)		
	
		
        self.src = src
##        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
