import cv2
import numpy as np
import pyrealsense2 as rs
from threading import Thread
import time

from yolov5.utils.augmentations import letterbox



class LoadRSStream:
    # Intel realsense streamloader for yolov5, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, source=2, img_size=640, stride=32, fps=30, auto=True, pipeline = None):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.fps = fps
        self.auto = auto
        #source = 0
        # initialize realsense pipeline

        self.sources = [source]
        # Start thread to read frames from video stream

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        self.color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        self.thread = Thread(target=self.update, args=(pipeline,), daemon=True)
        self.thread.start()

    def update(self, pipeline):
        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                self.color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                time.sleep(1 / self.fps)  # wait time
        except Exception as e: print(e)
        finally:
            # Stop streaming
            pipeline.stop()


    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = [self.color_image.copy()]
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in img0]
        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB???, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

if __name__ =="__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline.start(config)

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))