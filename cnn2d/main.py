"""
main module initializes yolov5,
rs pipeline,
and starts algorithm
"""

from threading import Thread
from time import sleep
import pyrealsense2 as rs
import cv2
import numpy as np
import sys
import torch
from pathlib import Path
import algorithm
from robot import getrobot, set_init_ori
from phoenixutils import valid_pred
import detect
#sys.path.append('yolov5')

robot = getrobot()
set_init_ori(robot)

# initialize object for storing color and depth frames from cameras
storage = detect.ProcessPrediction([416,736]) #should be set automatically

# configure realsense pipeline
pipeline = rs.pipeline()
config = rs.config()
pipeline.start(config)
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The script requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
#else:
#    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# initialization done

# configure yolov5 inference function
opt = vars(detect.parse_opt())
# set arguments
opt['weights'] = 'yolov5weights/type2_1.2.pt'  # model.pt path(s)
opt['source'] = 'realsense'  # ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
opt['imgsz'] = 800  # inference size (pixels)
opt['conf_thres'] = 0.25  # confidence threshold
opt['iou_thres'] = 0.1  # NMS IOU threshold
opt['max_det'] = 2  # maximum detections per image
opt['view_img'] = True
opt['store_prediction'] = storage # False if not to store prediction, ProcessPrediction object if you want to store it
opt['realsense_pipeline'] = pipeline

# run inference on the separated (background) thread
detection_thread = Thread(target=detect.run, kwargs=opt)
detection_thread.start()

# configure and start algorithm
algorithm = algorithm.Algorithm(pipeline, storage, robot)
algorithm.run()


# visualization MIGHT be unused
def visualizeimg(img_source):
    while True:
        cv2.namedWindow("PhoenixBot", cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow("PhoenixBot", 800, 600)
        print(img_source.anoimg)
        cv2.imshow("PhoenixBot", img_source.anoimg)
        cv2.waitKey(1000)  # 1 milliseond
visualization_thread = Thread(target=visualizeimg, args=(storage,))

"""
def midpoint_of_line(pts):
    return [(pts[0][0]+pts[0][1])/2, (pts[1][0]+pts[1][1])/2]

while True:
    frames = pipeline.wait_for_frames()
    sleep(1)
    if not frames:
        print("noframe")
        continue
    if len(storage.boxes):
        for box,cls,conf in zip(storage.boxes, storage.classes, storage.confs):

            if int(cls) == 1:
                # find coordinates of pixel in depth map
                depth_pixel = box[0]
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depthmap_dim = np.asanyarray(depth_frame.get_data()).shape

                colormap_dim = np.asanyarray(color_frame.get_data()).shape

                color_pixel = [depth_pixel[0]*colormap_dim[1],depth_pixel[1]*colormap_dim[0]]
                depth_pixel = [depth_pixel[0] *depthmap_dim[1], depth_pixel[1] *depthmap_dim[0]]

                depth_value = np.asanyarray(depth_frame.get_data())[int(depth_pixel[1])][int(depth_pixel[0])]
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
                storage.circle_coords = (int(color_pixel[0]),int(color_pixel[1]))
                print(depth_point)
        #print(storage.anoimg.shape)
        #print(check_imshow())
    #print("here")
    #cv2.namedWindow("PhoenixBot", cv2.WINDOW_AUTOSIZE)
    #print("there")
    #cv2.resizeWindow("PhoenixBot", 800, 600)
    #print(storage.anoimg)
    #cv2.imshow("PhoenixBot", storage.anoimg)
    #cv2.waitKey(10)# 1 milliseond
"""

"""
    storage.process()
    classes = []
    midpoint_classes = []
    for box in storage.boxes:
        classes.append(int(box.cls))
        midpoint_classes.append(box.midpoint())
        #print(storage.normalize(box.midpoint()))
    if len(classes)!=2:
        print('Not enough classes detected')
        continue
    if classes[0] == classes[1]:
        print ('Detected two boxes of one class')
        continue
    else:
        depth_pixel = midpoint_of_line(midpoint_classes)
        #print(depth_pixel)
        depth_frame = frames.get_depth_frame()
        #print(np.asanyarray(depth_frame.get_data()).shape)
        depth_value = np.asanyarray(depth_frame.get_data())[int(depth_pixel[1])][int(depth_pixel[0])]
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # = 0.5
        #depth_pixel = [depth_intrin.ppx, depth_intrin.ppy]
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
        #print(depth_point)
"""

"""
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

except Exception as e:
    print(e)
finally:
    # Stop streaming
    pipeline.stop()
    
"""