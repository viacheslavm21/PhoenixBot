"""
This script starts streaming from the realsense camera and saves a picture every 5 sec
It also saves robot joint angles at moments of capturing pics.
"""

import pyrealsense2 as rs
import numpy as np
import urx
import cv2
from threading import Thread
from time import sleep
import json

ROBOT_IP = "192.168.88.155"

robot = urx.Robot(ROBOT_IP, use_rt=True)

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
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

def save_color():
    img_count = 369

    while True:
        sleep(5)
        filename = 'data/dataset2a/frame'+str(img_count)
        try:
            cv2.imwrite(filename+'.jpg', color_image)
            j = {"j":robot.getj()}
            with open(filename+'.json', 'w') as outfile:
                json.dump(j, outfile)
            img_count += 1
            print(filename+' saved')
        except Exception as e:
            print(e)

saver = Thread(target=save_color)
saver.start()

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

        # Show images
        cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow('Demo', 640, 480)
        cv2.imshow('Demo', color_image)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
