"""
Computer vision demonstration: 3D bounding box, 2D BB, depth colormap
pose to see [-3.9696598688708704, -0.9562233130084437, 2.074517250061035, -4.197008434926168, -0.7210381666766565, -0.049352471028463185]
"""


import pyrealsense2 as rs
import numpy as np
import urx
import cv2
import copy
from time import time, sleep
import json

robot = urx.Robot("192.168.88.156", use_rt=True)

print(robot.getj())
mypoint = [-0.5642102400409144, -1.51782733598818, -2.3565333525287073, -2.3870723883258265, 4.127176761627197, -3.114164415990011]
#robot.movej(mypoint, acc=0.1, vel=0.1)
pipeline = rs.pipeline()
config = rs.config()
pipeline.start(config)

# get camera intrinsic matrix
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

w, h = color_intrinsics.width, color_intrinsics.height

fx = color_intrinsics.fx
fy = color_intrinsics.fy
u0 = color_intrinsics.ppx
v0 = color_intrinsics.ppy

K = np.asarray([[fx, 0, u0, 0],     # intrinsic_matrix
                [0, fy, v0, 0],
                [0,  0,  1, 0]])

# set point of interest
point_of_interest = [0.712, -0.315, 0.4, 1]

# draw boundig box around point in x y z directions
bbox_side_x = 0.06
bbox_side_y = 0.06
bbox_side_z = 0.06

bounding_box_pts = []
is_bounding_box_front = []

for zz in [-1, 1]:
    for yy in [-1, 1]:
        for xx in [-1, 1]:
            if xx == -1:
                is_bounding_box_front.append(True)
            else:
                is_bounding_box_front.append(False)
            bounding_box_pts.append(point_of_interest + np.asarray([xx*bbox_side_x, yy*bbox_side_y, zz*bbox_side_z,0]))

template = cv2.imread("data/template.jpg", 0)

def getuv(point):
    R = np.asarray(robot.get_pose().orient.array)
    T = np.asarray(robot.get_pose().pos.array)
    E = np.hstack((R, np.atleast_2d(T).T))
    E = np.vstack((E, [0, 0, 0, 1]))
    good_coords = np.dot(np.linalg.inv(R), point[:-1]-T)

    point_in_pixel_coosys = np.dot(np.dot(K, np.linalg.inv(E)), point)  #point_in_pixel_coosys = np.dot(np.dot(K, np.linalg.inv(E)), point)

    u = int(point_in_pixel_coosys[0]/point_in_pixel_coosys[2])
    v = int(point_in_pixel_coosys[1]/point_in_pixel_coosys[2])

    return u, v

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

# класс, который очень мне понадобится
# постараюсь сразу описать его хорошо
class BoundingBox:
    def __init__(self, points, is_front):
        self.points = points
        self.is_front = is_front
        self.coords2d = self.get2dcoords()

    def get2dcoords(self):
        max = [0,0]
        min = [0,0]
        ii = np.where(np.asarray(self.is_front) == True)[0].astype(int)
        #print(ii)
        t = np.asarray(self.points)[ii].T
        #ymin = t[1].min()
        #ymax = t[1].max()
        #xmin = t[0].min()
        #xmax = t[0].max()
        #print(xmin,xmax,ymin,ymax)
        #cv2.circle(color_image, (int(xmin), int(ymin)), 2, (0, 255, 255), 3)
        #cv2.circle(color_image, (xmax, ymin), 2, (0, 255, 255), 3)
        #cv2.circle(color_image, (xmin, ymax), 2, (0, 255, 255), 3)
        #cv2.circle(color_image, (xmax, ymax), 2, (0, 255, 255), 3)
        p1 = [int(t[0].min()), int(t[1].min())]
        p2 = [int(t[0].max()), int(t[1].min())]
        p3 = [int(t[0].min()), int(t[1].max())]
        p4 = [int(t[0].max()), int(t[1].max())]

        """
        for point in self.points[self.is_front.index(True)]:
            if point[0] < min[0]: min[0] = point[0]
            if point[0] > max[0]: max[0] = point[0]
            if point[1] < min[1]: min[1] = point[1]
            if point[1] > max[1]: max[1] = point[1]
        """
        return p1, p2, p3, p4

    def draw_boundbox2d(self, image):
        cv2.line(image, self.coords2d[0], self.coords2d[1], (0, 255, 255), 2)
        cv2.line(image, self.coords2d[1], self.coords2d[3], (0, 255, 255), 2)
        cv2.line(image, self.coords2d[3], self.coords2d[2], (0, 255, 255), 2)
        cv2.line(image, self.coords2d[2], self.coords2d[0], (0, 255, 255), 2)
        #print(self.coords2d,"pss")


def draw_boundbox3d(points):
    cv2.line(color_image, [points[0][0],points[0][1]], [points[1][0],points[1][1]], (0, 255, 255), 2)   #1
    cv2.line(color_image, points[0], points[2], (0, 255, 255), 2)
    cv2.line(color_image, points[0], points[4], (0, 255, 255), 2)
    cv2.line(color_image, points[1], points[5], (0, 255, 255), 2)
    cv2.line(color_image, points[1], points[3], (0, 255, 255), 2)
    cv2.line(color_image, points[2], points[3], (0, 255, 255), 2)   #6
    cv2.line(color_image, points[2], points[6], (0, 255, 255), 2)
    cv2.line(color_image, points[3], points[7], (0, 255, 255), 2)
    cv2.line(color_image, points[4], points[6], (0, 255, 255), 2)
    cv2.line(color_image, points[4], points[5], (0, 255, 255), 2)
    cv2.line(color_image, points[5], points[7], (0, 255, 255), 2)
    cv2.line(color_image, points[7], points[6], (0, 255, 255), 2)


try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_TURBO)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_copy = copy.deepcopy(color_image)
        # draw bounding box on the frame
        new_x, new_y = getuv(point_of_interest)
        draw_pts = []
        for point in bounding_box_pts:
            draw_pts.append(getuv(point))
        obj = BoundingBox(draw_pts, is_bounding_box_front)
        draw_boundbox3d(draw_pts)

        cv2.circle(color_image, (int(new_x), int(new_y)), 2, (0, 255, 255), 3)

        # apply template matching
        #gray = cv2.cvtColor(color_copy, cv2.COLOR_BGR2GRAY)
        #res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        #imager = res.astype(np.float32)  # convert to float
        #imager -= imager.min()  # ensure the minimal value is 0.0
        #imager /= imager.max()  # maximum value in image is now 1.0
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #top_left = max_loc
        #cv2.circle(imager, top_left, 10, (255, 255, 255), 2)
        #bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv2.rectangle(color_image, top_left, bottom_right, 255, 2)
        obj.draw_boundbox2d(color_copy)
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            resized_color_copy = cv2.resize(color_copy, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image,resized_color_copy, depth_colormap))
        else:
            images = np.hstack((color_image, color_copy, depth_colormap))

        cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow('Demo', 640*3, 480)
        cv2.imshow('Demo', images)
        #imager = cv2.applyColorMap(cv2.convertScaleAbs(imager, alpha=0.03), cv2.COLORMAP_TURBO)
        # Show images
        #cv2.namedWindow('Template', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Template', 640, 480)
        #cv2.imshow('Template', imager)

        #cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)

        #cv2.resizeWindow('RealSense', 640, 480)
        #cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
