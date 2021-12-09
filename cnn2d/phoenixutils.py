import numpy as np
import pyrealsense2 as rs
from copy import deepcopy

def middle_point(pts): # xyxy or xy,xy
    if len(pts) == 2:
        #print(pts)
        return [(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2]
    if len(pts) == 4:
        return [(pts[0] + pts[2]) / 2, (pts[1] + pts[3]) / 2]

def pixel_to_3d(pixel, pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depthmap_dim = np.asanyarray(depth_frame.get_data()).shape
    #print('depthmap dim', depthmap_dim)

    colormap_dim = np.asanyarray(color_frame.get_data()).shape
    #print('colormap dim', colormap_dim)
    depth_pixel = [pixel[0] * depthmap_dim[1], pixel[1] * depthmap_dim[0]]
    #print('depth_pixel', depth_pixel)
    depth_value = np.asanyarray(depth_frame.get_data())[int(depth_pixel[1])][int(depth_pixel[0])]
    #print('depth_value at depth_pixel', depth_value)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
    return depth_point

def plane_on_3pts(pts):
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a,b,c,d

def valid_pred(storage):
    pred = deepcopy(storage)
    CLS_0 = []#np.where(pred.classes == 0)
    CLS_1 = []#p.where(pred.classes == 1)
    BX_0 = []#pred.boxes[CLS_0]
    BX_1 = []#pred.boxes[CLS_1]
    CONF_0 = []
    CONF_1 = []
    i = 0
    for box, cls, conf in zip(pred.boxes, pred.classes,
                              pred.confs):  # box is defined by two pts
        print(box,cls,conf)
        if int(cls) == 1:
            CLS_1.append(i)
            BX_1.append(box)
            CONF_1.append(conf)
        if int(cls) == 0:
            CLS_0.append(i)
            BX_0.append(box)
            CONF_0.append(conf)
        i+=1
    print(CLS_0, CLS_1)
    if len(CLS_0) == 0 or len(CLS_1) == 0:
        return False

    if BX_0[0][1] < BX_1[0][1]:
        tempbox = storage.boxes[0]
        tempcls = storage.classes[0]
        tempconf = storage.confs[0]
        storage.boxes[0] = storage.boxes[1]
        storage.cls[0] = storage.classes[1]
        storage.confs[0] = storage.confs[1]
        storage.boxes[1] = tempbox
        storage.cls[1] = tempcls
        storage.conf[1] = tempconf
        print('Changed order')
        return True
    else:
        return True

def valid_pred_full(storage):
    pred = deepcopy(storage)
    CLS_0 = np.where(pred.classes == 0)
    CLS_1 = np.where(pred.classes == 1)
    BX_0 = np.asarray(pred.boxes)[CLS_0]
    BX_1 = np.asarray(pred.boxes)[CLS_1]

    mean_y_cls_0 = np.sum(BX_0[:, 0, 1])/len(BX_0)
    mean_y_cls_1 = np.sum(BX_1[:, 0, 1])/len(BX_1)

    valid0 = []
    valid1 = []
    i = 0
    for box, cls, conf in zip(pred.boxes, pred.classes,
                              pred.confs):  # box is defined by two pts

        if cls == 0:
            if box[0][1] < mean_y_cls_1:
                i+=1
                continue
            else:
                valid0.append(i)
                i += 1
        elif cls == 1:
            if box[0][1] > mean_y_cls_0:
                i += 1
                continue
            else:
                valid1.append(i)
                i += 1

    CNF_0_valid_max_id = np.argmax(pred.confs[CLS_0][valid0])
    CNF_1_valid_max_id = np.argmax(pred.confs[CLS_1][valid1])

    BOX_0_valid_maxconf = pred.boxes[valid0][CNF_0_valid_max_id]
    BOX_1_valid_maxconf = pred.boxes[valid1][CNF_1_valid_max_id]

    return_ids = [np.where(pred.boxes==BOX_0_valid_maxconf),np.where(pred.boxes==BOX_1_valid_maxconf)]

    print (return_ids)

    return







