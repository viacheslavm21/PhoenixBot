"""
This file is the implementation of following algorithm:
1. Obtain frame from camera.
2. Rotate camera so that socket is in the center of the image.
3. Move towards the socket until the average depth of pixels around socket is 35cm.
4. De-project several basepoints (points at socket surface) from pixel to camera coordinate system.
5. Build a normal to the plane defined by extracted points
6. Align TCP along the normal
7. Based on relative position of the socket parts (class 1 and class 2) calculate its orientation around z-axis.
"""
import numpy as np
import pyrealsense2 as rs
import math3d as m3d
import time
from time import sleep

from phoenixutils import pixel_to_3d, middle_point, plane_on_3pts, valid_pred

class Algorithm():
    def __init__(self, rspipeline, storage, robot):
        self.stage = 'DemoPaper'
        self.pipeline = rspipeline
        self.storage = storage
        self.robot = robot
        self.init_poss = [[-4.453077975903646, -1.8218439261065882, 2.3542661666870117, -2.765601460133688, -0.4438403288470667, -0.6408065001117151],
                          [-3.339505974446432, -1.3737195173846644, 2.485387086868286, -4.058104340230123, -1.593560043965475, -0.23038846651186162]]

    def preliminary_orient(self):
        print('BEGIN PRELIMINARY ORIENTING')
        print(self.robot.getj())
        while not len(self.storage.boxes):
            sleep(1)

        for box, cls, conf in zip(self.storage.boxes, self.storage.classes, self.storage.confs): # box is defined by two pts
            if int(cls) == 1:
                # find coordinates of pixel in depth map
                centerpixel = middle_point(box)
                print('Pixel of center', centerpixel)
                self.storage.circle_coords = (int(centerpixel[0]*self.storage.anoimg.shape[1]),
                                              int(centerpixel[1]*self.storage.anoimg.shape[0]))
                print("self.storage.circle_coords",self.storage.circle_coords)
                C = np.asarray(pixel_to_3d(centerpixel, self.pipeline))/1000
                print ('C in tool',C)
                print(np.asarray(self.robot.get_pose().array))
                C = np.asarray(self.robot.get_pose().array)@np.concatenate((C,[1]))
                print ('C in base', C)
        T = np.asarray(self.robot.get_pose().pos.array)
        orivec = C[:-1] - T

        print('Vector Z of orientation', orivec)

        vecZ = np.asarray([0, 0, -1])

        vec1 = np.array(orivec)
        vec2 = np.cross(vecZ, vec1)
        vec3 = np.cross(vec1, vec2)

        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = vec3 / np.linalg.norm(vec3)
        M = np.matrix([vec2, vec3, vec1]).T
        print('Orientation matrix/n')
        print(M)
        print('Orientation matrix determinant', np.linalg.det(M))

        while True:
            print('Type (y) to start robot movement ')
            a = input()
            if a == 'y':
                ori = m3d.Orientation(np.asarray(M))
                self.robot.set_orientation(ori, acc=0.01, vel=0.01)
                sleep(1)
                print('Successfully executed stage one.')
                return True, vec1
                #SUCCESS = self.preliminary_approach(vec1)

            else:
                print('(y) is not send. Exit the script.')
                return False, None

    def preliminary_approach(self, vec1):   # vec1 = vector of approach
        self.robot.speedl(np.concatenate((vec1, [0, 0, 0])) / 50, 0.1, 50)
        t = time.time()
        tx = time.time()
        BREAK = False
        while True:
            if tx - t > 100 or BREAK:
                self.robot.stop()
                print('Too long motion. Exit the script.')
                return False
            for box, cls, conf in zip(self.storage.boxes, self.storage.classes,
                                      self.storage.confs):  # box is defined by two pts
                loss_count = 0
                if int(cls) == 1:
                    # find coordinates of pixel in depth map
                    centerpixel = middle_point(box)
                    self.storage.circle_coords = (int(centerpixel[0]) * 640, int(centerpixel[1]) * 480)
                    C = np.asarray(pixel_to_3d(centerpixel, self.pipeline)) / 1000
                    print('C in tool', C)
                    print('distance to point', np.linalg.norm(C[:-1]))
                    if abs(C[2]) > 1E-4:
                        loss_count+=1
                    if loss_count > 50:
                        print('Object lost. Exit the script.')
                        self.robot.stop()
                        return False

                    if C[2] < 0.43 and abs(C[2]) > 1E-4:
                        self.robot.stop()
                        print('Successful execution of stage two.')
                        return True

    def iter_speedl(self, movvec):
        self.robot.speedl(movvec, 0.005, 60)

        prev_err = 1E6
        while True:
            ymaxbox = np.argmin(np.asarray(self.storage.boxes)[:,0,1])

            self.current_px = middle_point(self.storage.boxes[ymaxbox])
            self.storage.circle_coords = (int(self.current_px[0] * self.storage.anoimg.shape[1]),
                                          int(self.current_px[1] * self.storage.anoimg.shape[0]))

            # calculate error
            error = np.abs(np.sum(self.desired_px-self.current_px))
            #print('error', error)
            if error < self.pixel_tolerance:
                self.robot.stop()
                return True
            elif error <= prev_err:
                prev_err = error

            else:
                #print('Error starts to grow. Change direction')
                movvec = np.concatenate((self.current_px - self.desired_px, [0, 0, 0, 0]))
                movvec = movvec / abs(np.linalg.norm(movvec)) / 400
                SUCCESS = self.iter_speedl(movvec)
                if SUCCESS:
                    return True
                else:
                    continue
                    return False

    def move2d(self):
        print('BEGIN SECONDARY ORIENTING')
        while not len(self.storage.boxes):
            sleep(1)

        self.desired_px = np.asarray([0.5, 0.4])

        #valid_pred(self.storage)

        ymaxbox = np.argmin(np.asarray(self.storage.boxes)[:, 0, 1])

        self.current_px = middle_point(self.storage.boxes[ymaxbox])
        self.storage.circle_coords = (int(self.current_px[0] * self.storage.anoimg.shape[1]),
                                      int(self.current_px[1] * self.storage.anoimg.shape[0]))
        sleep(2)
        movvec2d = self.current_px-self.desired_px
        movvec2d = [movvec2d[0]*self.storage.anoimg.shape[1], movvec2d[1]*self.storage.anoimg.shape[0]] # надо умножить на разрешение кадра цвета АВТОМАТИЧЕСКИ!!!

        print('Vector between pixels ', movvec2d)
        movvec = np.concatenate((movvec2d,[0,0,0,0]))
        movvec = movvec/abs(np.linalg.norm(movvec))/400
        self.base_coord_sys = self.robot.csys
        print('Moving tool with ', movvec)
        self.robot.set_csys(self.robot.get_pose())
        self.pixel_tolerance = 0.008
        #print('Type (y) to start robot movement ')
        #a = input()
        #if a == 'y':
        SUCCESS = self.iter_speedl(movvec)

        return SUCCESS

    def setx(self):
        print('Type X coordinate of TCP in mm:')
        while True:
            try:
                x = int(input())/1000
                break
            except:
                print('Not integer')
                continue

        current_pose = self.robot.get_pose().pos.array
        x_to_move = x - current_pose[0]
        self.robot.movel((x_to_move, 0, 0, 0, 0, 0), 0.15, 0.15, relative=True)
        self.rxc = x
        return True

    def demo_approach(self):
        if abs(self.rxc - 0.7) < 1E-6:
            self.robot.movel((0.041, -0.11, 0.255, 0, 0, 0.05), 0.05, 0.05, relative=True)

    def run(self):
        if self.stage == 'Test':
            #self.iter_speedl_test()
            return
        if self.stage=='DemoPaper':
            STAGE_SET_X = self.setx()
            if STAGE_SET_X:
                STAGE_THREE = self.move2d()
                print('Center aligned')
            else:
                print ('Sth went wrong. Exit')
                return
            if STAGE_THREE:
                STAGE_FOUR = self.demo_approach()
                print('Approached')
            else:
                print ('Sth went wrong. Exit')
                return
            print('Done!')
            return
        if self.stage != 'Idle':
            print ("Already running or last stop was corrupted")
            return
        else:
            print("Algorithm started. Type (i) to move robot to i-th initial position")
            y = input()
            try:
                pos = int(y)
                if pos >= len(self.init_poss):
                    print('Pose not found. Skip initial movement')
                else:
                    self.robot.movej(self.init_poss[pos])
            except:
                print('Not a number. Skip initial movement')

            STAGE_ONE, move_vec = self.preliminary_orient()
            if STAGE_ONE:
                STAGE_TWO = self.preliminary_approach(move_vec)
            else:
                print('Not successful exit from stage one')
                return
            if STAGE_TWO:
                STAGE_THREE = self.move2d()
            else:
                print('Not successful exit from stage two.')
                return
            if STAGE_THREE:
                print("Good!")


"""
    def orient_along_normal_to_socket_plane(self):
        # STEP 1 (it should be changed to rotation due to possible collisions)
        # move in tool frame along XY plane to put socket in a good position on the map

        
        desired_pixel = np.asarray([0.5,0.2])
        if not len(self.storage.boxes):
            print('Not seen higher part of the socket.')
            return False
        for box, cls, conf in zip(self.storage.boxes, self.storage.classes, self.storage.confs): # box is defined by two pts
            if int(cls) == 1:
                # find coordinates of pixel in depth map
                current_pixel = middle_point(box)
        motion_dir = np.concatenate((desired_pixel-current_pixel,[0,1]))
        print('direction of motion in tool', motion_dir)
        motion_dir = np.linalg.inv(np.asarray(self.robot.get_pose().array)) @ motion_dir # in base frame
        print('direction of motion in base', motion_dir)
        self.robot.speedl(np.concatenate((motion_dir[:-1], [0, 0, 0])) / 50, 0.1, 50)
        t = time.time()
        tx = time.time()
        BREAK = False
        while True:
            if tx - t > 100 or BREAK:
                self.robot.stop()
                break
            for box, cls, conf in zip(self.storage.boxes, self.storage.classes,
                                      self.storage.confs):  # box is defined by two pts
                if int(cls) == 1:
                    # find coordinates of pixel in depth map
                    current_pixel = middle_point(box)
                    self.storage.circle_coords = (int(current_pixel[0]) * 1280, int(current_pixel[1]) * 720)
                    print('current pixel', current_pixel)
                    print('desired pixel', desired_pixel)
                    if abs(np.sum(current_pixel-desired_pixel)) < 1E-3:
                        print('sum', abs(np.sum(current_pixel-desired_pixel)))
                        self.robot.stop()
                        BREAK = True
                        break
        
        # STEP 1 -variant2
        while True:
            ONE = False
            ZERO = False
            if not len(self.storage.boxes):
                print('Not seen the socket. Trying to reorient.')

            for box, cls, conf in zip(self.storage.boxes, self.storage.classes,
                                      self.storage.confs):  # box is defined by two pts
                if int(cls) == 1:
                    # find coordinates of pixel in depth map
                    ONE = True
                if int(cls)  == 0:
                    ZERO = True
            if ZERO and ONE:
                break
            else:
                print('Only one socket part is seen. Trying to reorient.')
                STAGE_ONE, move_vec = self.preliminary_orient()

"""

"""
def test_ori(self):
    while True:
        # extract 6 points
        pts = []
        for box, cls, conf in zip(self.storage.boxes, self.storage.classes,
                                  self.storage.confs):  # box is defined by two pts
            if int(cls) == 1:
                pts.append([box[0][0], box[0][1]])
                pts.append([box[0][0], box[1][1]])
                pts.append([box[1][0], box[0][1]])
                pts.append([box[1][0], box[1][1]])
            if int(cls) == 0:
                pts.append([box[0][0], box[0][1]])
                pts.append([box[0][0], box[1][1]])
                pts.append([box[1][0], box[0][1]])
                pts.append([box[1][0], box[1][1]])
            # get 3d of points
        pts3d = []
        for pt in pts:
            C = np.asarray(pixel_to_3d(pt, self.pipeline)) / 1000
            pts3d.append(C)
        combinations = [[pts3d[1], pts3d[2], pts3d[0]]]

        normals = []
        for combination_pts in combinations:
            a, b, c, d = plane_on_3pts(combination_pts)
            normals.append([a, b, c])
        #print(normals)
        #print(np.sum(normals, axis=0) / len(combinations))
        print(combinations)

def orient_along_normal_to_socket_plane(self):
    print ('Starting stage three')
    # extract 6 points
    pts = []
    for box, cls, conf in zip(self.storage.boxes, self.storage.classes,
                              self.storage.confs):  # box is defined by two pts
        if int(cls) == 1:
            pts.append([box[0][0], box[0][1]])
            pts.append([box[0][0], box[1][1]])
            pts.append([box[1][0], box[0][1]])
            pts.append([box[1][0], box[1][1]])
        if int(cls) == 0:
            pts.append([box[0][0], box[0][1]])
            pts.append([box[0][0], box[1][1]])
            pts.append([box[1][0], box[0][1]])
            pts.append([box[1][0], box[1][1]])
        # get 3d of points
    pts3d = []
    for pt in pts:
        C = np.asarray(pixel_to_3d(pt, self.pipeline))/1000
        print('point in tool', C)

        C = np.asarray(self.robot.get_pose().array)@np.concatenate((C,[1]))
        print('point in base', C)
        pts3d.append(C[:-1])


    print(len(pts3d))
    print('Extracted points are:')
    print(pts3d)
    combinations = [[pts3d[1],pts3d[2],pts3d[0]],
                    [pts3d[5],pts3d[6],pts3d[2]],
                    [pts3d[0],pts3d[5],pts3d[3]],
                    [pts3d[5],pts3d[6],pts3d[0]],
                    [pts3d[7],pts3d[0],pts3d[5]],
                    [pts3d[4],pts3d[7],pts3d[2]]]

    normals = []
    for combination_pts in combinations:
       a,b,c,d = plane_on_3pts(combination_pts)
       normals.append([a,b,c])
    print(normals)
    print(np.sum(normals,axis=0)/len(combinations))
    print(normals[0])
    #normal = normals[1]
    while True:
        print('Type the normal to align along (0,1,2...)')
        i = int(input())
        normal = normals[i]
        print('normal', normal)
        # align TCP along the normal
        vecZ = np.asarray([0, 0, -1])

        vec1 = normal
        print(vec1)
        vec2 = np.cross(vecZ, vec1)
        vec3 = np.cross(vec1, vec2)

        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = vec3 / np.linalg.norm(vec3)
        M = np.matrix([vec2, vec3, vec1]).T
        print('Orientation matrix/n')
        print(M)
        print('Orientation matrix determinant', np.linalg.det(M))
        ori = m3d.Orientation(np.asarray(M))
        #move robot
        self.robot.set_orientation(ori, acc=0.01, vel=0.01)

"""
