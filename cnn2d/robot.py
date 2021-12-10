import urx
import numpy as np
import math3d as m3d
from time import sleep


def base_to_tool():
    pass

def tool_to_base():
    pass

def getrobot():
    robot = urx.Robot("192.168.88.156", use_rt=True)
    print(robot.getj())

    return robot

def set_init_ori(robot):
    ori = [[0.0, 0.0, 1.0],
           [-1.0, 0.0, 0.0],
           [0.0, -1.0, 0.0]]
    robot.set_orientation(ori,vel=0.1)

if __name__ == "__main__":
    robot = getrobot()
    print(robot.getl())
"""
#print(dir(robot))

#robot.set_orientation((0, 0, 1), acc=0.01, vel=0.01)


vecZ = np.asarray([0, 0, -1])

vec1 = np.array([1,0,-0.5])
vec2 = np.cross(vecZ, vec1)
vec3 = np.cross(vec1, vec2)

vec1 = vec1/np.linalg.norm(vec1)
vec2 = vec2/np.linalg.norm(vec2)
vec3 = vec3/np.linalg.norm(vec3)
print(vec1,vec2,vec3)
M = np.matrix([vec2, vec3, vec1]).T
print(M)
print(np.linalg.det(M))

ori = m3d.Orientation(np.asarray(M))
robot.set_orientation(ori, acc=0.01, vel=0.01)
print('vec1',np.concatenate((vec1,[0,0,0])))

robot.speedl(np.concatenate((vec1,[0,0,0]))/50,0.1,50)
sleep(3)
robot.stop()
"""