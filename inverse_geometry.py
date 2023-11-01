#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement, collision
from pinocchio import Quaternion, SE3

import scipy.optimize as optim
import time

def computeqgrasppose(robot : pin.RobotWrapper, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    print(getcubeplacement(cube, LEFT_HOOK))
    print(getcubeplacement(cube, RIGHT_HOOK))

    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    def callback(q):
        if viz:
            viz.display(q)

    def cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q)

        def to_p_quat(oMf : SE3):
            return oMf.translation, Quaternion(oMf.rotation)

        lhand_p, lhand_quat = to_p_quat(robot.data.oMf[left_id])
        lhook_p, lhook_quat = to_p_quat(getcubeplacement(cube, LEFT_HOOK))

        rhand_p, rhand_quat = to_p_quat(robot.data.oMf[right_id])
        rhook_p, rhook_quat = to_p_quat(getcubeplacement(cube, RIGHT_HOOK))

        cost = (norm(lhand_p - lhook_p) ** 2) + (norm(rhand_p - rhook_p) ** 2)
        cost += lhand_quat.angularDistance(lhook_quat) + rhand_quat.angularDistance(rhook_quat)

        
        cost += collision(robot, q) * 100

        return cost
    
    q_sol = optim.minimize(cost, qcurrent, callback=callback)

    return q_sol.x, q_sol.success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    print(successinit, successend)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
