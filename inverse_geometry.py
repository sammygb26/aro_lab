#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from pinocchio.utils import rotate
from numpy.linalg import pinv, inv, norm, svd, eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement, collision, jointlimitsviolated, projecttojointlimits
from pinocchio import Quaternion, SE3

from scipy.optimize import fmin_bfgs
from util import *
import time


#def check_cube_collision(robot, cube, cubeplacement, viz):
#    setcubeplacement(robot, cube, cubeplacement)
#    col = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
#    if viz != None:
#        log_cube(cubeplacement, not col, viz) 
    

def computeqgrasppose(robot: pin.RobotWrapper, qcurrent, cube, cubetarget, viz=None):
    """Return a collision free configuration grasping a cube at a specific location and a success flag"""
    setcubeplacement(robot, cube, cubetarget)

    if pin.computeCollisions(cube.collision_model, cube.collision_data, False):
        return qcurrent, False

    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    def callback(q):
        if viz:
            viz.display(q)

    q = qcurrent.copy()
    dt = 0.1 

    vq = np.array([10.0])

    count = 0
    eff_back_off = 0.0001
    while norm(vq) > 1e-2 and count < 1000:
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        def offset(oMf, x, y, z):
            oMf.translation = (oMf @ np.array([x, y, z, 1.0]))[:3]

        oM_lh = robot.data.oMf[left_id]
        oM_lc = getcubeplacement(cube, LEFT_HOOK)
        offset(oM_lc, 0.0, eff_back_off, 0.0)

        oM_rh = robot.data.oMf[right_id - 1]
        oM_rc = getcubeplacement(cube, RIGHT_HOOK)
        offset(oM_rc, 0.0, eff_back_off, 0.0)

        oM_rc = oM_rh @ inv(robot.data.oMf[right_id]) @ oM_rc

        l_nu = pin.log(oM_lc) - pin.log(oM_lh)
        r_nu = pin.log(oM_rc) - pin.log(oM_rh) 

        o_Jleft = pin.computeFrameJacobian(robot.model, robot.data, q, left_id)
        o_Jright = pin.computeFrameJacobian(robot.model, robot.data, q, right_id)

        vq = pinv(o_Jright) @ r_nu
        Pr = np.eye(robot.nv) - (pinv(o_Jright) @ o_Jright)
        vq += pinv(o_Jleft @ Pr) @ (l_nu - o_Jleft @ vq)

        q = projecttojointlimits(robot, pin.integrate(robot.model, q, vq * dt))

        if count % 1 == 0:
            callback(q)
        count += 1

    q_sol = q

    cost = norm(l_nu) + norm(r_nu)

    print("Collision: ", collision(robot, q_sol))
    print("Joints: ", jointlimitsviolated(robot, q_sol))
    print("Cost: ", cost)

    valid_config = (
        1e-1 >= cost and 
        not collision(robot, q_sol) 
        and not jointlimitsviolated(robot, q_sol))

    return q_sol, valid_config

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    print(successinit, successend)

    updatevisuals(viz, robot, cube, q0)
