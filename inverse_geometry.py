#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, inv, norm, svd, eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement, collision, jointlimitscost, jointlimitsviolated
from pinocchio import Quaternion, SE3

from scipy.optimize import fmin_bfgs
import time


def computeqgrasppose(robot: pin.RobotWrapper, qcurrent, cube, cubetarget, viz=None):
    """Return a collision free configuration grasping a cube at a specific location and a success flag"""
    setcubeplacement(robot, cube, cubetarget)

    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    def callback(q):
        if viz:
            viz.display(q)

    eps = 0

    def cost(q):
        pin.framesForwardKinematics(robot.model, robot.data, q)

        def to_p_quat(oMf: SE3):
            return oMf.translation, Quaternion(oMf.rotation)

        lhand_p, lhand_quat = to_p_quat(robot.data.oMf[left_id])
        _, lhook_quat = to_p_quat(getcubeplacement(cube, LEFT_HOOK))

        rhand_p, rhand_quat = to_p_quat(robot.data.oMf[right_id])
        _, rhook_quat = to_p_quat(getcubeplacement(cube, RIGHT_HOOK))

        p_l = (getcubeplacement(cube, LEFT_HOOK) @ np.array([0.0, eps, 0.0, 1.0]))[:3]
        p_r = (getcubeplacement(cube, RIGHT_HOOK) @ np.array([0.0, eps, 0.0, 1.0]))[:3]

        cost = (norm(lhand_p - p_l) ** 2) + (norm(rhand_p - p_r) ** 2)
        cost += (
            lhand_quat.angularDistance(lhook_quat) + 
            rhand_quat.angularDistance(rhook_quat))

        cost += jointlimitscost(robot, q)

        return cost

    q_sol = fmin_bfgs(cost, qcurrent, callback=callback, disp=False, gtol=1e-5)

    print("Collision: ", collision(robot, q_sol))
    print("Joints: ", jointlimitsviolated(robot, q_sol))
    print("Cost: ", cost(q_sol))

    valid_config = (
        1e-1 >= cost(q_sol) and 
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
