#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, inv, norm, svd, eig
from tools import (
    collision,
    getcubeplacement,
    setcubeplacement,
    projecttojointlimits,
    jointlimitsviolated,
)
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON, DT
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, OBSTACLE_PLACEMENT

from scipy.optimize import fmin_bfgs
from util import *
import time


def computeqgrasppose(robot: pin.RobotWrapper, qcurrent, cube, cubetarget, viz=None):
    """Return a collision free configuration grasping a cube at a specific location and a success flag"""
    setcubeplacement(robot, cube, cubetarget)
    q = qcurrent.copy()

    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    oM_lc = getcubeplacement(cube, LEFT_HOOK)
    # Add a small offset to avoid singularities
    xOffset = 0
    yOffset = 0
    zOffset = 0.01
    oM_lc = pin.SE3(
        oM_lc
        + np.array(
            [[0, 0, 0, xOffset], [0, 0, 0, yOffset], [0, 0, 0, zOffset], [0, 0, 0, 0]]
        )
    )
    oM_rc = getcubeplacement(cube, RIGHT_HOOK)
    oM_rc = pin.SE3(
        oM_rc
        + np.array(
            [[0, 0, 0, xOffset], [0, 0, 0, -yOffset], [0, 0, 0, zOffset], [0, 0, 0, 0]]
        )
    )

    cost = 10
    speedup = 100
    count = 0

    while cost > 0.0001 and count < 1000:
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)

        oM_lh = robot.data.oMf[left_id]
        o_Jleft = pin.computeFrameJacobian(
            robot.model, robot.data, q, left_id, pin.LOCAL
        )

        oM_rh = robot.data.oMf[right_id]
        o_Jright = pin.computeFrameJacobian(
            robot.model, robot.data, q, right_id, pin.LOCAL
        )

        lh_nu = pin.log(oM_lh.inverse() * oM_lc).vector
        rh_nu = pin.log(oM_rh.inverse() * oM_rc).vector

        vq = pinv(o_Jright) @ rh_nu
        Pr = np.eye(robot.nv) - pinv(o_Jright) @ o_Jright
        vq += pinv(o_Jleft @ Pr) @ (lh_nu - o_Jleft @ vq)

        q = pin.integrate(robot.model, q, vq * DT * speedup)
        cost = norm(lh_nu) + norm(rh_nu)
        if viz:
            viz.display(q)
            time.sleep(0.001)

        count += 1

    # Make sure cube is ignored in collisions
    q_sol = projecttojointlimits(
        robot, pin.integrate(robot.model, q, vq * DT * speedup)
    )
    setcubeplacement(robot, cube, cubetarget)

    valid_config = (
        not collision(robot, q_sol)
        and not jointlimitsviolated(robot, q_sol)
        and cost < 0.001
    )

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
