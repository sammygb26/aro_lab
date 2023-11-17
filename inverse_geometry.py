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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def computeqgrasppose(robot: pin.RobotWrapper, qcurrent, cube, cubetarget, viz=None):
    """Return a collision free configuration grasping a cube at a specific location and a success flag"""
    setcubeplacement(robot, cube, cubetarget)
    q = qcurrent.copy()

    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    oM_lc = getcubeplacement(cube, LEFT_HOOK)
    oM_rc = getcubeplacement(cube, RIGHT_HOOK)

    cost = 10
    speedup = 100
    count = 0

    while cost > 0.001 and count < 250:
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

        q = projecttojointlimits(robot, q)
        cost = norm(lh_nu) + norm(rh_nu)
        if viz:
            viz.display(q)
            time.sleep(0.001)

        count += 1

    # Make sure cube is ignored in collisions

    setcubeplacement(robot, cube, cubetarget)

    valid_config = (
        not collision(robot, q) and not jointlimitsviolated(robot, q) and cost < 0.001
    )
    return q, valid_config


from scipy.spatial.transform import Rotation as R
from pinocchio.utils import rotate
import matplotlib.image as mpimg


def testInvGeom():
    x = np.linspace(0.2, 0.8, 18)
    y = np.linspace(-0.6, 0.6, 36)
    z = np.linspace(0.9, 1.5, 18)
    cube_placements = np.zeros((x.shape[0], y.shape[0], z.shape[0]), dtype=object)
    success = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
    for i in range(x.shape[0]):
        print(i)
        for j in range(y.shape[0]):
            print(j)
            for k in range(z.shape[0]):
                cube_placements[i, j, k] = pin.SE3(
                    rotate("z", 0), np.array([x[i], y[j], z[k]])
                )
                pose, flag = computeqgrasppose(
                    robot, q, cube, cube_placements[i, j, k], None
                )
                if flag:
                    success[i, j, k] = 1

    for bitmap in success:
        # Display the bitmap using matplotlib
        plt.imshow(
            bitmap, cmap="gray"
        )  # You can adjust the colormap based on your image type
        plt.title("2D Bitmap")
        plt.axis("off")  # Turn off axis labels
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = np.where(success == 1)
    ax.scatter(x, y, z, c="g", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Use transpose of `cube` to get the direction right
    # (bottom->up rather than left->right)
    ax.voxels(success.T, edgecolor="k")
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    plt.show()

def repetitions():
    count = 0
    q = q0
    for i in range(100):
        qNew, success = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT)
        if success:
            count += 1
    print(count)

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    print(successinit, successend)

    updatevisuals(viz, robot, cube, q0)
