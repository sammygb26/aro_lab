#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
from numpy.linalg import norm, inv, pinv
import pinocchio as pin
import time

from pinocchio import Quaternion, SE3
from pinocchio.utils import rotate
from bezier import Bezier
from config import DT, LEFT_HAND, RIGHT_HAND
from scipy.optimize import fmin_bfgs
from tools import distanceToObstacle
from util import log_cube


# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 1200  # proportional gain (P of PD)
Ki = 600
i = 0
Kd = 1 * np.sqrt(Kp)

CUBE_PLACEMENT_UP = pin.SE3(rotate("z", 0.0), np.array([0.33, -0.3, 1.13]))


graspForce = 100
cubeweight = 150
graspTorque = 1


def norm2(x):
    return np.sum(np.power(x, 2))


def getGraspForces(sim, robot):
    fext = [pin.Force.Zero()] * (robot.nv + 1)

    lid = (
        sim.bulletCtrlJointsInPinOrder.index(sim.bullet_names2indices["LARM_EFF"] - 1)
        + 1
    )
    rid = (
        sim.bulletCtrlJointsInPinOrder.index(sim.bullet_names2indices["RARM_EFF"] - 1)
        + 1
    )

    fext[lid] = pin.Force(np.array([-graspForce, 0, -cubeweight, 0, 0, graspTorque]))
    fext[rid] = pin.Force(np.array([-graspForce, 0, -cubeweight, 0, 0, -graspTorque]))

    fext_wrapper = pin.StdVec_Force()
    for ext_force in fext:
        fext_wrapper.append(ext_force)

    return fext_wrapper


def controllaw(sim, robot, trajs, tcurrent, cube):
    global i
    q, vq = sim.getpybulletstate()
    q_of_t, vq_of_t, vvq_of_t = trajs

    # PID
    e_q = q_of_t(tcurrent) - q
    i += e_q * DT
    e_qd = vq_of_t(tcurrent) - vq
    dvq = Kp * e_q + Kd * e_qd + i * Ki

    # Add grasp force
    graps_ext = getGraspForces(sim, robot)

    h = pin.rnea(robot.model, robot.data, q, vq, dvq, graps_ext)
    M = pin.crba(robot.model, robot.data, q)

    torques = M @ dvq + h

    sim.step(torques)


def calculate_lMr(robot, q):
    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)
    oMl = robot.data.oMf[left_id]
    oMr = robot.data.oMf[right_id]
    lMr = oMf_to_quat_trans(oMl.inverse() * oMr)

    return lMr, oMl, oMr


def compress_q(q, q_optimize):
    return np.array([a for i, a in enumerate(q) if q_optimize[i]])


def decompress_q(cq, q_optimize):
    q = q0.copy()
    i = 0
    for a in cq:
        while not q_optimize[i]:
            i += 1
        q[i] = a
        i += 1
    return np.array(q)


def dist_cost(x, r, rho):
    xr_norm = norm(x - r)
    if xr_norm > rho:
        return 0
    else:
        return 1 / (xr_norm + 1) - 1 / (rho + 1)


def calculate_potential_gradient(point, oMob, w, h, d, rho=0.2):
    point = (oMob.inverse() @ np.array([*list(point), 1]))[:3]

    closest_point = point.copy()

    if np.abs(point[0]) > w:
        closest_point[0] = w * np.sign(point[0])
    if np.abs(point[2]) > h:
        closest_point[2] = h * np.sign(point[2])
    if np.abs(point[1]) > w:
        closest_point[1] = d * np.sign(point[1])

    mag = dist_cost(np.zeros(3), point, rho)

    return oMob.rotation @ (point / norm(point)), mag


def calculate_potential_attaction(from_point, to_point):
    dif = to_point - from_point
    return dif * norm(dif)


def maketraj_cube_pot(q0, q1, max_acc, path, robot, cube, viz=None):
    n = len(path)
    cube_points = []
    for q in path:
        _, oMl, oMr = calculate_lMr(robot, q)
        cube_pos = (oMl.translation + oMr.translation) * 0.5
        cube_points.append(cube_pos)

    eps = 1e-1
    cube_points_opt = [cube_pos.copy() for cube_pos in cube_points]
    for i in range(50):
        for i in range(1, n - 1):
            cube_point = cube_points_opt[i]
            cube_point_prev = cube_points_opt[i - 1]
            cube_point_next = cube_points_opt[i + 1]
            cube_point_ref = cube_points[i]

            grad, mag = calculate_potential_gradient(
                cube_point, OBSTACLE_PLACEMENT, 0.03, 0.3, 0.12, 0.25
            )

            force = grad * mag
            force += calculate_potential_attaction(cube_point, cube_point_prev)
            # force += calculate_potential_attaction(cube_point, cube_point_next)
            force += calculate_potential_attaction(cube_point, cube_point_ref)

            cube_point += force * eps

        if viz != None:
            for i, cube_point in enumerate(cube_points_opt):
                cubetarget = SE3(rotate("z", 0), cube_point)
                log_cube(cubetarget, True, viz, f"cube{i}")

    if viz != None:
        for i in range(n):
            viz.delete(f"cube{i}")

    q_prev = q0
    path = []
    for cube_point in cube_points_opt:
        cubetarget = SE3(rotate("z", 0), cube_point)
        q, _ = computeqgrasppose(robot, q_prev, cube, cubetarget)
        path.append(q)

    return path_to_bezier([q0, q0, q0, *path, q1, q1, q1], max_acc)


def maketraj_bfgs_trap(
    q0,
    q1,
    max_acc,
    path,
    q_optimize,
    robot,
    npoints=6,
    ntest=10,
    lambda_lMr=12,
    lambda_odist=6,
    lambda_ref=1,
):
    npoints = min(npoints, len(path))
    skip = int(len(path) / ntest)
    dt = 1 / len(path)

    ref_lMr, _, _ = calculate_lMr(robot, q0)

    left_id = robot.model.getFrameId(LEFT_HAND)
    right_id = robot.model.getFrameId(RIGHT_HAND)

    p0 = np.array(
        [
            compress_q(path[int(t)], q_optimize)
            for t in np.linspace(0, len(path) - 1, npoints)
        ]
    )

    def np_to_path(p):
        p = np.reshape(p, p0.shape)
        return (
            q0,
            q0,
            q0,
            *[decompress_q(p[i, :], q_optimize) for i in range(npoints)],
            q1,
            q1,
            q1,
        )

    def np_to_traj(p):
        q_of_t = Bezier(np_to_path(p), t_max=1)
        return q_of_t

    lambda_total = lambda_lMr + lambda_odist + lambda_ref
    lambda_lMr /= lambda_total
    lambda_odist /= lambda_total
    lambda_ref /= lambda_total

    def cost(p):
        ret = 0

        q_of_t = np_to_traj(p)
        prev_loss = 0
        for i in range(0, len(path), skip):
            qt = q_of_t(i * dt)
            q_ref = path[i]

            loss = norm(qt - q_ref, ord=2) * lambda_ref

            # Actual lMr
            pin.framesForwardKinematics(robot.model, robot.data, qt)

            oMl = robot.data.oMf[left_id]
            oMr = robot.data.oMf[right_id]
            lMr_quat, lMr_trans = oMf_to_quat_trans(oMl.inverse() * oMr)

            ## Deviation added to cost
            lMr_cost = Quaternion.angularDistance(ref_lMr[0], lMr_quat)
            lMr_cost += norm(ref_lMr[1] - lMr_trans)

            loss += lMr_cost * lambda_lMr

            # Distance to obsticle
            odist = dist_cost(oMl.translation, OBSTACLE_PLACEMENT.translation, 0.2)
            odist += dist_cost(oMr.translation, OBSTACLE_PLACEMENT.translation, 0.2)

            loss += odist * lambda_odist

            ret += 0.5 * dt * (prev_loss + loss)
            prev_loss = loss

        return ret

    def callback(p):
        print("Cost: ", cost(p))

    p_sol = fmin_bfgs(cost, p0, callback=callback, gtol=5e-3)
    path = np_to_path(p_sol)

    return path_to_bezier(path, max_acc)


def path_to_bezier(path, max_acc):
    q_of_t = Bezier(path, t_max=1)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)

    max_vvq = np.max(np.abs(vvq_of_t.control_points_))
    T = 1 / np.sqrt(max_acc / max_vvq)

    q_of_t = Bezier(path, t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)

    return q_of_t, vq_of_t, vvq_of_t


if __name__ == "__main__":
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil

    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, OBSTACLE_PLACEMENT
    from inverse_geometry import computeqgrasppose
    from path import computepath
    from tools import setupwithpybulletandmeshcat

    robot, sim, cube, viz = setupwithpybulletandmeshcat()

    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(
        robot,
        robot.q0,
        cube,
        CUBE_PLACEMENT_TARGET,
        None
        # robot, robot.q0, cube, CUBE_PLACEMENT_UP, None
    )
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    # path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_UP)

    # setting initial configuration
    sim.setqsim(q0)

    # optimization pruning
    q_optimize = np.var(path, axis=0) > 1e-10

    def oMf_to_quat_trans(oMf):
        return Quaternion(oMf.rotation), oMf.translation

    # trajs = maketraj_bfgs_trap(q0, qe, 2, path, q_optimize, robot)
    trajs = maketraj_cube_pot(q0, qe, 32, path, robot, cube, viz=viz)
    T = trajs[0].T_max_

    if True:
        for t in np.linspace(0, T, 100):
            viz.display(trajs[0](t))

    tcur = 0.0

    while tcur < T:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
