#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
from numpy.linalg import norm
from numpy.linalg import pinv
import pinocchio as pin
import time

from pinocchio.utils import rotate
from bezier import Bezier
from config import DT, LEFT_HAND, RIGHT_HAND
from scipy.optimize import fmin_bfgs


# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 1000  # proportional gain (P of PD)
Ki = 2400
i = 0
Kd = 1 * np.sqrt(Kp)

CUBE_PLACEMENT_UP = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 1.13]))


graspForce = 100
cubeweight = 150
graspTorque = 1

def getGraspForces(sim, robot):
    fext = [pin.Force.Zero()] * (robot.nv + 1)

    lid = sim.bulletCtrlJointsInPinOrder.index(sim.bullet_names2indices['LARM_EFF'] - 1) + 1
    rid = sim.bulletCtrlJointsInPinOrder.index(sim.bullet_names2indices['RARM_EFF'] - 1) + 1

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
    dvq = Kp * e_q + Kd * e_qd + i * Ki  + vvq_of_t(tcurrent)

    # Add grasp force
    graps_ext = getGraspForces(sim, robot) 

    h = pin.rnea(robot.model, robot.data, q, vq, dvq, graps_ext)
    M = pin.crba(robot.model, robot.data, q)

    torques = M @ dvq + h

    sim.step(torques)


if __name__ == "__main__":
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil

    robot, sim, cube = setupwithpybullet()

    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(
        robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None
        #robot, robot.q0, cube, CUBE_PLACEMENT_UP, None
    )
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    #path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_UP)

    # setting initial configuration
    sim.setqsim(q0)

    # TODO this is just an example, you are free to do as you please.
    # In any case this trajectory does not follow the path
    # 0 init and end velocities
    def maketraj(q0, q1, T):  # TODO compute a real trajectory !
        ref_q = Bezier(path, t_max=1)

        p0 = np.array(path)
        steps = 20
        npoints = 30
        p0 = np.array([ ref_q(t) for t in np.linspace(0,1,npoints)])

        def np_to_path(p):
            p = np.reshape(p, p0.shape)
            return q0, q0, q0, *[p[i,:] for i in range(npoints)], q1, q1, q1
        
        def np_to_traj(p):
            q_of_t = Bezier(np_to_path(p), t_max=1)
            vq_of_t = q_of_t.derivative(1)
            vvq_of_t = vq_of_t.derivative(1)
            return q_of_t, vq_of_t, vvq_of_t
        
        def cost(p):
            ret = 0

            q_of_t, vq_of_t, vvq_of_t = np_to_traj(p)

            dt = 1 / steps
            prev_loss = 0
            for t in np.linspace(0, 1, steps):
                loss = norm(ref_q(t) - q_of_t(t), ord=1) 

                ret += 0.5 * dt * (prev_loss + loss)
                prev_loss = loss

            return ret
        

        #p_sol = fmin_bfgs(cost, p0)
        p_sol = p0

        q_of_t = Bezier(np_to_path(p_sol), t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t

    # TODO this is just a random trajectory, you need to do this yourself
    total_time = 5.0
    trajs = maketraj(q0, qe, total_time)

    from tools import setupwithmeshcat
    if True:
        robot, cube, viz = setupwithmeshcat()
        for t in np.linspace(0, total_time, 100):
            viz.display(trajs[0](t)) 
        


    tcur = 0.0

    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
