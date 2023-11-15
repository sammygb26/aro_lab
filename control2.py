from typing import Any
import numpy as np
from numpy.linalg import norm
from numpy.linalg import pinv
import pinocchio as pin
import time
from pinocchio.utils import rotate
from bezier import Bezier
from config import DT, LEFT_HAND, RIGHT_HAND
from scipy.optimize import fmin_bfgs
import math


# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 1200  # proportional gain (P of PD)
Ki = 24
i = 0
Kd = 2 * np.sqrt(Kp)

CUBE_PLACEMENT_UP = pin.SE3(rotate("z", 0.0), np.array([0.33, -0.3, 1.13]))


graspForce = 100
cubeweight = 100
graspTorque = 1


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
    dvq = Kp * e_q + Kd * e_qd + i * Ki + vvq_of_t(tcurrent)

    # Add grasp force
    graps_ext = getGraspForces(sim, robot)

    b = pin.rnea(robot.model, robot.data, q, vq, dvq, graps_ext)
    M = pin.crba(robot.model, robot.data, q)

    torques = M @ dvq + b

    sim.step(torques)


if __name__ == "__main__":
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil

    robot, sim, cube = setupwithpybullet()

    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

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

    # TODO this is just an example, you are free to do as you please.
    # In any case this trajectory does not follow the path
    # 0 init and end velocities

    class curve_wrapper:
        def __init__(self, curves, maxtime=10):
            self.curves = curves
            self.segments = len(curves)
            self.T = maxtime

        def __call__(self, t):
            curveNo = math.floor(((t / self.T) * self.segments))
            if curveNo == self.segments:
                curve = self.curves[self.segments - 1]
                return curve(1)
            else:
                curve = self.curves[curveNo]
                t = t * self.segments / self.T - curveNo
                return curve(t)

    def getControls(path, segments):
        start = path[0]
        end = path[-1]
        noPoints = len(path) - 2
        segmentSize = noPoints // segments
        print(noPoints, segmentSize)
        controls = [None] * segments
        curves = [None] * segments
        for i in range(segments - 1):
            path[(i + 1) * segmentSize] = (
                path[(i + 1) * segmentSize - 1] + path[(i + 1) * segmentSize + 1]
            ) / 2
            controls[i] = path[(i * segmentSize) : (i + 1) * segmentSize + 1]

        controls[0] = [start] * 2 + controls[0]
        controls[-1] = path[(segments - 1) * segmentSize + 1 :] + [end] * 2

        for i in range(segments):
            curves[i] = Bezier(
                controls[i],
                t_max=1,
            )

        return curves

    def maketraj(q0, q1, T):  # TODO compute a real trajectory !
        curves = getControls(path, 4)
        vqs = [curve.derivative(1) for curve in curves]
        vvqs = [vq.derivative(1) for vq in vqs]
        curves = curve_wrapper(curves, T)
        vqs = curve_wrapper(vqs, T)
        vvqs = curve_wrapper(vvqs, T)

        return curves, vqs, vvqs

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
