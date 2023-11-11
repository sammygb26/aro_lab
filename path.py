#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from pinocchio.utils import rotate
from numpy.linalg import pinv, inv, norm, svd, eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

import time
from inverse_geometry import computeqgrasppose
from util import log_cube
from pinocchio import SE3


# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    sampleNo = 100
    print("Starting sampling")
    samples = []  # sampleCubePlacements(robot, qinit, cube, sampleNo, viz=None)
    print("Starting RRT")
    RRT = RRTConnect(
        robot, cube, cubeplacementq0, cubeplacementqgoal, samples, qinit, qgoal
    )
    flag, (path, configurations) = RRT.plan()
    if flag == False:
        print(":sad_face:")
        return (
            flag,
            configurations,
        )

    return configurations

def sampleCubePlacements(robot, q, cube, noSamples, viz=None):
    # Randomly sample cube placements
    samples = np.empty(noSamples, dtype=pin.SE3)
    for i in range(noSamples):
        print(i)
        placement = samplePlacement(robot, q, cube, viz)
        samples[i] = placement
    return samples

def randomCubePlacement():
    minimums = np.array([0.2, -0.8, 1.0])
    maximums = np.array([0.8, 0.8, 2.5])
    t = np.random.rand(3)
    t = (t * (maximums - minimums)) + minimums
    return pin.SE3(rotate("z", 0), t)

def samplePlacement(robot, q, cube):
    while True:
        cube_placement = randomCubePlacement()
        _, success = computeqgrasppose(robot, q, cube, cube_placement, viz)
        if success:
            return cube_placement


# Class to implement rapidly exploring random trees
TRAPPED = 0
ADVANCED = 1
REACHED = 2

class Node:
    def __init__(self, state, pose, parent=None):
        self.state = state
        self.parent = parent
        self.pose = pose


class RRTConnect:
    def __init__(
        self, robot, cube, start, goal, samples, q0, qe, step_size=0.025, iterations=500
    ):
        self.start_tree = [Node(start, q0)]
        self.goal_tree = [Node(goal, qe)]
        self.samples = samples
        self.step_size = step_size
        self.iterations = iterations
        self.robot = robot
        self.cube = cube
        self.q = q0

    def nearest_neighbor(self, tree, sample) -> Node:
        distances = [self.calcDist(node.state, sample) for node in tree]
        nearest_node = tree[np.argmin(distances)]
        return nearest_node

    def calcDist(self, target : Node, sample : Node):
        distance = np.linalg.norm(
            sample.translation - target.translation
        )  # np.array([np.linalg.norm(pin.log(sample * np.linalg.inv(node.state))) for node in tree])
        return distance
    
    def new_config(self, c_target : SE3, n_near : Node):
        c_near = n_near.state
        if self.calcDist(n_near.state, c_target) < self.step_size:
            c_new = c_target
        else:
            move_dir = c_target.translation - c_near.translation
            move_dir /= norm(move_dir)
            step = move_dir * self.step_size
            c_new = pin.SE3(rotate("z", 0), c_near.translation + step)

        q, success = computeqgrasppose(
            self.robot, 
            n_near.pose, 
            self.cube, 
            c_new)
        
        return q, c_new, success

    def extend(self, tree, c_target):
        n_near = self.nearest_neighbor(tree, c_target)

        q, c_new, success = self.new_config(c_target, n_near)
        if not success:
            return TRAPPED, None
        
        n_new = Node(c_new, q, n_near)
        tree.append(n_new) 
        
        if c_new == c_target:
            return REACHED, n_new
        else:
            return ADVANCED, n_new
        
    def connect(self, tree, c_target):
        S = ADVANCED
        while S == ADVANCED:
            S, n_new = self.extend(tree, c_target)
        return S, n_new

    def plan(self):
        tree_a = self.start_tree
        tree_b = self.goal_tree
        switched = False

        for _ in range(self.iterations):
            c_rand = randomCubePlacement()

            S_e, n_new_e = self.extend(tree_a, c_rand)
            if not (S_e == TRAPPED):
                S_c, n_new_c = self.connect(tree_b, n_new_e.state)
                if S_c == REACHED:
                    if switched:
                        return True, self.extract_path(n_new_c, n_new_e)
                    else:
                        return True, self.extract_path(n_new_e, n_new_c)
                
            tmp = tree_a
            tree_a = tree_b
            tree_b = tmp
            switched = not switched 

        return False, []

    def extract_path(self, node_start, node_goal):
        path = []
        configurations = []
        while node_start is not None:
            path.insert(0, node_start.state)
            configurations.insert(0, node_start.pose)
            node_start = node_start.parent
        node_goal = node_goal.parent
        while node_goal is not None:
            path.append(node_goal.state)
            configurations.append(node_goal.pose)
            node_goal = node_goal.parent

        print(path, configurations)
        return path, configurations


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)

    print(successinit, successend)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    displaypath(robot, path, dt=0.02, viz=viz)  # you ll probably want to lower dt
