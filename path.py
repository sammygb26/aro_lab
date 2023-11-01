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


# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    sampleNo = 100
    samples = sampleCubePlacement(robot, qinit, cube, sampleNo, viz=None)
    RRT = RRTConnect(cubeplacementq0, cubeplacementq0, samples)
    path, configurations = RRT.plan()

    return configurations


def sampleCubePlacement(robot, q, cube, noSamples, viz=None):
    # Randomly sample cube placements
    samples = np.empty(noSamples, dtype=pin.SE3)
    for i in range(noSamples):
        placement = samplePlacement()
        while checkCollision(robot, q, cube, placement):
            placement = samplePlacement()
        samples[i] = placement
    return samples


def checkCollision(robot, q, cube, placement):
    q1, _ = computeqgrasppose(robot, q, cube, placement, viz)
    return collision(robot, q1)


def samplePlacement():
    t = np.random.rand(3)
    minimums = np.array([0.2, -0.7, 0.93])
    maximums = np.array([0.6, 0.4, 1.3])
    t = (t * (maximums - minimums)) + minimums
    cube_placement = pin.SE3(rotate("z", 0), t)
    return cube_placement


# Class to implement rapidly exploring random trees


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


class RRTConnect:
    def __init__(self, start, goal, samples, step_size=0.1, iterations=100):
        self.start_tree = [Node(start)]
        self.goal_tree = [Node(goal)]
        self.samples = samples
        self.step_size = step_size
        self.iterations = iterations

    def nearest_neighbor(self, tree, sample):
        distances = [self.calcDist(node.state, sample) for node in tree]
        nearest_node = tree[np.argmin(distances)]
        return nearest_node

    def calcDist(self, target, sample):
        distance = np.linalg.norm(
            sample.translation - target.translation
        )  # np.array([np.linalg.norm(pin.log(sample * np.linalg.inv(node.state))) for node in tree])
        return distance

    def new_state(self, nearest_node, sample):
        # Calculate the direction from the nearest state to the random state
        direction_translation = sample.translation - nearest_node.state.translation
        direction_rotation = pin.log(inv(nearest_node.state.rotation) * sample.rotation)

        # Ensure the step size is not greater than the distance
        magnitude_translation = np.linalg.norm(direction_translation)
        magnitude_rotation = np.linalg.norm(direction_rotation)

        if magnitude_translation < self.step_size:
            new_translation = sample.translation
        else:
            new_translation = (
                nearest_node.state.translation
                + (direction_translation / magnitude_translation) * self.step_size
            )

        if magnitude_rotation < self.step_size:
            new_rotation = sample.rotation
        else:
            new_rotation = nearest_node.state.rotation * pin.exp(
                (direction_rotation / magnitude_rotation) * self.step_size
            )

        # Create the new SE3 state
        new_state = pin.SE3(new_rotation, new_translation)
        return new_state

    def plan(self):
        for _ in range(self.iterations):
            sample = np.random.choice(self.samples)
            # Extend the tree from start towards the sample
            nearest_node_start = self.nearest_neighbor(self.start_tree, sample)
            new_state_start = self.new_state(nearest_node_start, sample)
            if not checkCollision(robot, q, cube, new_state_start):
                new_node_start = Node(new_state_start, nearest_node_start)
                self.start_tree.append(new_node_start)
            else:
                new_node_start = nearest_node_start

            # Extend the tree from goal towards the sample
            nearest_node_goal = self.nearest_neighbor(self.goal_tree, new_state_start)
            new_state_goal = self.new_state(nearest_node_goal, new_state_start)
            if not checkCollision(robot, q, cube, new_state_goal):
                new_node_goal = Node(new_state_goal, nearest_node_goal)
                self.goal_tree.append(new_node_goal)
            else:
                new_node_goal = nearest_node_goal

            # Check if we've connected the two trees
            if (
                self.calcDist(new_node_start.state, new_node_goal.state)
                < self.step_size
            ):
                return self.extract_path(new_node_start, new_node_goal)

        return None

    def extract_path(self, node_start, node_goal):
        path = []
        configurations = []
        while node_start is not None:
            path.insert(0, node_start.state)
            q1, _ = computeqgrasppose(robot, q, cube, node_start.state)
            configurations.insert(0, q1)
            node_start = node_start.parent
        node_goal = node_goal.parent
        while node_goal is not None:
            path.append(node_goal.state)
            q1, _ = computeqgrasppose(robot, q, cube, node_goal.state)
            configurations.insert(0, q1)
            node_goal = node_goal.parent
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

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    print(qe)
    print(path)

    displaypath(robot, path, dt=0.5, viz=viz)  # you ll probably want to lower dt
