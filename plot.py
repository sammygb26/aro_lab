import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from bezier import Bezier

step_distance = 0.05
heat = 0.5
start_point = np.zeros(2)
direction = np.array([0.5, 0.5])
n = 30

points = np.zeros((n, 2))
points[0, :] = start_point
for i in range(1, n):
    points[i,:] = points[i-1,:] + direction * step_distance
    direction += np.random.normal(np.zeros(2), np.ones(2)) * heat
    direction /= norm(direction)

b = Bezier(points, t_max=1)
bn = 1000
t = np.linspace(0, 1, bn)
bpoints = np.zeros((bn, 2))
for i in range(bn):
    bpoints[i,:] = b(t[i])

plt.scatter(points[:, 0], points[:, 1], label='Origional', c='b')
plt.plot(bpoints[:, 0], bpoints[:, 1], label="Bezier", c='r')
plt.legend()
plt.savefig("Bezier.png", dpi=600)