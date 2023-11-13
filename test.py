import numpy as np
import matplotlib.pyplot as plt

from bezier import Bezier

P = np.array([[0,0], [0.5, 1], [1,-10]])

bez = Bezier(P, 0.0, 1.0)
T = np.linspace(0, 1, 1000)
points = np.array([bez(t) for t in T])

plt.scatter(P[:,0], P[:,1])
plt.plot(points[:,0], points[:, 1])
plt.show()