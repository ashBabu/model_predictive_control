import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from scipy.spatial.transform import Rotation as R

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
r = [0, 1]
p = R.from_rotvec(np.pi/6 * np.array([1, 1, 1]))
p = p.as_dcm()
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        s, e = p.dot(s), p.dot(e)
        ax.plot3D(*zip(s, e), color="b")
plt.show()
