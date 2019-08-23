# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from itertools import product, combinations
# from scipy.spatial.transform import Rotation as R
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect("equal")
#
#
# # draw cube
# def draw_cube(*args):
#     ang_xs, ang_ys, ang_zs, rs_x, rs_y, rs_z = args
#     r = [0, 1]
#     for i in range(len(ang_xs)):
#         Rx = R.from_rotvec(ang_xs[i] * np.array([1, 0, 0]))
#         Rx = Rx.as_dcm()
#         Ry = R.from_rotvec(ang_ys[i] * np.array([0, 1, 0]))
#         Ry = Ry.as_dcm()
#         Rz = R.from_rotvec(ang_zs[i] * np.array([0, 0, 1]))
#         Rz = Rz.as_dcm()
#         Rt = Rx @ Ry @ Rz
#
#         for s, e in combinations(np.array(list(product(r, r, r))), 2):
#             if np.sum(np.abs(s-e)) == r[1]-r[0]:
#                 s, e = Rt.dot(s), Rt.dot(e)
#                 ax.plot3D(*zip(s, e), color="b")
#                 # plt.gca().set_aspect('equal', adjustable='box')
#             scale = 2
#             ax.set_xlim(-1 * scale, 1 * scale)
#             ax.set_ylim(-1 * scale, 1 * scale)
#         plt.xlabel('X')
#         plt.ylabel('Y')
#         plt.pause(0.4)
#
# aa = np.pi
# ang_xs, ang_ys, ang_zs, rs_x, rs_y, rs_z = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), np.array([0, 0.01*aa, 0.05*aa, 0.1*aa]), 0., 0., 0.
# draw_cube(ang_xs, ang_ys, ang_zs, rs_x, rs_y, rs_z)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cuboid_data(o, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0), size=(1,1,1), ax=None,**kwargs):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data(pos, size )
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)

positions = [(0, 0, 0)]
sizes = [(2,2,2), (3,3,7)]
colors = ["crimson","limegreen"]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

for p,s,c in zip(positions,sizes,colors):
    plotCubeAt(pos=p, size=s, ax=ax, color=c)

plt.show()