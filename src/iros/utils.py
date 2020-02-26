import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3)


def rot_mat_3d(*args):
    if isinstance(args, tuple):
        ang_x, ang_y, ang_z = args[0]
    else:
        ang_x, ang_y, ang_z = args
    Rx = R.from_rotvec(ang_x * np.array([1, 0, 0]))
    Ry = R.from_rotvec(ang_y * np.array([0, 1, 0]))
    Rz = R.from_rotvec(ang_z * np.array([0, 0, 1]))
    Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
    return Rx @ Ry @ Rz


def rotation_matrix(ang):
    rx = [[1, 0, 0], [0, np.cos(ang[0]), -np.sin(ang[0])], [0, np.sin(ang[0]), np.cos(ang[0])]]
    ry = [[np.cos(ang[1]), 0, np.sin(ang[1])], [0, 1, 0], [-np.sin(ang[1]), 0, np.cos(ang[1])]]
    rz = [[np.cos(ang[2]), -np.sin(ang[2]), 0], [np.sin(ang[2]), np.cos(ang[2]), 0], [0, 0, 1]]
    rx, ry, rz = np.array(rx), np.array(ry), np.array(rz)
    Rot = rx @ ry @ rz
    return Rot


def get_curves(ref_path):
    aa = list()
    aa.append(ref_path.T)
    s, g = ref_path[0, :], ref_path[-1, :]
    for i in range(1, 5):
        bb = np.array([-3.5 + i, i * 2, 1.2 * i])
        aa.append(quadratic_bezier(s, bb, g))
    return aa


def cubic_bezier(start, b, c, goal):
    t = np.linspace(0, 1, 50)
    curve = np.zeros((3, len(t)))
    for i, t in enumerate(t):
        curve[:, i] = (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * goal
    return curve


def quadratic_bezier(start, b, goal):
    t = np.linspace(0, 1, 50)
    curve = np.zeros((3, len(t)))
    for i, t in enumerate(t):
        curve[:, i] = (1 - t)** 2 * start + 2 * (1 - t) * t * b + t ** 2 * goal
    return curve


if __name__ == '__main__':
    save_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = np.load(save_dir + '/save_data_inv_kin/data/ref_path_xyz1.npy', allow_pickle=True)
    start, goal = ref_path[0, :], ref_path[-1, :],  # np.array([1, 1, 1])
    center = start + 0.5 * (goal - start)
    # goal = np.array([1, 2, 3])
    b = (start + goal) * 0.5
    b = np.array([-3, 0, 4])
    quad_B_curve = quadratic_bezier(start, b, goal)
    curves = get_curves(ref_path)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(quad_B_curve[0, :], quad_B_curve[1, :], quad_B_curve[2, :], 'r^')
    plt.pause(0.05)
    ax.scatter(start[0], start[1], start[2], 'g*')
    ax.scatter(goal[0], goal[1], goal[2], 'g*')
    ax.scatter(b[0], b[1], b[2], 'g*')
    ax.scatter(center[0], center[1], center[2], marker='8')

    # plt.figure()
    for i in range(len(curves)):
        pass
        # ax.plot(curves[i][0, :], curves[i][1, :], curves[i][2, :], 'b^')
    plt.show()
