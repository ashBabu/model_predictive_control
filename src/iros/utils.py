import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3)


class Utilities:
    def __init__(self, DH=None):
        if not isinstance(DH, dict):
            eef_dist = 0.3
            a = np.array([0., 0., 1., 0.7, 0.5, 0.5, 0., 0.])
            d = np.array([0.5, 0., 0., 0., 0., 0., 1.5, eef_dist])
            alpha = np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.])
            self.DH = {'a': a, 'd': d, 'alpha': alpha}
        else:
            self.DH = DH

    def rot_mat_3d(self, *args):
        if isinstance(args, tuple):
            ang_x, ang_y, ang_z = args[0]
        else:
            ang_x, ang_y, ang_z = args
        Rx = R.from_rotvec(ang_x * np.array([1, 0, 0]))
        Ry = R.from_rotvec(ang_y * np.array([0, 1, 0]))
        Rz = R.from_rotvec(ang_z * np.array([0, 0, 1]))
        Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
        return Rx @ Ry @ Rz

    def rotation_matrix(self, ang):
        rx = [[1, 0, 0], [0, np.cos(ang[0]), -np.sin(ang[0])], [0, np.sin(ang[0]), np.cos(ang[0])]]
        ry = [[np.cos(ang[1]), 0, np.sin(ang[1])], [0, 1, 0], [-np.sin(ang[1]), 0, np.cos(ang[1])]]
        rz = [[np.cos(ang[2]), -np.sin(ang[2]), 0], [np.sin(ang[2]), np.cos(ang[2]), 0], [0, 0, 1]]
        rx, ry, rz = np.array(rx), np.array(ry), np.array(rz)
        Rot = rx @ ry @ rz
        return Rot

    def get_curves(self, ref_path):
        aa = list()
        aa.append(ref_path.T)
        s, g = ref_path[0, :], ref_path[-1, :]
        for i in range(1, 5):
            bb = np.array([-3.5 + i, i * 2, 1.2 * i])
            aa.append(self.quadratic_bezier(s, bb, g))
        return aa

    def cubic_bezier(self, start, b, c, goal):
        t = np.linspace(0, 1, 50)
        curve = np.zeros((3, len(t)))
        for i, t in enumerate(t):
            curve[:, i] = (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * goal
        return curve

    def quadratic_bezier(self, start, b, goal):
        if isinstance(start, list):
            start = np.array(start)
        if isinstance(goal, list):
            goal = np.array(goal)
        t = np.linspace(0, 1, 50)
        curve = np.zeros((3, len(t)))
        for i, t in enumerate(t):
            curve[:, i] = (1 - t) ** 2 * start + 2 * (1 - t) * t * b + t ** 2 * goal
        return curve

    def skew_matrix(self, w):  # skew_symmetric matrix of the vector 'w'
        S = np.zeros((3, 3))
        S[0, 1] = -w[2]
        S[0, 2] = w[1]
        S[1, 0] = w[2]
        S[1, 2] = -w[0]
        S[2, 0] = -w[1]
        S[2, 1] = w[0]
        return S

    def euler_transformations(self, args=None):
        """
        :param args: array containing rotation and translation values along x, y, z
        :return: A 4 x 4 transformation matrix
        """
        if not isinstance(args, (list, tuple, np.ndarray,)):
            args = np.zeros(6)
        # ang_x, ang_y, ang_z, r0x, r0y, r0z = args
        Rot = self.rot_mat_3d(args[0:3])
        Rot = np.vstack((Rot, np.zeros(3)))
        T = np.hstack((Rot, np.array([[args[3]], [args[4]], [args[5]], [1]])))
        return T

    def transformationMatrix(self, q=None, DH=None,):
        """
        Forward kinematics of the manipulator
        :param q: joint angles of the manipulator/robot arm
        :param DH: Denavit Hartenberg parameters including the length of end-eff flange as shown in __init__()
        :return: T_joint: (len(a) x 4 x4) transformation from base to each of the joints
                 T_i: (len(a) x 4 x4) transformation from joint 'i' to 'i+i'
        """
        if not isinstance(DH, dict):
            DH = self.DH
        a, d, alpha = DH['a'], DH['d'], DH['alpha']
        t, T_joint, Ti = np.eye(4), np.zeros((len(a), 4, 4)), np.zeros((len(a), 4, 4))
        q = np.insert(q, len(q), 0.0, axis=0)  # for end-effector (just a translation for the fixed joint)
        for i in range(q.shape[0]):
            T = np.array([[np.cos(q[i]), -np.sin(q[i]), 0, a[i]],
                          [np.sin(q[i]) * np.cos(alpha[i]), np.cos(q[i]) * np.cos(alpha[i]),
                           -np.sin(alpha[i]), -np.sin(alpha[i]) * d[i]],
                          [np.sin(q[i]) * np.sin(alpha[i]), np.cos(q[i]) * np.sin(alpha[i]),
                           np.cos(alpha[i]), np.cos(alpha[i]) * d[i]],
                          [0, 0, 0, 1]], dtype='float')
            t = t @ T
            Ti[i, :, :] = T
            T_joint[i, :, :] = t
        return T_joint, Ti

    def plot_settings(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_aspect('equal')
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.grid(False)
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
        # X1, Y1, arr = self.image_draw('space.png')
        # ax.plot_surface(X1, Y1, np.ones(X1.shape), rstride=1, cstride=1, facecolors=arr)
        plt.axis('off')


if __name__ == '__main__':
    save_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = np.load(save_dir + '/save_data_inv_kin/data/ref_path_xyz1.npy', allow_pickle=True)
    start, goal = ref_path[0, :], ref_path[-1, :],  # np.array([1, 1, 1])

    center = start + 0.5 * (goal - start)
    scale = 0.1

    def get_circle(scale=0.1, start=None, goal=None, cent=None):
        angles = np.linspace(0, 2*np.pi, 100)
        circ = np.zeros((3, angles.shape[0]))
        dir_vec = goal - start
        """
            A vector perpendicular to dir_vec has np.dot(dir_vec, perp_vec) = 0
            Let perp_vec = (a, b, c) implies a x + b y + c z = 0. Put arbitrary values for a, b 
            which means c = -(1/z) * (a x + b y) 
            """
        a, b = 1, 1
        c = -(1 / dir_vec[2]) * (dir_vec[0] + dir_vec[1])
        perp_vec = np.array([a, b, c])

        for i, ang in enumerate(angles):
            rot = R.from_rotvec(ang * dir_vec)
            circ[:, i] = scale * (rot.as_dcm() @ perp_vec) + cent
        return circ

    circ1 = get_circle(start=start, goal=goal, cent=center)
    cen = center + 0.75 * (goal - center) + np.array([0.9, 5, 0])
    circ2 = get_circle(start=start, goal=goal, cent=cen)

    util = Utilities()
    curves = util.get_curves(ref_path)

    a = 3
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(start[0], start[1], start[2], 'g*')
    ax.scatter(goal[0], goal[1], goal[2], 'g*')
    ax.scatter(center[0], center[1], center[2], marker='8')
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    for points in circ1.T:
        quad_B_curve = util.quadratic_bezier(start, points, goal)
        ax.plot(quad_B_curve[0, :], quad_B_curve[1, :], quad_B_curve[2, :], 'r-')
        # ax.plot(circ[0, :], circ[1, :], circ[2, :])
        plt.pause(0.05)

    for p1, p2 in zip(circ1.T, circ2.T):
        cubic_B_curve = util.cubic_bezier(start, p1, p2, goal)
        ax.plot(cubic_B_curve[0, :], cubic_B_curve[1, :], cubic_B_curve[2, :], 'g-')
        # ax.plot(circ[0, :], circ[1, :], circ[2, :])
        plt.pause(0.05)
    # plt.figure()
    for i in range(len(curves)):
        pass
        # ax.plot(curves[i][0, :], curves[i][1, :], curves[i][2, :], 'b^')
    plt.show()
