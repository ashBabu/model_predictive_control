# considering 2 link planar manipulator fixed on a free-floating spacecraft at the moment
import numpy as np
from scipy.spatial.transform import Rotation as rm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from graphics import *
# win = GraphWin()


class eom():
    def __init__(self):
        self.numJoints = 2
        self.m_s, self.m1, self.m2 = 10, 4, 3   # m_s = spacecraft mass, m_i = mass of i^th link
        self.m = [self.m_s, self.m1, self.m2]
        # DH parameters
        self.l1, self.l2 = 1.5, 1   # l_i = length of i^th link
        self.a = np.array([0, self.l1, self.l2])
        self.d = np.array([0., 0.])
        self.alpha = np.array([0., 0.])

        # Dynamic Parameters
        # Inertia tensor wrt centre of mass of each link
        # Is = satellite MOI, I1,I2 = link MOI about its COM
        self.Is = np.zeros((3, 3))
        self.Is[2, 2] = 10
        self.I1 = np.zeros((3, 3))
        self.I1[2, 2] = 3
        self.I2 = np.zeros((3, 3))
        self.I2[2, 2] = 1
        self.I = [self.Is, self.I1, self.I2]

        self.b0 = np.array([0.2, 0.3, 0.])  # vector from the spacecraft COM to the base of the robot in spacecraft CS

    def skew_matrix(self, w):  # skew_symmetric matrix of the vector 'w'
        S = np.zeros((3, 3))
        S[0, 1] = -w[2]
        S[0, 2] = w[1]
        S[1, 0] = w[2]
        S[1, 2] = -w[0]
        S[2, 0] = -w[1]
        S[2, 1] = w[0]
        return S

    def robot_transformation_matrix(self, *args):  # transformation matrix relating i to i+1^th CS using DH convention
        q, a, d, alpha = args
        c_q, s_q = np.cos(q), np.sin(q)
        c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
        T = np.array([[c_q, -s_q, 0, a],
                      [s_q * c_alpha, c_q * c_alpha, -s_alpha, -s_alpha * d],
                      [s_q * s_alpha, c_q * s_alpha, c_alpha, c_alpha * d],
                      [0, 0, 0, 1]])
        return T

    def euler_transformation_mat(self, *args):
        # spacecraft COM location with respect to inertial CS is defined as three consecutive  Euler rotations and a
        # translation
        # Also,  from spacecraft  COM location to manipulator base with respect to spacecraft CS is defined as three
        # consecutive  Euler rotations and a translation (a constant matrix)
        ang_x, ang_y, ang_z, r0 = args
        rx, ry, rz = rm.from_euler('x', ang_x), rm.from_euler('y', ang_y), rm.from_euler('z', ang_z)
        temp = rx * ry * rz
        r = temp.as_dcm()
        T = np.zeros((4, 4))
        T[0:3, 0:3] = r
        tmp = np.insert(r0, len(r0), 1)
        T[:, -1] = tmp
        return T

    def fwd_kin(self, q, l=None):  # ln is the length of the last link on which the EE is attached.
        # q = np.insert(q, len(q), 0.0, axis=0)  # added 0 at the end for the fixed joint
        # As per Craigs convention, T = Rotx * Trans x * Rotz * Trans z; Franka follows this
        if isinstance(l, (list, tuple, np.ndarray)):
            a = l
        else:
            a = self.a
        T = np.eye(4)
        T_joint = np.zeros([self.numJoints+1, 4, 4])
        for i in range(self.numJoints):
            ac = self.robot_transformation_matrix(q[i], a[i], self.d[i], self.alpha[i])
            T = np.dot(T, ac)
            T_joint[i, :, :] = T  # gives the transformation of each joint wrt base CS
        temp = np.eye(4)
        temp[0, 3] = self.a[-1]
        T_ee = np.dot(T, temp)
        T_joint[-1, :, :] = T_ee
        return T_ee, T_joint

    def position_vectors(self, *args): # position vectors of COM of each link wrt inertial CS, {j}
        # {s}, {ji} are respectively the CS of spacecraft at its COM and joint CS of the manipulator
        q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        j_T_s = self.euler_transformation_mat(ang_xs, ang_ys, ang_zs, r0)
        s_T_j1 = self.euler_transformation_mat(ang_xb, ang_yb, ang_zb, b0)  # a constant 4 x 4 matrix
        T_ee, T_joint = self.fwd_kin(q, self.a)  #
        j_T_j1 = j_T_s @ s_T_j1
        j_T_full = np.zeros([self.numJoints+3, 4, 4])  # j_T_full is n x 4 x 4 transf. matrices
        # containing satellite, robot base and each of the joint CS
        j_T_full[0, :, :] = j_T_s
        j_T_full[1, :, :] = j_T_j1
        k = 0
        for i in range(2, 3+self.numJoints):
            j_T_full[i, :, :] = j_T_j1 @ T_joint[k, :, :]
            k += 1
        pv_origins = j_T_full[:, 0:3, 3]  # position vector of the origins of all coordinate system wrt inertial {j}
        pv_com = np.zeros((3, self.numJoints+1))  # position vector of the COM of
        # spacecraft + each of the links wrt inertial {j}
        pv_com[:, 0] = r0
        kk = 1
        for i in range(2, j_T_full.shape[0] - 1):
            trans_temp = pv_origins[i, :]
            rot_temp = j_T_full[i, 0:3, 0:3]
            pv_com[:, i-1] = trans_temp[0] + 0.5 * self.a[kk] * rot_temp[0, 0], trans_temp[1] + 0.5 * self.a[kk] * rot_temp[1, 0], 0
            kk += 1
        return j_T_full, np.transpose(pv_origins), pv_com

    def system_com(self, *args):
        q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        M = np.sum(self.m)
        _, _, pv_com = self.position_vectors(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
        j_r_c = 0
        for i in range(pv_com.shape[1]):
            j_r_c += (1/M) * self.m[i] * pv_com[:, i]
        j_r_0c = j_r_c - r0
        return j_r_c, j_r_0c

    def draw_sat_rect(self, T, l, b):
        p1, p2, p3, p4 = np.array([[l], [b], [0]]) * 0.5, np.array([[-l], [b], [0]]) * 0.5, np.array([[-l], [-b], [0]]) * 0.5, np.array([[l], [-b], [0]]) * 0.5
        centre = T[0:3, 3]
        rot = T[0:3, 0:3]
        points = centre.reshape(-1, 1) + rot @ np.hstack((p1, p2, p3, p4))
        return points

    def calculate_j_I_i(self, *args):
        # q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        j_T_full, pv_origins, pv_com = self.position_vectors(*args)
        R = j_T_full[0, 0:3, 0:3]
        j_I_i = np.zeros((self.numJoints, 3, 3))
        for i in range(2, j_T_full.shape[0] - 1):
            rot = j_T_full[i, 0:3, 0:3]  # rot matrix of joint CS wrt {j}. Here this CS is assumed to be same as CS at COM
            j_I_i[i - 2, :, :] = rot @ self.I[i - 1] @ rot.transpose()
        return j_I_i

    def calculate_jr_0x(self, *args):  # skew-sym matrix of vector j_r_0i. j_r_0i is the vector from the spacecraft COM to each of the link COM
        _, _, pv_com = self.position_vectors(*args)
        j_r_0ix = np.zeros((self.numJoints, 3, 3))
        for i in range(self.numJoints):
            j_r_0i = pv_com[:, i+1] - pv_com[:, 0]
            j_r_0ix[i, :, :] = self.skew_matrix(j_r_0i)
        return j_r_0ix

    def calculate_H_s(self, *args):
        j_T_full, pv_origins, pv_com = self.position_vectors(*args)
        R = j_T_full[0, 0:3, 0:3]
        j_I_0 = R @ self.I[0] @ R.transpose() # transforming satellite MOI express in its COM to inertial, {j}
        j_I_i = self.calculate_j_I_i(*args)
        j_I = 0
        j_r_0ix = self.calculate_jr_0x(*args)
        for i in range(j_I_i.shape[0]):
            j_I += j_I_i[i] - self.m[i + 1] * (j_r_0ix[i] @ j_r_0ix[i])
        return j_I + j_I_0

    def calculate_H_0(self, *args):
        H_s = self.calculate_H_s(*args)
        M = np.sum(self.m)
        A = M * np.eye(3)
        _, j_r_0c = self.system_com(*args)
        j_r_0cx = self.skew_matrix(j_r_0c)
        B = M * j_r_0cx
        C = np.vstack((A, B))
        D = np.vstack((-B, H_s))
        H_0 = np.hstack((C, D))
        return H_0


    def jacobian(self, *args):
        # q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        j_T_full, pv_origins, pv_com = self.position_vectors(*args)
        j_T_i = np.zeros((self.numJoints, 3, self.numJoints))  # Translational part of jacobian
        j_R_i = np.zeros((self.numJoints, 3, self.numJoints))  # Rotational part of jacobian
        num = np.arange(1, self.numJoints+1)
        j_rot_axis = np.transpose(j_T_full[2:-1, 0:3, 2])  # All rotation axes of the manipulator wrt {j}
        pv_origins = np.delete(pv_origins, -1, 1)  # deleting end-eff origin
        pv_origins = np.delete(pv_origins, 1, 1)  # [robot_base} and {j1} CS has same origin. Hence deleting
        j_p_r = pv_com - pv_origins
        for i in range(self.numJoints):
            for j in range(num[i]):
                j_T_i[i, :, j] = np.cross(j_rot_axis[:, j], j_p_r[:, j+1])
                j_R_i[i, :, j] = j_rot_axis[:, j]
        return j_R_i, j_T_i

    def calculate_j_TS(self, *args):
        _, j_T_i = self.jacobian(*args)
        j_TS = 0
        for i in range(j_T_i.shape[0]):
            j_TS += self.m[i+1] * j_T_i[i]
        return j_TS

    def calculate_H_sq(self, *args):
        j_I_i = self.calculate_j_I_i(*args)
        j_R_i, j_T_i = self.jacobian(*args)
        j_r_0ix = self.calculate_jr_0x(*args)
        H_sq = 0
        for i in range(self.numJoints):
            H_sq += j_I_i[i] @ j_R_i[i] + self.m[i + 1] * (j_r_0ix[i] @ j_T_i[i])
        return H_sq

    def calculate_H_0m(self, *args):
        j_TS = self.calculate_j_TS(*args)
        H_sq = self.calculate_H_sq(*args)
        H_0m = np.vstack((j_TS, H_sq))
        return H_0m

    def calculate_H_m(self, *args):
        j_I_i = self.calculate_j_I_i(*args)
        j_R_i, j_T_i = self.jacobian(*args)
        H_m = 0
        for i in range(j_T_i.shape[0]):
            H_m += j_R_i[i].transpose() @ j_I_i[i] @ j_R_i + self.m[i+1] * j_T_i.transpose() @ j_T_i
        return H_m

    def calculate_H_star(self, *args):
        H_m = self.calculate_H_m(*args)
        H_0m = self.calculate_H_0m(*args)
        H_0 = self.calculate_H_0(*args)
        H_star = H_m - H_0m.transpose() @ np.linalg.solve(H_0, H_0m)
        return H_star

    def plotter(self, ax, points, j_T_full, pv_origins, pv_com, j_r_c):
        ax.plot(points[0, :], points[1, :])  # draw rectangular satellite
        ax.arrow(0, 0.0, 0.5, 0., head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial x axis
        ax.arrow(0, 0., 0, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial y axis
        sc, kk = 0.25, 1
        for i in range(j_T_full.shape[0]):
            trans_temp = pv_origins[:, i]
            rot_temp = j_T_full[i, 0:3, 0:3]
            ax.plot([0, trans_temp[0]],
                    [0, trans_temp[1]])  # vector from origin of {j} to each of the origins of {j_i} CS
            ax.arrow(trans_temp[0], trans_temp[1], sc * rot_temp[0, 0], sc * rot_temp[1, 0], head_width=0.05,
                     head_length=0.1, fc='k', ec='k')  # x axis of {j_i} th CS
            ax.arrow(trans_temp[0], trans_temp[1], sc * rot_temp[0, 1], sc * rot_temp[1, 1], head_width=0.05,
                     head_length=0.1, fc='k', ec='k')  # y axis of {j_i} th CS

        for com in pv_com.transpose():
            ax.plot(com[0], com[1], 'ro', markersize=7.)  # COM of spacecraft + each of the link
        for i in range(1, pv_origins.shape[1] - 1):
            ax.plot([pv_origins[0, i], pv_origins[0, i + 1]],
                    [pv_origins[1, i], pv_origins[1, i + 1]], linewidth=4.0)  # plots the links of the manipulator
        ax.plot(j_r_c[0], j_r_c[1], 'g^', markersize=12.)  # COM of the whole system (satellite + manipulator)
        ax.axis('equal')
        ax.set_ylim(0, 4.5)
        plt.xlabel('X')
        plt.ylabel('Y')


if __name__ == '__main__':
    eom = eom()
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    ax = fig.add_subplot(111,)
    q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = [np.pi/6, np.pi/6], 0., 0., np.pi/24, 0., 0., np.pi/4, \
                                                                np.array([1., 1., 0.]), np.array([0.5, 0.25, 0.])
    j_T_full, pv_origins, pv_com = eom.position_vectors(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    points = eom.draw_sat_rect(j_T_full[0], 1., 0.5)
    aa = points[:, 0]
    points = np.hstack((points, aa.reshape(-1, 1)))  # in order to draw closed rectangle

    # j_r_c is  the vector from {j} to system COM and j_r_0c is the vector from satellite COM to system COM
    j_r_c, j_r_0c = eom.system_com(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    H_s = eom.calculate_H_s(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    H_sq = eom.calculate_H_sq(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    H_0m = eom.calculate_H_0m(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    H_0 = eom.calculate_H_0(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    eom.plotter(ax, points, j_T_full, pv_origins, pv_com, j_r_c)

    plt.show()
    print('hi')

