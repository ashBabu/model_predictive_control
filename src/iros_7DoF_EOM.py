"""
paper: Resolved Motion Rate Control of Space Manipulators with Generalized Jacobian Matrix, IEEE TRo
Author: Ash Babu
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sympy import *
from sympy.physics.mechanics import *
from sympy.tensor.array import Array
from utils import rot_mat_3d
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=2)
# init_printing()


class Kinematics:

    def __init__(self, nDoF=3, robot='3DoF'):
        # b0 = x, y, z location of robot's base
        self.nDoF = nDoF
        self.qm, self.qdm, self.l, self.r = self.initializing(self.nDoF)  # qm = manipulator joint angles
        # qdm = manipulator joint velocities, l = link lengths, r = COM vectors from joints

        if robot == '3DoF':  # as given in umeneti and yoshida: resolved motion rate control of space manipulators
            self.l_num = np.array([3.5, 0.25, 2.5, 2.5])
            self.ang_s0 = Array([0., 0., 0.])
            self.q0 = Array([pi / 3 * 0, 5*pi / 4, pi/2])  # as given in Umaneti and Yoshida: Resolved..
            # self.q0 = Array([pi / 3 * 0, pi / 2, 0])
            # DH parameters:
            self.alpha = Array([-pi / 2, pi / 2, 0.])
            self.a = Array([0., 0., 2.5])
            self.d = Array([0.25, 0., 0.])
            self.eef_dist = 2.50  # l3
        elif robot == '7DoF':
            self.l_num = np.array([3.5, 0.5, 0.5, 1., 0.7, 0.5, 0.5, 1.5])
            self.ang_s0 = Array([0., 0., 0.])
            self.q0 = np.array([0., 5*pi/4, 0., 0., 0., pi/2, 0.])

            self.eef_dist = 0.3
            self.a = np.array([0., 0., 1., 0.7, 0.5, 0.5, 0., 0.])
            self.d = np.array([0.5, 0., 0., 0., 0., 0., 1.5, self.eef_dist])
            self.alpha = np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.])

        else:
            self.a = Array([0, *self.l])
            self.d = Array([0.0, 0.0])
            self.alpha = Array([0.0, 0.0])
        hh = 2.1
        self.size = [(hh, hh, hh)]  # satellite dimension
        x, y, z = 0.5*self.size[0][0], 0.5*self.size[0][0], 0
        self.b0 = np.array([x, y, 0.], dtype=float)  # vector from spacecraft COM to robot base wrt spacecraft CS
        self.r_sx, self.r_sy, self.r_sz, = dynamicsymbols('r_sx r_sy r_sz')  # r_s = satellite pos_vec wrt inertial
        self.ang_xs, self.ang_ys, self.ang_zs = dynamicsymbols("ang_xs ang_ys ang_zs ")
        self.r_sxd, self.r_syd, self.r_szd, = dynamicsymbols('r_sx r_sy r_sz', 1)  # satellite linear vel wrt inertial
        self.w_sxd, self.w_syd, self.w_szd = dynamicsymbols("ang_xs ang_ys ang_zs ",
                                                            1)  # satellite angular vel wrt inertial
        self.q = Matrix([self.r_sx, self.r_sy, self.r_sz, self.ang_xs, self.ang_ys, self.ang_zs, *self.qm])
        self.qd = Matrix([self.r_sxd, self.r_syd, self.r_szd, self.w_sxd, self.w_syd, self.w_szd, *self.qdm])

        self.rx = symbols(["r%d" % x for x in range(1, nDoF+1)])  # x component of COM vector of the links
        self.aa = symbols(["aa%d" % x for x in range(1, nDoF+1)])  # x component of vector from joint i to COM of link i
        self.bb = symbols(["bb%d" % x for x in range(nDoF)])  # x component of vector frm COM of link i to joint i+1

        self.q_i, self.alpha_i, self.a_i, self.d_i = symbols("q_i alpha_i a_i d_i")
        # self.ang_xb, self.ang_yb, self.ang_zb, b0[0, b0[1, b0[2 = symbols(
        #     "ang_xb ang_yb ang_zb b0x b0y b0z")
        # self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z = symbols("ang_x ang_y ang_z r0x, r0y, r0z")
        # self.b0 = np.array([0.2, 0.3, 0.])  # vector from the spacecraft COM to the base of the robot in spacecraft CS

    def skew_matrix(self, w):  # skew_symmetric matrix of the vector 'w'
        S = np.zeros((3, 3))
        S[0, 1] = -w[2]
        S[0, 2] = w[1]
        S[1, 0] = w[2]
        S[1, 2] = -w[0]
        S[2, 0] = -w[1]
        S[2, 1] = w[0]
        return S

    def initializing(self, nDoF):
        # q = dynamicsymbols('q:{0}'.format(nDoF))
        # r = dynamicsymbols('r:{0}'.format(nDoF))
        # for i in range(1, nDoF+1):
        q = dynamicsymbols(['q%d' % x for x in range(1, nDoF+1)])
        qd = dynamicsymbols(["q%d" % x for x in range(1, nDoF+1)], 1)
        l = symbols(["l%d" % x for x in range(1, nDoF+1)])
        rx = symbols(["r%d" % x for x in range(1, nDoF+1)])  # x component of COM vector of the links
        r = []
        [r.append(zeros(3, 1)) for i in range(1, nDoF+1)]  # MOI matrix for the satellite and each of the links
        for i in range(nDoF):
            r[i][0] = 0.5 * rx[i]
        return q, qd, l, r

    def euler_transformations(self, args=None):
        if args:
            ang_x, ang_y, ang_z, r0x, r0y, r0z = args[0], args[1], args[2], args[3], args[4], args[5]
        else:
            ang_x, ang_y, ang_z, r0x, r0y, r0z = self.ang_xs, self.ang_ys, self.ang_zs, self.r_sx, self.r_sy, self.r_sz
        cx, cy, cz = np.cos(ang_x), np.cos(ang_y), np.cos(ang_z)
        sx, sy, sz = np.sin(ang_x), np.sin(ang_y), np.sin(ang_z)
        T = np.array([[cy*cz, -cy*sz, sy, r0x],
                    [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy, r0y],
                    [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy, r0z],
                    [0, 0, 0, 1]])
        return T

    def fwd_kin_manip(self, q):   # forward kinematics of the manipulator alone
        t, T_joint, Ti = np.eye(4), np.zeros((self.nDoF+1, 4, 4)), np.zeros((self.nDoF+1, 4, 4))
        q = np.array(simplify(q)).astype(np.float64)
        q = np.insert(q, len(q), 0.0, axis=0)  # for end-effector (just a translation for the fixed joint)
        for i in range(q.shape[0]):
            T = np.array([[np.cos(q[i]), -np.sin(q[i]), 0, self.a[i]],
                          [np.sin(q[i]) * np.cos(self.alpha[i]), np.cos(q[i]) * np.cos(self.alpha[i]), -np.sin(self.alpha[i]), -np.sin(self.alpha[i]) * self.d[i]],
                          [np.sin(q[i]) * np.sin(self.alpha[i]), np.cos(q[i]) * np.sin(self.alpha[i]), np.cos(self.alpha[i]), np.cos(self.alpha[i]) * self.d[i]],
                          [0, 0, 0, 1]], dtype='float')
            t = t @ T
            Ti[i, :, :] = T
            T_joint[i, :, :] = t
        return T_joint, Ti

    def robot_base_ang(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.b0
        an = (np.arctan2(b0[1], b0[0]) * 180 / np.pi) % 360  # gives an angles in 0 - 360 degrees
        an = (an - 90.) * np.pi / 180  # This means y axis is along the robot's first link as per DH
        ang_xb, ang_yb, ang_zb = 0., 0., an
        ang_b = np.array([ang_xb, ang_yb, ang_zb], dtype=float)
        return ang_b

    def fwd_kin_spacecraft(self, b0=None):
        # s_T_b = transformation from robot_base to spacecraft
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.b0
        ang_b = self.robot_base_ang(b0=b0)
        s_T_b = self.euler_transformations([ang_b[0], ang_b[1], ang_b[2], b0[0], b0[1], b0[2]])  # a constant 4 x 4 matrix
        return s_T_b

    def fwd_kin_full(self, q, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        T_joint, _ = self.fwd_kin_manip(q)
        s_T_b = self.fwd_kin_spacecraft(b0=b0)
        s_T_full = np.zeros((T_joint.shape[0]+1, 4, 4)) # j_T_full is n x 4 x 4 transf. matrices
        # j_T_full = [s_T_b, s_T_j1, s_T_j2,..., s_T_ee]
        # containing satellite, robot base and each of the joint CS
        s_T_full[0, :, :] = s_T_b
        for i in range(1, 2+self.nDoF):
            s_T_full[i, :, :] = s_T_b @ T_joint[i - 1]
        return s_T_full

    def pos_vect(self, q, b0=None):
        """
        position vectors of COM of each link wrt spacecraft CS, {s}
        {s}, {ji} are respectively the CS of spacecraft at its COM and joint CS of the manipulator
        """
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        s_T_full = self.fwd_kin_full(q, b0=b0)
        ax = plt.axes(projection='3d')
        # ax.scatter(0, 0, 0)  # just to know where (0, 0, 0) is
        def plott_check():  # plots the origin of the joints from satellite co-ordinate system which is at (0, 0, 0)
            for i in range(s_T_full.shape[0]):
                x, y, z = s_T_full[i, 0, 3], s_T_full[i, 1, 3], s_T_full[i, 2, 3]
                ax.scatter(x, y, z)
                ax.set_zlim(0, 2)
                plt.pause(0.01)
        # plott_check()
        pv_origins = np.zeros((3, s_T_full.shape[0]))  # pos vector of the origins of all CS wrt satellite {s}
        pv_com = []
        for i in range(0, s_T_full.shape[0]):  # includes end-eff origin
            pv_origins[:, i] = s_T_full[i, 0:3, 3]  # [s_r_b, s_r_j1, ...s_r_eef]
        j_com_vec = []
        kk = 0
        for i in range(0, pv_origins.shape[1]-1):
            v = pv_origins[:, i+1] - pv_origins[:, i]
            if v[0] or v[1] or v[2]:
                pv_com.append(pv_origins[:, i] + 0.5 * v)  # assuming COM exactly at the middle of the link
                j_com_vec.append(0.5 * v)  # vector from joint i to COM of link i described in spacecraft CS.
                # vector 'a' in Umeneti and Yoshida (but not wrt inertial)
                # ax.scatter(pv_com[kk][0], pv_com[kk][1], pv_com[kk][2], marker="v")  # to plot the COM except satellite
                kk += 1
        return pv_origins, np.array(pv_com).T, np.array(j_com_vec).T

    def rots_from_inertial(self, q, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.b0
        j_T_full = self.fwd_kin_full(q, b0=b0)
        rot_full = list()
        for i in range(len(j_T_full)):
            rot_full.append(j_T_full[i][0:3, 0:3])  # rotation matrix of spacecraft COM + each joint CS wrt inertial
            # including end-eff (which is same as link n). rot_full = [0_R_s, 0_R_rb, 0_R_j1, 0_R_j2, ... 0_R_jeef].
            # rb = robot base or joint 0 {j0}
        return rot_full

    def ab_vects(self, ang_s, q, com_vec=None, b0=None):
        """
        Note: The vectors are transformed to inertial in here
        As described in the paper, Here 'a' = vector from joint i to COM of link i wrt inertial
            # 'b' = vector from COM of link i to joint i+1 wrt inertial
        """
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.b0
        _, _, j_com_vec = self.pos_vect(q, b0=b0)
        if not isinstance(com_vec, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            R = rot_mat_3d(ang_s)
            a = R @ j_com_vec   # transformation to inertial of a
            b0 = R @ b0  # transformation to inertial of b
            b = np.hstack((b0.reshape(-1, 1), a))
        else:
            a, b = com_vec
        return a, b

    def velocities(self, q, *args, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.b0
        ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz = args
        j_T_full = self.fwd_kin_full(q, ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz, b0=b0)
        pv_origins, pv_com, j_com_vec = self.pos_vect(q, ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz, b0=b0)
        # j_T_full = [0_T_s, 0_T_j0, 0_T_j1, 0_T_j2,..., 0_T_ee]
        # j_com_vec =  vector from joint i to COM of link i wrt in inertial. vector 'a' in Umeneti and Yoshida
        omega = np.zeros(3, self.nDoF+2)
        joint_velocity = np.zeros(3, self.nDoF+2)
        com_vel = np.zeros(3, self.nDoF+1)
        b = np.array([[b0[0]], [b0[1]], [b0[2]]])
        omega[:, 0] = np.array([[self.w_sxd], [self.w_syd], [self.w_szd]])  # 0_w_s = ang vel of satellite wrt 0
        omega[:, 1] = np.array([[self.w_sxd], [self.w_syd], [self.w_szd]])  # 0_w_j0 = ang vel of robot base
        for i in range(2, 2+self.nDoF):
            temp = j_T_full[i][0:3, 2] * self.qdm[i - 2]
            omega[:, i] = omega[:, i-1] + temp

        joint_velocity[:, 0] = np.array([[self.r_sxd], [self.r_syd], [self.r_szd]])  # satellite linear vel of COM
        joint_velocity[:, 1] = joint_velocity[:, 0] +\
                               omega[:, 0].cross((j_T_full[0][0:3, 0:3] @ b))  # lin vel of robot_base ({j0})
        # Note: j_T_full[0][0:3, 0:3] @ b = pv_origins[:, 1] - pv_origins[:, 0]
        for i in range(2, 2+self.nDoF - 1):  # not considering end-eff vel
            v = pv_origins[:, i] - pv_origins[:, i-1]
            joint_velocity[:, i] = joint_velocity[:, i - 1] + omega[:, i-1].cross(v)
        jk = 0
        com_vel[:, 0] = joint_velocity[:, 0]
        for i in range(1, joint_velocity.shape[1]-1):
            if joint_velocity[:, i] == joint_velocity[:, i+1]:
                jk += 1
            com_vel[:, i] = joint_velocity[:, i+jk] + omega[:, i+1].cross(j_com_vec[:, i-1])
        return omega, com_vel, joint_velocity

    def plotter(self, ax, points, j_T_full, pv_origins, pv_com, j_r_c=(0, 0, 0)):
        ax.plot(points[0, :], points[1, :])  # draw rectangular satellite
        ax.arrow(0, 0.0, 0.5, 0., head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial x axis
        ax.arrow(0, 0., 0, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial y axis
        sc, kk = 0.25, 1
        for i in range(j_T_full.shape[0]):
            j_T_num = j_T_full[i].reshape(4, 4)
            trans_temp = pv_origins[:, i]
            rot_temp = j_T_num[0:3, 0:3]  # [i, 0:3, 0:3]
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
        # ax.plot(j_r_c[0], j_r_c[1], 'g^', markersize=12.)  # COM of the whole system (satellite + manipulator)
        ax.axis('equal')
        ax.set_ylim(0, 4.5)
        plt.xlabel('X')
        plt.ylabel('Y')


class Dynamics:

    def __init__(self, nDoF=3, robot='3DoF'):
        self.nDoF = nDoF
        self.Is_xx, self.Is_yy, self.Is_zz = symbols('Is_xx, Is_yy, Is_zz')
        self.Ixx = symbols(["Ixx%d" % x for x in range(1, self.nDoF+1)])  # x component of MOI of the links about its COm
        self.Iyy = symbols(["Iyy%d" % x for x in range(1, self.nDoF+1)])  # y component of MOI of the links about its COm
        self.Izz = symbols(["Izz%d" % x for x in range(1, self.nDoF+1)])  # z component of MOI of the links about its COm
        self.m = symbols(["m%d" % x for x in range(self.nDoF+1)])   # mass of space-craft and each of the links
        self.tau, self.I = self.initializing(self.nDoF)
        self.I_full = [self.Is_xx, self.Is_yy, self.Is_zz, *self.Ixx, *self.Iyy, *self.Izz]

        self.kin = Kinematics(nDoF=self.nDoF, robot=robot)

        # numeric values
        if robot == '3DoF':
            self.mass = np.array([200.0, 20.0, 50.0, 50.0], dtype=float)  # mass of satellite and each of the links respec
            self.Is = Matrix([[1400.0, 0.0, 0.0], [0.0, 1400.0, 0.0], [0.0, 0.0, 2040.0]])
            self.I1 = Matrix([[0.10, 0.0, 0.0], [0.0, 0.10, 0], [0.0, 0.0, 0.10]])
            self.I2 = Matrix([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
            self.I3 = Matrix([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
            self.I_num = np.array([1400.0, 1400.0, 2040.0, 0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])

        elif robot == '7DoF':
            self.mass = np.array([200.0, 20.0, 30.0, 30.0, 20.0, 20.0, 20.0, 20.0],
                                 dtype=float)  # mass of satellite and each of the links respec
            # self.Is = Matrix([[1400.0, 0.0, 0.0], [0.0, 1400.0, 0.0], [0.0, 0.0, 2040.0]])
            # self.I1 = Matrix([[0.10, 0.0, 0.0], [0.0, 0.10, 0], [0.0, 0.0, 0.10]])
            # self.I2 = Matrix([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
            # self.I3 = Matrix([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
            self.I_num = np.array([1400.0, 1400.0, 2040.0, 0.10, 0.25, 0.25, 0.10, 0.25, 0.25, 0.10, 0.25, 0.25,
                                   0.10, 0.25, 0.25, 0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26], dtype=float)

    def initializing(self, nDoF):
        # q = dynamicsymbols('q:{0}'.format(nDoF))
        # r = dynamicsymbols('r:{0}'.format(nDoF))
        # for i in range(1, nDoF+1):
        tau = symbols(["tau%d" % x for x in range(1, nDoF+1)])   # mass of space-craft and each of the links
        I = []
        [I.append(zeros(3)) for i in range(nDoF+1)]  # MOI matrix for the satellite and each of the links
        I[0][0, 0], I[0][1, 1], I[0][2, 2] = self.Is_xx, self.Is_yy, self.Is_zz
        for i in range(nDoF):
            I[i+1][0, 0] = self.Ixx[i]
            I[i+1][1, 1] = self.Iyy[i]
            I[i+1][2, 2] = self.Izz[i]
        return tau, I

    def mass_frac(self):
        M = np.sum(self.mass)
        k11 = np.zeros(self.nDoF)
        for j in range(1, self.nDoF+1):
            k11[j-1] = (-1 / M) * np.sum(self.mass[j:])
        return k11

    def com_pos_vect(self, q, ang_s, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        # rs, r1, r2 etc are pv from inertial to COM of spacecraft, link1, lin2, ...
        k11 = self.mass_frac()
        aa, bb = self.kin.ab_vects(q, ang_s, b0=b0)
        r0 = np.zeros(3)
        pv_com = []
        for i in range(self.nDoF):
            r0 += k11[i] * (bb[:, i] + aa[:, i])
        pv_com.append(r0)
        for i in range(1, self.nDoF + 1):
            tp = pv_com[i - 1] + bb[:, i - 1] + aa[:, i - 1]   # [j_rs, j_r1, j_r2, ...]
            pv_com.append(tp)
        pv_eef = pv_com[:, -1] + bb[:, -1]  # pos vec of eef wrt inertial = last entry of pv_origin + bb
        pv_origin = np.zeros((3, self.nDoF + 3))  # includes eef origin
        pv_origin[:, 0] = r0
        pv_origin[:, 1] = pv_origin[:, 0] + bb[:, 0]
        pv_orig_kin, _, _ = self.kin.pos_vect(q, b0=b0)
        h, ia, ib = 1, 0, 1
        for i in range(2, pv_orig_kin.shape[1]):
            v = pv_orig_kin[:, h+1] - pv_orig_kin[:, h]
            if not v[0] + v[1] + v[2]:
                pv_origin[:, i] = pv_origin[:, i - 1]
            else:
                pv_origin[:, i] = pv_origin[:, i-1] + aa[:, ia] + bb[:, ib]  # includes eef origin
                ia += 1
                ib += 1
            h += 1
        return pv_com, pv_eef, pv_origin

    def spacecraft_com_pos(self, ang_s, q, b0=None):
        """
        The spacecraft CG position is defined fully when the angular displacement (Euler angles) and manipulator joint
        angles are specified. Umentani and Yoshida's 'Resolved motion rate control' paper
        Summation m_i r_i = 0 (The CG of the whole spacecraft manipulator arm system is the origin of inertial CS) and
        r_i - r_(i-1) = a_i + b_(i-1)
        Solving the above simultaneous equation gives the CG position of spacecraft (r_s) wrt inertial.
        """
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.kin.b0
        k11 = self.mass_frac()
        a, b = self.kin.ab_vects(ang_s, q, b0=b0)
        r_s = np.zeros(3)
        for i in range(a.shape[1]):
            r_s += k11[i] * (b[:, i] + a[:, i])
        return r_s

    def jacobian_satellite(self, q, *args, b0=None):
        ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz = args
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        _, pv_eef, pv_origin = self.com_pos_vect(q, ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz, b0=b0)
        r_e_0 = pv_eef - pv_origin[:, 0]
        r_e0x = self.kin.skew_matrix(r_e_0)
        I, Z = np.eye(3), np.zeros((3, 3))
        tp = np.vstack((I, Z))
        tr = -np.vstack((r_e0x, I))
        J_sat = np.hstack((tp, tr))
        return J_sat

    def geometric_jacobian_manip(self, q, *args, b0=None):    # Method 2: for finding Jacobian
        ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz = args
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        _, pv_eef, pv_origin = self.com_pos_vect(q, ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz, b0=b0)
        j_T_full = self.kin.fwd_kin_full(q, b0=b0)  # [0_T_s, 0_T_b, 0_T_j1, 0_T_j2,..., 0_T_ee]
        J_manip = np.zeros((6, self.nDoF))  # initilizing jacobian
        h = list()
        for i in range(1, pv_origin.shape[1]-1):
            v = pv_origin[:, i+1] - pv_origin[:, i]
            if not v[0] + v[1] + v[2]:
                h.append(i+1)
        for i in h:
            pv_origin = np.delete(pv_origin, i, 1)
        for i in range(self.nDoF):
            pos_vec = pv_eef - pv_origin[:, i+1]  # pv_origin[:, 0] is satellite COM
            rot_axis = j_T_full[i+2][0:3, 2]
            rot_axis_x = self.kin.skew_matrix(rot_axis)
            J_manip[0:3, i] = rot_axis_x @ pos_vec
            J_manip[3:6, i] = rot_axis
        return J_manip

    def velocities_frm_momentum_conservation(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        t = Symbol('t')
        j_omega, _, _ = self.kin.velocities(b0=b0)
        pv_com, pv_eef, _ = self.com_pos_vect(b0=b0)
        j_vel_com = diff(pv_com, t)
        j_vel_eef = diff(pv_eef, t)
        return ImmutableMatrix(j_omega), j_vel_com, j_vel_eef

    def linear_momentum_conservation(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        j_omega, j_vel_com, _ = self.velocities_frm_momentum_conservation(b0=b0)
        L = zeros(3, 1)
        for i in range(self.nDoF+1):
            L += self.m[i] * j_vel_com[:, i]
        return L

    def momentOfInertia_transform(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        pv_com, _, _ = self.com_pos_vect(b0=b0)
        I = self.I
        rot_full = self.kin.rotations_from_inertial(b0=b0)
        rot_full.remove(rot_full[1])  # rot_full = [0_R_s, 0_R_j1, 0_R_j2, ...].
        I_transformed = list()
        for i in range(self.nDoF + 1):
            Ii = rot_full[i] @ I[i] @ rot_full[i].T
            tmp = pv_com[:, i].T @ pv_com[:, i]
            t1 = tmp[0] * eye(3) - (pv_com[:, i] @ pv_com[:, i].T)
            I_transformed.append(Ii + self.m[i] * t1)
        return I_transformed

    def ang_momentum_conservation(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        I = self.momentOfInertia_transform(b0=b0)
        j_omega, _, _ = self.velocities_frm_momentum_conservation(b0=b0)
        j_omega = j_omega.col_del(1)  # [0_w_s, 0_w_1, 0_w_2...] robot base and satellite has same angular velocity
        L = zeros(3, 1)
        for i in range(self.nDoF + 1):
            L += I[i] @ j_omega[:, i]  # + self.m[i] * pv_com[:, i].cross(j_vel_com[:, i])
            # The second term is not required since the coordinate system considered is at COM
            # I_o = I_com + I_/com
        return L

    def substitute(self, parm,  m=None, l=None, I=None, r_s0=None, ang_s0=None, q0=None,
                   omega_s=None, dq=None):
        if isinstance(m, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(m)):
                parm = msubs(parm, {self.m[i]: m[i]})
        if isinstance(I, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(I)):
                parm = msubs(parm, {self.I_full[i]: I[i]})
        if isinstance(l, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(l)):
                parm = msubs(parm, {self.kin.l[i]: l[i]})
        if isinstance(q0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(q0)):
                parm = msubs(parm, {self.kin.qm[i]: q0[i]})
        if isinstance(dq, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(dq)):
                parm = msubs(parm, {self.kin.qdm[i]: dq[i]})
        # if isinstance(b, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
        #     parm = msubs(parm, {self.kin.b0x: b[0], self.kin.b0y: b[1], self.kin.b0z: b[2]})
        if isinstance(ang_s0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            parm = msubs(parm, {self.kin.ang_xs: ang_s0[0], self.kin.ang_ys: ang_s0[1], self.kin.ang_zs: ang_s0[2]})
        if isinstance(omega_s, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            parm = msubs(parm, {self.kin.w_sxd: omega_s[0], self.kin.w_syd: omega_s[1], self.kin.w_szd: omega_s[2]})
        if isinstance(r_s0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            parm = msubs(parm, {self.kin.r_sx: r_s0[0], self.kin.r_sy: r_s0[1], self.kin.r_sz: r_s0[2]})
        # if isinstance(ang_b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
        #     parm = msubs(ang_b0, {self.kin.ang_xb: ang_b0[0], self.kin.ang_yb: ang_b0[1], self.kin.ang_zb: ang_b0[2]})
        return parm.evalf()

    def ang_moment_parsing(self, m=None, l=None, I=None, b0=None, ang_s0=None, q0=None, numeric=True):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        L = self.ang_momentum_conservation(b0=b0)
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        if numeric:
            L_num = self.substitute(L, m=m, l=l, I=I, ang_s0=ang_s0, q0=q0)
            Ls, Lm = L_num.jacobian(qd_s), L_num.jacobian(qd_m)
            Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)
        else:
            # L_sym = self.substitute(L, m=m, l=l, I=I,)
            Ls, Lm = L.jacobian(qd_s), L.jacobian(qd_m)
        return Ls, Lm

    def calculate_spacecraft_ang_vel(self, m=None, l=None, I=None, b0=None, ang_s0=None, q0=None, qdm=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        Ls, Lm = self.ang_moment_parsing(m=m, l=l, I=I, b0=b0, ang_s0=ang_s0, q0=q0,)
        shp = qdm.shape[1]
        omega_s = np.zeros((3, shp))
        for i in range(shp):
            omega_s[:, i] = -np.linalg.solve(Ls, (Lm @ qdm[:, i]))
        return omega_s

    def calculate_spacecraft_lin_vel(self,  m=None, l=None, I=None, b0=None, ang_s0=None, q0=None, qdm=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        omega_s = self.calculate_spacecraft_ang_vel(m=m, l=l, I=I, b0=b0, ang_s0=ang_s0, q0=q0, qdm=qdm)
        shp = omega_s.shape[1]
        j_omega, j_vel_com, j_vel_eef = self.velocities_frm_momentum_conservation(b0=b0)
        j_vel_com_num = self.substitute(j_vel_com, m=m, l=l, I=I, ang_s0=ang_s0, q0=q0)
        v_com = np.zeros((shp, 3, self.nDoF+1))
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        vcm = j_vel_com_num
        for j in range(shp):
            for i in range(len(qd_s)):
                j_vel_com_num = msubs(j_vel_com_num, {qd_s[i]: omega_s[i, j]})
            for k in range(len(qd_m)):
                j_vel_com_num = msubs(j_vel_com_num, {qd_m[k]: qdm[k, j]})
            v_com[j, :, :] = j_vel_com_num
            j_vel_com_num = vcm
        return v_com

    def kinetic_energy(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        j_I = self.momentOfInertia_transform(b0=b0)
        # w, com_vel, _ = self.kin.velocities()  # for the full 9 x 9 matrix (6 DOF for spacecraft and 3DOF for arm)
        w, com_vel, _ = self.velocities_frm_momentum_conservation(b0=b0)
        K = 0
        for i in range(self.nDoF + 1):
            K += 0.5*self.m[i]*com_vel[:, i].dot(com_vel[:, i]) + 0.5*w[:, i].dot(j_I[i] @ w[:, i])
        return K

    def get_dyn_para(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        K = self.kinetic_energy(b0=b0)
        q, qd = self.kin.q[3:], self.kin.qd[3:]
        # P = self.potential_energy()
        L = K   # Lagrangian. Potential energy at space is insignificant (microgravity envrnt)
        temp = transpose(Matrix([[K]]).jacobian(qd))
        M = temp.jacobian(qd) #.applyfunc(trigsimp)  # Mass matrix
        C = temp.jacobian(q) * Matrix(qd) - transpose(Matrix([[K]]).jacobian(q))  # Coriolis vector
        # C = C.applyfunc(trigsimp)
        # G = transpose(Matrix([[P]]).jacobian(q)).applyfunc(trigsimp)  # Gravity vector
        # LM = LagrangesMethod(L, q)
        # LM.form_lagranges_equations()
        # print LM.mass_matrix.applyfunc(trigsimp)
        # Matrix([P]).applyfunc(trigsimp)
        return M, C

    def get_dyn_para_num(self, m=None, l=None, I=None, b0=None, ang_s0=None, q0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        K = self.kinetic_energy(b0=b0)
        q, qd = self.kin.q[3:], self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        K_num = self.substitute(K, m=m, l=l, I=I, ang_s0=ang_s0, q0=q0)
        temp = transpose(Matrix([[K_num]]).jacobian(qd))
        Mt = temp.jacobian(qd)  # .applyfunc(trigsimp)  # Mass matrix
        Ct = temp.jacobian(q) * Matrix(qd) - transpose(Matrix([[K_num]]).jacobian(q))  # Coriolis vector
        # Ks, Km = K_num.jacobian(qd_s), K_num.jacobian(qd_m)
        # Ls, Lm = self.ang_moment_sparsing(m=m, l=l, I=I, b=b, ang_b0=ang_b0,)# ang_s0=ang_s0, q0=q0)
        # temp = np.linalg.solve(Ls, Lm)
        # K_num = (Km - Ks @ temp)

        # temp1 = transpose(Matrix([[K_num]]).jacobian(qd_m))
        # M = temp1.jacobian(qd_m)  # .applyfunc(trigsimp)  # Mass matrix
        # C = temp1.jacobian(self.kin.qm) * Matrix(qd_m) - transpose(Matrix([[K]]).jacobian(self.kin.qm))  # Coriolis vector
        return Mt, Ct

    def jnt_vel_prof_req(self):  # as given in Umetani and Yoshida
        # cc1, cc2 = -np.pi * 5 / 180, -np.pi * 10 / 180,
        t1, t2, t3, t4, t5 = np.linspace(0, 1.9, 10), np.linspace(2, 13.9, 20), np.linspace(14, 15.9, 10), \
                             np.linspace(16, 27.9, 20), np.linspace(28, 40, 20)
        t = np.hstack((t1, t2, t3, t4, t5))
        # t = np.linspace(0, 40, 80)
        def qdot(cc1):
            cc2, w = 2*cc1, 0.05*cc1
            y1, y2, y3, y4, y5 = np.linspace(0, cc1-w, 10), cc1*np.ones(20), np.linspace(cc1, 0, 10), \
                                 np.linspace(0+w, cc2, 20), np.linspace(cc2, 0, 20)
            # y1, y2, y3, y4, y5 = np.arange(0, cc1, 0.03), cc1*np.ones(20), np.linspace(cc1, 0, 10), \
            #                      np.linspace(0+w, cc2, 20), np.linspace(cc2, 0, 20)
            return np.hstack((y1, y2, y3, y4, y5))
        dt = 0.03
        # t = np.arange(0.03, 40, 0.03)
        tp = qdot(2)
        t = np.linspace(dt, 40, len(tp))
        q1_dot = np.zeros((len(tp)))
        q2_dot = qdot(-3*np.pi/(4*38))
        q3_dot = qdot(-np.pi/(2*36))
        return q1_dot, q2_dot, q3_dot, t

    def get_positions(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            b0 = self.kin.b0
        solver = Solver()
        m, I = self.mass, self.I_num
        l = self.kin.l_num[1:]  # cutting out satellite length l0
        ang_s0, q0, = self.kin.ang_s0, self.kin.q0
        ang_b0 = self.kin.robot_base_ang(b0=b0)
        q1_dot, q2_dot, q3_dot, t = self.jnt_vel_prof_req()
        qdm_numeric = np.vstack((q1_dot, q2_dot, q3_dot))

        pv_com, _, _ = self.com_pos_vect(b0=b0)
        pv_com_num0 = self.substitute(pv_com, m=m, l=l, I=I, ang_s0=ang_s0, q0=q0)
        pv_com_num0 = np.array(pv_com_num0).astype(np.float64)
        r_s0 = pv_com_num0[:, 0].reshape((3, 1))
        omega_s = self.calculate_spacecraft_ang_vel(m=m, l=l, I=I, b0=b0, ang_s0=ang_s0, q0=q0, qdm=qdm_numeric)
        ang_s0 = np.array(ang_s0).astype(np.float64).reshape((3, 1))

        q = solver.num_integration(qdm_numeric, q0, t)
        ang_s = solver.num_integration(omega_s, ang_s0, t)  # angular position of spacecraft COM as a function of time

        pv_com_num = np.zeros((len(t)+1, 3, self.nDoF+1))
        pv_com_num[0, :, :] = pv_com_num0
        temp = self.substitute(pv_com, m=m, l=l, I=I)
        for i in range(1, len(t)):
            tr = self.substitute(temp, ang_s0=ang_s[:, i], q0=q[:, i])
            pv_com_num[i, :, :] = np.array(tr).astype(np.float64)

        qm = self.kin.qm
        ang_sat = self.kin.q[3:6]
        parm = temp[:, 0]   # satellite com vector
        rs = zeros(3, q.shape[1])
        for j in range(q.shape[1]):
            for i in range(3):
                parm = msubs(parm, {ang_sat[i]: ang_s[i, j]})
            for k in range(len(qm)):
                parm = msubs(parm, {qm[k]: q[k, j]})
            rs[:, j] = parm
            parm = temp[:, 0]

        r_s = np.array(rs).astype(np.float64)
        # plt.plot(t, r_s[0, :], label='x_s')
        # plt.plot(t, r_s[1, :], label='y_s')
        # plt.plot(t, rs[0, :], label='xs')
        # plt.plot(t, rs[1, :], label='ys')
        # plt.legend()

        q = np.c_[q0, q]
        r_s = np.c_[r_s0, r_s]
        ang_s = np.c_[ang_s0, ang_s]
        t = np.insert(t, 0, 0)
        qdm_numeric = np.insert(qdm_numeric, 0, np.zeros(3), axis=1)
        return r_s, ang_s, q, qdm_numeric, t, pv_com_num


class Solver(object):

    def __init__(self):
        pass

    def num_integration(self, *args):
        derivative, init_pos, t = args
        # derivative is a 3 x t matrix whose integration is to be carried out
        shp = derivative.shape
        ln = len(t)
        integrand = np.zeros(shp)
        for i in range(shp[1]):
            if i == ln-1:
                dt = t[i] - t[i-1]
            else:
                dt = t[i+1] - t[i]
            integrand[:, i] = derivative[:, i] * dt + np.squeeze(init_pos)
            init_pos = integrand[:, i]
        return integrand


if __name__ == '__main__':

    nDoF = 7
    robot ='7DoF'
    kin = Kinematics(nDoF=nDoF, robot=robot)
    dyn = Dynamics(nDoF=nDoF, robot=robot)
    solver = Solver()

    m, I = dyn.mass, dyn.I_num
    l = kin.l_num[1:]  # cutting out satellite length l0
    b0 = kin.b0
    ang_b = kin.robot_base_ang(b0=b0)

    ang_s0 = np.array([0., 0., 0.])
    # q0 = Array([pi / 3 * 0, 5 * pi / 4, pi / 2])
    q0 = np.array([0., 5 * np.pi / 4, 0., 0., 0., 0, 0.])
    a, b, c = kin.pos_vect(q0)

    s = dyn.spacecraft_com_pos(ang_s0, q0)
    Ls, Lm = dyn.ang_moment_parsing(numeric=False, b0=b0)
    print('found Ls, Lm')
    Ls_d, Lm_d = diff(Ls, t), diff(Lm, t)
    print('found Ls_d, Lm_d')
    # deriv = dyn.spacecraft_acceleration(m=m, l=l, I=I)

    # M_num = dyn.substitute(M, m=m, l=l, I=I, ang_s0=ang_s0, q0=q0)
    # C_num = dyn.substitute(C, m=m, l=l, I=I, ang_s0=ang_s0, q0=q0)

    q, qd = kin.q[3:], kin.qd[3:]
    alpha, beta, gamma = symbols('alpha beta gamma')
    alpha_d, beta_d, gamma_d = symbols('alpha_d beta_d gamma_d')
    theta_1, theta_2, theta_3 = symbols('theta_1 theta_2 theta_3')
    theta_1d, theta_2d, theta_3d = symbols('theta_1d theta_2d theta_3d')
    ang_s = [alpha, beta, gamma]
    omega_s = [alpha_d, beta_d, gamma_d]
    theta = [theta_1, theta_2, theta_3]
    theta_d = [theta_1d, theta_2d, theta_3d]

    Ls_t = dyn.substitute(Ls, ang_s0=ang_s, q0=theta, omega_s=omega_s, dq=theta_d)
    print('found Ls_t')
    Lm_t = dyn.substitute(Lm, ang_s0=ang_s, q0=theta, omega_s=omega_s, dq=theta_d)
    print('found Lm_t')
    Ls_dt = dyn.substitute(Ls_d, ang_s0=ang_s, q0=theta, omega_s=omega_s, dq=theta_d)
    print('found Ls_dt')
    Lm_dt = dyn.substitute(Lm_d, ang_s0=ang_s, q0=theta, omega_s=omega_s, dq=theta_d)
    print('found Lm_dt')

    with open('Ls.pickle', 'wb') as Lsw:
        Lsw.write(pickle.dumps(Ls_t))
    with open('Lm.pickle', 'wb') as Lmw:
        Lmw.write(pickle.dumps(Lm_t))
    with open('Ls_d.pickle', 'wb') as Lsdw:
        Lsdw.write(pickle.dumps(Ls_dt))
    with open('Lm_d.pickle', 'wb') as Lmdw:
        Lmdw.write(pickle.dumps(Lm_dt))

    M, C = dyn.get_dyn_para(b0=b0)
    print('found M, C')
    Mt = dyn.substitute(M, ang_s0=ang_s, q0=theta, omega_s=omega_s, dq=theta_d)
    print('found Mt')
    Ct = dyn.substitute(C, ang_s0=ang_s, q0=theta, omega_s=omega_s, dq=theta_d)
    print('found Ct')

    with open('MassMat_sym.pickle', 'wb') as outM:
        outM.write(pickle.dumps(Mt))

    with open('Corioli_sym.pickle', 'wb') as outC:
        outC.write(pickle.dumps(Ct))
    print('done')
