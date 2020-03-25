"""
paper: Resolved Motion Rate Control of Space Manipulators with Generalized Jacobian Matrix, IEEE TRo
Author: Ash Babu
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import Utilities
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=2)


class Kinematics:

    def __init__(self, nDoF=3, robot='3DoF'):
        # b0 = x, y, z location of robot's base
        self.nDoF = nDoF

        if robot == '7DoF':
            self.l_num = np.array([2.1, 0.5, 0.9, 0.9, 0.8, 0.8, 0.8, 0.9])
            self.ang_s0 = np.array([0.0, 0.0, 0.0])

            self.eef_dist = 0.9
            self.a = np.array([0., 0., 0.9, 0.9, 0.8, 0.8, 0., 0.])
            self.d = np.array([0.5, 0., 0., 0., 0., 0., 0.8, self.eef_dist])
            self.alpha = np.array([-np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 0., -np.pi / 2, -np.pi / 2, 0.])
            self.q0 = np.array([0., 5*np.pi/4, 0., 0., np.pi / 2, -np.pi / 2, 0.])
            self.DH = {'a': self.a, 'd': self.d, 'alpha': self.alpha}

        else:
            self.l_num = np.array([3.5, 0.25, 2.5, 2.5])
            self.a = np.array([1, 1])
            self.d = np.array([0.0, 0.0])
            self.alpha = np.array([0.0, 0.0])
        self.util = Utilities(DH=self.DH)
        hh = self.l_num[0]
        self.size = [(hh, hh, hh)]  # satellite dimension
        x, y, z = 0.5*self.size[0][0], 0.5*self.size[0][0], 0
        self.b0 = np.array([x, y, 0.], dtype=float)  # vector from spacecraft COM to robot base wrt spacecraft CS

    def skew_matrix(self, w):  # skew_symmetric matrix of the vector 'w'
        S = np.zeros((3, 3))
        S[0, 1] = -w[2]
        S[0, 2] = w[1]
        S[1, 0] = w[2]
        S[1, 2] = -w[0]
        S[2, 0] = -w[1]
        S[2, 1] = w[0]
        return S

    def euler_transformations(self, args):
        ang_x, ang_y, ang_z, r0x, r0y, r0z = args  # args[0], args[1], args[2], args[3], args[4], args[5]
        cx, cy, cz = np.cos(ang_x), np.cos(ang_y), np.cos(ang_z)
        sx, sy, sz = np.sin(ang_x), np.sin(ang_y), np.sin(ang_z)
        T = np.array([[cy*cz, -cy*sz, sy, r0x],
                    [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy, r0y],
                    [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy, r0z],
                    [0, 0, 0, 1]])
        return T

    def fwd_kin_manip(self, q=None,):   # forward kinematics of the manipulator alone
        t, T_joint, Ti = np.eye(4), np.zeros((self.nDoF+1, 4, 4)), np.zeros((self.nDoF+1, 4, 4))
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
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        an = (np.arctan2(b0[1], b0[0]) * 180 / np.pi) % 360  # gives an angles in 0 - 360 degrees
        an = (an - 90.) * np.pi / 180  # This means y axis is along the robot's first link as per DH
        ang_xb, ang_yb, ang_zb = 0., 0., an
        ang_b = np.array([ang_xb, ang_yb, ang_zb], dtype=float)
        return ang_b

    def fwd_kin_spacecraft(self, b0=None):
        # s_T_b = transformation from robot_base to spacecraft
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        ang_b = self.robot_base_ang(b0=b0)
        s_T_b = self.euler_transformations([ang_b[0], ang_b[1], ang_b[2], b0[0], b0[1], b0[2]])  # a const 4 x 4 matrix
        return s_T_b

    def fwd_kin_full(self, q=None, b0=None):
        """
        s_T_full is the matrix containing robot base and all other joint CS wrt spacecraft co-ordinate system
        """
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        T_joint, _ = self.fwd_kin_manip(q=q)
        s_T_b = self.fwd_kin_spacecraft(b0=b0)
        s_T_full = np.zeros((T_joint.shape[0]+1, 4, 4))  # s_T_full is n x 4 x 4 transf. matrices
        """s_T_full = [s_T_b, s_T_j1, s_T_j2,..., s_T_ee]
        # containing robot base (fixed joint 0) and each of the joint CS """
        s_T_full[0, :, :] = s_T_b
        for i in range(1, 2+self.nDoF):
            s_T_full[i, :, :] = s_T_b @ T_joint[i - 1]
        return s_T_full

    def pos_vect(self, q=None, b0=None, pltt=False):
        """
        position vectors of COM of each link wrt spacecraft CS, {s}
        {s}, {j_i} are respectively the CS of spacecraft at its COM and joint CS of the manipulator
        """
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        pltt = pltt
        s_T_full = self.fwd_kin_full(q=q, b0=b0)   # s_T_full = [s_T_b, s_T_j1, s_T_j2,..., s_T_ee]
        def plott_check(ax):  # plots the origin of the joints from satellite co-ordinate system which is at (0, 0, 0)
            # ax.scatter(0, 0, 0)  # just to know where (0, 0, 0) is
            for i in range(s_T_full.shape[0]):
                x, y, z = s_T_full[i, 0, 3], s_T_full[i, 1, 3], s_T_full[i, 2, 3]
                ax.scatter(x, y, z)
                ax.set_zlim(0, 2)
                plt.pause(0.01)
        if pltt:
            ax = plt.axes(projection='3d')
            plott_check(ax)
        pv_com = []
        pv_origins = s_T_full[:, 0:3, 3].T   # s_T_full = [s_T_b, s_T_j1, s_T_j2,..., s_T_ee]
        j_com_vec = []
        kk = 0
        for i in range(pv_origins.shape[1]-1):
            v = pv_origins[:, i+1] - pv_origins[:, i]
            if v[0] or v[1] or v[2]:
                pv_com.append(pv_origins[:, i] + 0.5 * v)  # assuming COM exactly at the middle of the link
                j_com_vec.append(0.5 * v)  # vector from joint i to COM of link i described in spacecraft CS.
                # vector 'a' in Umeneti and Yoshida (but not wrt inertial)
                if pltt:
                    ax.scatter(pv_com[kk][0], pv_com[kk][1], pv_com[kk][2], marker="v")  # to plot the COM except satellite
                kk += 1
        return pv_origins, np.array(pv_com).T, np.array(j_com_vec).T

    def ab_vects(self, q=None, ang_s=None, com_vec=None, b0=None):
        """
        Note: The vectors are transformed to inertial CS by multiplying with the rotation matrix obtained using
        ang_s. The translations are handled separately.
        As described in the paper, Here 'a' = vector from joint i to COM of link i wrt inertial
            # 'b' = vector from COM of link i to joint i+1 wrt inertial
        """
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        _, _, j_com_vec = self.pos_vect(q=q, b0=b0)
        if not isinstance(com_vec, (list, tuple, np.ndarray)):
            j_R_s = self.util.rot_mat_3d(ang_s)
            a = j_R_s @ j_com_vec   # transformation to inertial of 'a'
            b0 = j_R_s @ b0  # transformation to inertial of b
            b = np.hstack((b0.reshape(-1, 1), a))
        else:
            a, b = com_vec
        return a, b

    def rots_from_inertial(self, q=None, ang_s=None, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.b0
        s_T_full = self.fwd_kin_full(q=q, b0=b0)  # wrt spacecraft {s}
        j_R_s = self.util.rot_mat_3d(ang_s)  # rot mat of {s} wrt {j}
        rot_full_inertial = np.zeros((s_T_full.shape[0]+1, 3, 3))
        rot_full_inertial[0, :, :] = j_R_s
        for i in range(1, rot_full_inertial.shape[0]):
            rot_full_inertial[i, :, :] = j_R_s @ s_T_full[i-1, 0:3, 0:3]
        return rot_full_inertial

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
        self.kin = Kinematics(nDoF=self.nDoF, robot=robot)

        # numeric values
        if robot == '3DoF':
            self.mass = np.array([200.0, 20.0, 50.0, 50.0], dtype=float)  # mass of satellite and each of the links respec
            self.Is = np.array([[1400.0, 0.0, 0.0], [0.0, 1400.0, 0.0], [0.0, 0.0, 2040.0]])
            self.I1 = np.array([[0.10, 0.0, 0.0], [0.0, 0.10, 0], [0.0, 0.0, 0.10]])
            self.I2 = np.array([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
            self.I3 = np.array([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
            self.I_num = np.array([1400.0, 1400.0, 2040.0, 0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])

        elif robot == '7DoF':
            self.mass = np.array([200.0, 20.0, 30.0, 30.0, 20.0, 20.0, 20.0, 20.0],
                                 dtype=float)  # mass of satellite and each of the links respec
            self.Is = np.array([[1400.0, 0.0, 0.0], [0.0, 1400.0, 0.0], [0.0, 0.0, 2040.0]])
            self.I1 = np.array([[0.10, 0.0, 0.0], [0.0, 0.10, 0], [0.0, 0.0, 0.10]])
            self.I2 = np.array([[0.25, 0.0, 0.0], [0.0, 25.0, 0], [0.0, 0.0, 25.0]])
            self.I3 = np.array([[0.25, 0.0, 0.0], [0.0, 25.0, 0], [0.0, 0.0, 25.0]])
            self.I4 = np.array([[0.25, 0.0, 0.0], [0.0, 25.0, 0], [0.0, 0.0, 25.0]])
            self.I5 = np.array([[0.25, 0.0, 0.0], [0.0, 25.0, 0], [0.0, 0.0, 25.0]])
            self.I6 = np.array([[0.25, 0.0, 0.0], [0.0, 25.0, 0], [0.0, 0.0, 25.0]])
            self.I7 = np.array([[0.25, 0.0, 0.0], [0.0, 25.0, 0], [0.0, 0.0, 25.0]])
            self.I = [self.Is, self.I1, self.I2, self.I3, self.I4, self.I5, self.I6, self.I7]
            self.I_num = np.array(np.hstack((self.Is.diagonal(), self.I1.diagonal(), self.I2.diagonal(),
                                             self.I3.diagonal(), self.I4.diagonal(), self.I5.diagonal(),
                                             self.I6.diagonal(), self.I7.diagonal())), dtype=float)

    def mass_frac(self):
        M = np.sum(self.mass)
        k11 = np.zeros(self.nDoF)
        for j in range(1, self.nDoF+1):
            k11[j-1] = (-1 / M) * np.sum(self.mass[j:])
        return k11

    def spacecraft_com_pos(self, q=None, ang_s=None, b0=None):
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
        a, b = self.kin.ab_vects(q=q, ang_s=ang_s, b0=b0)
        r_s = np.zeros(3)
        for i in range(a.shape[1]):
            r_s += k11[i] * (b[:, i] + a[:, i])
        return r_s

    def transf_from_inertial(self, q=None, ang_s=None, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.kin.b0
        j_R_s = self.kin.util.rot_mat_3d(ang_s)  # rotation matrix of satellite CS, {s}, wrt inertial CS, {j}
        j_R_s = np.vstack((j_R_s, np.zeros(3)))
        r0 = self.spacecraft_com_pos(q=q, ang_s=ang_s, b0=b0)  # wrt inertial {j}
        r0 = np.hstack((r0, 1.))  # homogeneous vector
        j_T_s = np.hstack((j_R_s, r0.reshape(-1, 1)))  # 4 x 4 homogeneous transf. matrix of spacecraft wrt inertial
        s_T_full = self.kin.fwd_kin_full(q=q, b0=b0)  # [s_T_b, s_T_j1, s_T_j2,..., s_T_ee]; wrt spacecraft
        ns = s_T_full.shape[0] + 1
        j_T_full = np.zeros((ns, 4, 4))  # wrt inertial
        j_T_full[0, :, :] = j_T_s
        for i in range(1, ns):
            j_T_full[i, :, :] = j_T_s @ s_T_full[i - 1]  # wrt inertial {j}
        return j_T_full

    def pos_vect_inertial(self, q=None, ang_s=None, b0=None, pltt=False):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.kin.b0
        # rs, r1, r2 etc are pv from inertial to COM of spacecraft, link1, lin2, ...
        r0 = self.spacecraft_com_pos(q=q, ang_s=ang_s, b0=b0)  # wrt inertial {j}
        pv_orig_kin, pv_com_kin, _ = self.kin.pos_vect(q=q, b0=b0)  # wrt spacecraft CS
        a, b = self.kin.ab_vects(q=q, ang_s=ang_s, b0=b0)
        j_R_s = self.kin.util.rot_mat_3d(ang_s)  # rotation matrix of satellite CS, {s}, wrt inertial CS, {j}
        r_tiled = np.tile(r0, (9, 1)).T

        pv_origin = j_R_s @ pv_orig_kin + r_tiled
        pv_origin = np.hstack((r0.reshape(-1, 1), pv_origin))  # adding spacecraft origin (which is its CG)

        pv_com = j_R_s @ pv_com_kin + r_tiled[:, 0:pv_com_kin.shape[1]]
        pv_com = np.hstack((r0.reshape(-1, 1), pv_com))  # adding spacecraft CG

        pv_eef = pv_com[:, -1] + b[:, -1]  # pos vec of eef wrt inertial = last entry of pv_com + bb

        def plott_check():  # plots the origin of the joints from satellite co-ordinate system which is at (0, 0, 0)
            for i in range(pv_origin.shape[1]):
                ox, oy, oz = pv_origin[0, i], pv_origin[1, i], pv_origin[2, i]
                ax.scatter(ox, oy, oz)
                ax.set_zlim(0, 2)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                # plt.pause(0.01)
            for i in range(pv_com.shape[1]):
                ox, oy, oz = pv_com[0, i], pv_com[1, i], pv_com[2, i]
                ax.scatter(ox, oy, oz, marker='^')
                ax.set_zlim(0, 2)
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                plt.pause(0.01)
        if pltt:
            ax = plt.axes(projection='3d')
            # ax.scatter(0, 0, 0)  # just to know where (0, 0, 0) is
            plott_check()
        return pv_com, pv_eef, pv_origin

    def geometric_jacobian_satellite(self, q=None, ang_s=None, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.kin.b0
        j_R_s = self.kin.util.rot_mat_3d(ang_s)
        x_axis, y_axis, z_axis = j_R_s[:, 0], j_R_s[:, 1], j_R_s[:, 2]  # wrt inertial
        _, pv_eef, pv_origin = self.pos_vect_inertial(q=q, ang_s=ang_s, b0=b0)
        s_r_eef = pv_eef - pv_origin[:, 0]
        # r_e0x = self.kin.skew_matrix(r_e_0)
        I, Z = np.eye(3), np.zeros((3, 3))
        lin_vel_jac = np.vstack((I, Z))
        sa, sb, sc = np.cross(x_axis, s_r_eef), np.cross(y_axis, s_r_eef), np.cross(z_axis, s_r_eef)
        ta = np.vstack((sa, sb, sc)).T
        tb = np.vstack((x_axis, y_axis, z_axis)).T
        ang_vel_jac = np.vstack((ta, tb))  # J_s (as described in Umetani & Yoshida)
        J_sat = np.hstack((lin_vel_jac, ang_vel_jac))
        return ang_vel_jac, J_sat

    def geometric_jacobian_manip(self, q=None, ang_s=None, b0=None):    # Method 2: for finding Jacobian
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.kin.b0
        _, pv_eef, pv_origin = self.pos_vect_inertial(q=q, ang_s=ang_s, b0=b0)
        j_T_full = self.transf_from_inertial(q=q, ang_s=ang_s, b0=b0)
        J_manip = np.zeros((6, self.nDoF))  # initilizing jacobian
        for i in range(self.nDoF):
            pos_vec = pv_eef - pv_origin[:, i+2]  # pv_origin[:, 0], pv_origin[:, 1] are spacecraft and robot_base COM
            rot_axis = j_T_full[i+2, 0:3, 2]
            rot_axis_x = self.kin.skew_matrix(rot_axis)
            J_manip[0:3, i] = rot_axis_x @ pos_vec
            J_manip[3:6, i] = rot_axis
        return J_manip

    def momentOfInertia_transform(self, q=None, ang_s=None, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.kin.b0
        pv_com, _, _ = self.pos_vect_inertial(q=q, ang_s=ang_s, b0=b0)
        I = self.I
        j_T_full_inertial = self.transf_from_inertial(q=q, ang_s=ang_s, b0=b0)
        rot_full_inertial = j_T_full_inertial[:, 0:3, 0:3]
        manip_joint_axis = rot_full_inertial[2:-1, 0:3, 2].T  # manipulator joint axis
        tm = list(rot_full_inertial)
        tm.pop(1)  # removing robot_base coord system which is fixed on the satellite (do not rotate)
        rot_full_inertial = np.array(tm)  # rot_full = [0_R_s, 0_R_j1, 0_R_j2, ...].
        Ii = rot_full_inertial[0] @ I[0] @ rot_full_inertial[0].T
        tmp = pv_com[:, 0].T @ pv_com[:, 0]
        t0 = tmp * np.eye(3) - (pv_com[:, 0] @ pv_com[:, 0].T)
        Is = Ii + self.mass[0] * t0   # MOI of satellite transformed. Now wrt to inertial
        Im = np.zeros((3, self.nDoF))
        for i in range(1, self.nDoF + 1):
            Ii = rot_full_inertial[i] @ I[i] @ rot_full_inertial[i].T
            tmp = pv_com[:, i].T @ pv_com[:, i]
            t1 = tmp * np.eye(3) - (pv_com[:, i] @ pv_com[:, i].T)
            I_transformed = Ii + self.mass[i] * t1
            Im[:, i-1] = I_transformed @ manip_joint_axis[:, i-1]
        return Is, Im

    def generalized_jacobian(self, q=None, ang_s=None, b0=None):
        Jm = self.geometric_jacobian_manip(q=q, ang_s=ang_s, b0=b0)
        Js, _ = self.geometric_jacobian_satellite(q=q, ang_s=ang_s, b0=b0)
        Is, Im = self.momentOfInertia_transform(q=q, ang_s=ang_s, b0=b0)
        J = Jm - (Js @ np.linalg.solve(Is, Im))
        return J


class Solver(object):

    def __init__(self):
        pass

    def num_integration(self, *args):
        derivative, init_pos, t = args
        """ derivative is a 3 x t matrix whose integration is to be carried out """
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
    q0 = kin.q0
    # a, b, c = kin.pos_vect(q0)
    # s = dyn.spacecraft_com_pos(ang_s=ang_s0, q=q0)
    # a, b, c = dyn.pos_vect_inertial(q=q0, ang_s=ang_s0)
    J_s, Jac_sat = dyn.geometric_jacobian_satellite(q=q0, ang_s=ang_s0, b0=b0)
    Jac_manip = dyn.geometric_jacobian_manip(q=q0, ang_s=ang_s0, b0=b0)
    a = dyn.momentOfInertia_transform(q=q0, ang_s=ang_s0, b0=b0)
    gen_jacob = dyn.generalized_jacobian(q=q0, ang_s=ang_s0, b0=b0)
    # a, b, c = dyn.velocities(q=q0, ang_s=ang_s0, q_dot=0, b0=b0)

