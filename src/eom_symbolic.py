import numpy as np
from sympy import *
from sympy.physics.mechanics import *
from sympy.tensor.array import Array
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# msubs(omega, {self.ang_xs:0, self.ang_ys:0, self.ang_zs:0, self.ang_xb:0, self.ang_yb:0, self.ang_zb:0})
# Ms =  msubs(M, {kin.ang_xs:0, kin.ang_ys:0, kin.ang_zs:0, kin.ang_xb:0, kin.ang_yb:0, kin.ang_zb:0, kin.r_sx:0, kin.r_sy:0, kin.r_sz:0, kin.b0x:0, kin.b0y:0, kin.b0z:0, dyn.Is_xx:0, dyn.Is_yy:0, dyn.Is_zz:0, dyn.m[0]:0})


class kinematics():

    def __init__(self, nDoF=3, robot='3DoF'):
        self.nDoF = nDoF
        self.qm, self.qdm, self.l, self.r = self.initializing(self.nDoF)  # qm = manipulator joint angles
        # qdm = manipulator joint velocities, l = link lengths, r = COM vectors from joints
        # DH parameters:
        if robot == '3DoF':  # as given in umeneti and yoshida: resolved motion rate control of space manipulators
            self.l_numeric = Array([3.5, 0.25, 2.5, 2.5])
            # self.alpha = Array([-np.pi / 2, np.pi / 2, 0])
            # self.a = Array([0.0, 0.0, self.l[1]])
            # self.d = Array([self.l[0], 0.0, 0.0])
            self.ang_s0, self.ang_b = np.array([0., 0., 0.]), np.array([0.0, 0.0, 0.])
            self.r_s0 = np.array([0.01, 0.01, 0.0])
            # self.q0 = np.array([np.pi / 3 * 0, -3*np.pi / 4, -np.pi/2])
            self.q0 = np.array([pi / 3 * 0, pi / 2, 0])

            self.alpha = np.array([-pi / 2, pi / 2, 0.])
            self.a = np.array([0., 0., 1.5])
            self.d = np.array([1.0, 0., 0.])
            self.eef_dist = 1.20  # l3
        else:
            self.a = Array([0, *self.l])
            self.d = Array([0.0, 0.0])
            self.alpha = Array([0.0, 0.0])
        self.sizes = [(0.5, 0.5, 0.5)]  # satellite dimension
        self.b0 = np.array([0.5*self.sizes[0][0], 0., 0.])  # vector from spacecraft COM to robot base wrt spacecraft CS

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
        self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z = 0., 0., -pi/2, 0.5 * self.sizes[0][0], 0., 0.
        # self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z = symbols(
        #     "ang_xb ang_yb ang_zb b0x b0y b0z")
        # self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z = symbols("ang_x ang_y ang_z r0x, r0y, r0z")
        # self.b0 = np.array([0.2, 0.3, 0.])  # vector from the spacecraft COM to the base of the robot in spacecraft CS

    def skew_matrix(self, w):  # skew_symmetric matrix of the vector 'w'
        S = zeros(3, 3)
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
        cx, cy, cz = cos(ang_x), cos(ang_y), cos(ang_z)
        sx, sy, sz = sin(ang_x), sin(ang_y), sin(ang_z)
        T = Matrix([[cy*cz, -cy*sz, sy, r0x],
                    [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy, r0y],
                    [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy, r0z],
                    [0, 0, 0, 1]])
        return T

    def robot_DH_matrix(self):
        T = Matrix([[cos(self.q_i), -sin(self.q_i), 0, self.a_i],
                    [sin(self.q_i) * cos(self.alpha_i), cos(self.q_i) * cos(self.alpha_i), -sin(self.alpha_i), -sin(self.alpha_i) * self.d_i],
                    [sin(self.q_i) * sin(self.alpha_i), cos(self.q_i) * sin(self.alpha_i), cos(self.alpha_i), cos(self.alpha_i) * self.d_i],
                    [0, 0, 0, 1]])
        return T

    def fwd_kin_symbolic(self, q):   # forward kinematics of the manipulator alone
        T = self.robot_DH_matrix()
        T_joint, T_i_i1 = [], []  # T_i_i1 is the 4x4 transformation matrix relating i+1 frame to i
        t = eye(4)
        for i in range(len(q)):
            temp = msubs(T, {self.alpha_i: self.alpha[i], self.a_i: self.a[i], self.d_i: self.d[i], self.q_i: q[i]})
            t = t @ temp
            T_joint.append(t)  # joint transformation matrix wrt base
            T_i_i1.append(temp)
        tmp = eye(4)
        tmp[0, 3] = self.eef_dist
        T_i_i1.append(tmp)  # Adding eef transformation (basically an identity matrix with a translation)
        T_ee = t @ tmp
        T_joint.append(T_ee)
        return T_joint, T_i_i1

    def position_vectors(self,): # position vectors of COM of each link wrt inertial CS, {j}
        # {s}, {ji} are respectively the CS of spacecraft at its COM and joint CS of the manipulator
        # q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        j_T_s = self.euler_transformations([self.ang_xs, self.ang_ys, self.ang_zs, self.r_sx, self.r_sy, self.r_sz])
        s_T_j0 = self.euler_transformations([self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z])  # a constant 4 x 4 matrix
        j_T_j0 = j_T_s @ s_T_j0  # transformation from inertial to robot base
        T_joint, T_i_i1 = self.fwd_kin_symbolic(self.qm)  #
        j_T_full = []  # j_T_full is n x 4 x 4 transf. matrices # j_T_full = [0_T_s, 0_T_j0, 0_T_j1, 0_T_j2,..., 0_T_ee]
        # containing satellite, robot base and each of the joint CS
        j_T_full.extend([j_T_s, j_T_j0])
        pv_origins = zeros(3, self.nDoF+3)  # position vector of the origins of all coordinate system wrt inertial {j}
        pv_com = zeros(3, self.nDoF+1)  # position vector of the COM of spacecraft + each of the links wrt inertial {j}
        for i in range(2, 3+self.nDoF):
            j_T_full.append(j_T_j0 @ T_joint[i - 2])
        for i in range(self.nDoF+3):  # includes end-eff origin
            pv_origins[:, i] = j_T_full[i][0:3, 3]  # [0_r_s, 0_r_b, 0_r_j1, ...0_r_eef]
        kk = 1
        pv_com[:, 0] = pv_origins[:, 0]
        j_com_vec, ll = zeros(3, self.nDoF), 0
        for i in range(1, pv_origins.shape[1]-1):
            v = pv_origins[:, i+1] - pv_origins[:, i]
            if v[0] or v[1] or v[2]:
                pv_com[:, kk] = pv_origins[:, i] + 0.5 * v  # assuming COM exactly at the middle of the link
                j_com_vec[:, ll] = 0.5 * v  # vector from joint i to COM of link i described in inertial.
                # vector 'a' in Umeneti and Yoshida
                kk += 1
                ll += 1
        return j_T_full, pv_origins, pv_com, j_com_vec

    def rotations_from_inertial(self):
        j_T_full, _, _, _ = self.position_vectors()
        rot_full = list()
        for i in range(len(j_T_full)):
            rot_full.append(j_T_full[i][0:3, 0:3])  # rotation matrix of spacecraft COM + each joint CS wrt inertial
            # including end-eff (which is same as link n). rot_full = [0_R_s, 0_R_rb, 0_R_j1, 0_R_j2, ... 0_R_jeef].
            # rb = robot base or joint 0 {j0}
        return rot_full

    def ab_vectors(self, com_vec=None):
        j_T_full, _, _, j_com_vec = self.position_vectors()
        if not com_vec:
            a = j_com_vec
            b0 = j_T_full[0][0:3, 0:3] @ transpose(Matrix([[self.b0x, self.b0y, self.b0z]]))
            b = a.col_insert(0, b0)
        else:
            a, b = com_vec
        return a, b,

    def velocities(self):
        j_T_full, pv_origins, pv_com, j_com_vec = self.position_vectors()
        # j_T_full = [0_T_s, 0_T_j0, 0_T_j1, 0_T_j2,..., 0_T_ee]
        # j_com_vec =  vector from joint i to COM of link i wrt in inertial. vector 'a' in Umeneti and Yoshida
        omega = zeros(3, self.nDoF+2)
        joint_velocity = zeros(3, self.nDoF+2)
        com_vel = zeros(3, self.nDoF+1)
        b = Matrix([[self.b0x], [self.b0y], [self.b0z]])
        omega[:, 0] = Matrix([[self.w_sxd], [self.w_syd], [self.w_szd]])  # 0_w_s = ang vel of satellite wrt 0
        omega[:, 1] = Matrix([[self.w_sxd], [self.w_syd], [self.w_szd]])  # 0_w_j0 = ang vel of robot base
        for i in range(2, 2+self.nDoF):
            temp = j_T_full[i][0:3, 2] * self.qdm[i - 2]
            omega[:, i] = omega[:, i-1] + temp

        joint_velocity[:, 0] = Matrix([[self.r_sxd], [self.r_syd], [self.r_szd]])  # satellite linear vel of COM
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


class dynamics():

    def __init__(self, nDoF=3, robot='3DoF'):
        self.nDoF = nDoF
        self.Is_xx, self.Is_yy, self.Is_zz = symbols('Is_xx, Is_yy, Is_zz')
        self.Ixx = symbols(["Ixx%d" % x for x in range(1, self.nDoF+1)])  # x component of MOI of the links about its COm
        self.Iyy = symbols(["Iyy%d" % x for x in range(1, self.nDoF+1)])  # y component of MOI of the links about its COm
        self.Izz = symbols(["Izz%d" % x for x in range(1, self.nDoF+1)])  # z component of MOI of the links about its COm
        self.m = symbols(["m%d" % x for x in range(self.nDoF+1)])   # mass of space-craft and each of the links
        self.tau, self.I = self.initializing(self.nDoF)
        self.I_full = [self.Is_xx, self.Is_yy, self.Is_zz, *self.Ixx, *self.Iyy, *self.Izz]

        self.kin = kinematics()

        # numeric values
        self.mass = Array([2000.0, 20.0, 50.0, 50.0])  # mass of satellite and each of the links respec
        self.Is = Matrix([[1400.0, 0.0, 0.0], [0.0, 1400.0, 0.0], [0.0, 0.0, 2040.0]])
        self.I1 = Matrix([[0.10, 0.0, 0.0], [0.0, 0.10, 0], [0.0, 0.0, 0.10]])
        self.I2 = Matrix([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
        self.I3 = Matrix([[0.25, 0.0, 0.0], [0.0, 26.0, 0], [0.0, 0.0, 26.0]])
        self.I_numeric = np.array([1400.0, 1400.0, 2040.0, 0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])

        # self.M, self.C, self.G = self.get_dyn_para(self.kin.q, self.kin.qd)

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
        M = sum(self.m)
        k11 = zeros(1, self.nDoF)
        for j in range(1, self.nDoF+1):
            k11[j-1] = (-1 / M) * sum(self.m[j:])
        return k11

    def com_pos_vect(self):
        # rs, r1, r2 etc are pv from inertial to COM of spacecraft, link1, lin2, ...
        k11 = self.mass_frac()
        j_T_full, _, _, j_com_vec = self.kin.position_vectors()
        aa, bb = self.kin.ab_vectors()
        r0 = zeros(3, 1)
        pv_com = zeros(3, self.nDoF + 1)  # matrix of pos vec frm system COM to COM of spacecraft + each of the links
        for i in range(self.nDoF):
            r0 += k11[i] * (bb[:, i] + aa[:, i])
            # r0 += k11[i] * (bb[:, i] + aa[:, i])
        pv_com[:, 0] = r0
        for i in range(1, self.nDoF + 1):
            pv_com[:, i] = pv_com[:, i - 1] + bb[:, i - 1] + aa[:, i - 1]
            # temp.applyfunc(simplify)
        return pv_com  # [j_rs, j_r1, j_r2, ...]

    def velocities_frm_momentum_conservation(self):
        j_omega, _, _ = self.kin.velocities()
        j_T_full, pv_origins, pv_com, j_com_vec = self.kin.position_vectors()
        k = list()
        omega = j_omega
        # To find out which all joints have same origins. Refer notebook picture. q1 rotation does not change a1
        # as it is on the rotation axis and hence there is no change in COM bcoz of q1_dot
        for i in range(pv_origins.shape[1] - 1):
            tm = pv_origins[:, i+1] - pv_origins[:, i]
            if not (tm[0] or tm[1] or tm[2]):
                k.append(i)
        k = np.array(k)
        if np.any(k):
            for k in k:
                omega.col_del(k)  # w = [0_w_s, 0_w_b, 0_w_j2, ...], 0_w_j1 deleted as this and origin j2 coincides
        # pv_com = self.com_pos_vect()
        # omega = zeros(3, self.nDoF + 1)  # matrix containing ang vel of spacecraft + each of the links wrt inertial
        # omega[:, 0] = Matrix([[self.kin.w_sxd], [self.kin.w_syd], [self.kin.w_szd]])
        # rot_full = self.kin.rotations_from_inertial()  # rot_full = [0_R_s, 0_R_rb, 0_R_j1, 0_R_j2, ...0_R_jeef]
        # for i in range(1, self.nDoF + 1):
        #     temp = rot_full[i + 1][:, 2] * self.kin.qdm[i - 1]
        #     omega[:, i] = omega[:, i - 1] + temp  # j_w = [0_w_s, 0_w_j1, 0_w_j2, ...]
        omega_skew_sym = list()
        for i in range(omega.shape[1]):
            omega_skew_sym.append(self.kin.skew_matrix(omega[:, i]))

        j_vel_com = zeros(3, self.nDoF + 1)  # matrix of linear vel of spacecraft + each of the links wrt inertial
        k11 = self.mass_frac()
        rs_d = zeros(3, 1)
        aa, bb = self.kin.ab_vectors()
        for i in range(self.nDoF):
            rs_d += k11[i] * (omega_skew_sym[i] @ bb[:, i] + omega_skew_sym[i + 1] @ aa[:, i])
        j_vel_com[:, 0] = rs_d
        for i in range(1, self.nDoF + 1):
            temp = omega_skew_sym[i - 1] @ bb[:, i - 1] + omega_skew_sym[i] @ aa[:, i - 1]
            j_vel_com[:, i] = j_vel_com[:, i - 1] + temp  # [0_v_s, 0_v_com1, 0_v_com2, ...]
        j_omega, _, _ = self.kin.velocities()  # this is called again since there is a bug in sympy. omega.col_del()
        # deletes the same column in j_omega
        return j_omega, j_vel_com

    def linear_momentum_conservation(self):
        j_omega, j_vel_com = self.velocities_frm_momentum_conservation()
        L = zeros(3, 1)
        for i in range(self.nDoF+1):
            L += self.m[i] * j_vel_com[:, i]
        return L

    def momentOfInertia_transform(self):
        j_T_full, pv_origins, pv_com, j_com_vec = self.kin.position_vectors()
        I = self.I
        rot_full = list()
        for i in range(len(j_T_full)):
            rot_full.append(j_T_full[i][0:3, 0:3])  # rotation matrix of spacecraft COM + each joint CS wrt inertial
            # including end-eff (which is same as link n). rot_full = [0_R_s, 0_R_rb, 0_R_j1, 0_R_j2, ...].
            # rb = robot base or joint 0 {j0}
        rot_full.remove(rot_full[1])  # rot_full = [0_R_s, 0_R_j1, 0_R_j2, ...].
        I_transformed = list()
        for i in range(self.nDoF + 1):
            Ii = rot_full[i] @ I[i] @ rot_full[i].T
            tmp = pv_com[:, i].T @ pv_com[:, i]
            t1 = tmp[0] * eye(3) - (-pv_com[:, i] @ -pv_com[:, i].T)
            I_transformed.append(Ii + self.m[i] * t1)
        return I_transformed

    def ang_momentum_conservation(self):
        I = self.momentOfInertia_transform()
        j_omega, j_vel_com = self.velocities_frm_momentum_conservation()
        pv_com = self.com_pos_vect()
        L = zeros(3, 1)
        for i in range(self.nDoF + 1):
            L += I[i] @ j_omega[:, i]  # + self.m[i] * pv_com[:, i].cross(j_vel_com[:, i])
            # The second term is not required since the coordinate system considered is at COM
            # I_o = I_com + I_/com
        return L

    def calculate_spacecraft_ang_vel(self, m, l, I, b, ang_s0, ang_b0, r_s0, q0, qdm_numeric):
        L = self.ang_momentum_conservation()
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        L_num = L
        for i in range(len(m)):
            L_num = msubs(L_num, {self.m[i]: m[i]})
        for i in range(len(I)):
            L_num = msubs(L_num, {self.I_full[i]: I[i]})
        for i in range(len(q0)):
            L_num = msubs(L_num, {self.kin.qm[i]: q0[i]})
        L_num = msubs(L_num, {self.kin.b0x: b[0], self.kin.b0y: b[1], self.kin.b0z: b[2], self.kin.ang_xs: ang_s0[0], self.kin.ang_ys: ang_s0[1],
                              self.kin.ang_zs: ang_s0[2], self.kin.ang_xb: ang_b0[0], self.kin.ang_yb: ang_b0[1],
                              self.kin.ang_zb: ang_b0[2], self.kin.r_sx: r_s0[0], self.kin.r_sy: r_s0[1],
                              self.kin.r_sz: r_s0[2]})
        for i in range(len(l)):
            L_num = msubs(L_num, {self.kin.l[i]: l[i]})
        Ls, Lm = L_num.jacobian(qd_s), L_num.jacobian(qd_m)
        Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)
        shp = qdm_numeric.shape[1]
        omega_s = np.zeros((3, shp))
        for i in range(shp):
            omega_s[:, i] = np.linalg.solve(Ls, (Lm @ qdm_numeric[:, i]))
        return omega_s

    def calculate_spacecraft_lin_vel(self,  m, l, I, b, ang_s0, ang_b0, r_s0, q0, qdm_numeric):
        omega_s = self.calculate_spacecraft_ang_vel(m, l, I, b, ang_s0, ang_b0, r_s0, q0, qdm_numeric)
        shp = omega_s.shape[1]
        j_omega, j_vel_com = self.velocities_frm_momentum_conservation()

        j_vel_com_num = j_vel_com
        for i in range(len(m)):
            j_vel_com_num = msubs(j_vel_com_num, {self.m[i]: m[i]})
        for i in range(len(l)):
            j_vel_com_num = msubs(j_vel_com_num, {self.kin.l[i]: l[i]})
        for i in range(len(I)):
            j_vel_com_num = msubs(j_vel_com_num, {self.I_full[i]: I[i]})
        for i in range(len(q0)):
            j_vel_com_num = msubs(j_vel_com_num, {self.kin.qm[i]: q0[i]})
        j_vel_com_num = msubs(j_vel_com_num, {self.kin.b0x: b[0], self.kin.b0y: b[1], self.kin.b0z: b[2],
                                              self.kin.ang_xs: ang_s0[0], self.kin.ang_ys: ang_s0[1],
                                              self.kin.ang_zs: ang_s0[2], self.kin.ang_xb: ang_b0[0], self.kin.ang_yb: ang_b0[1],
                                              self.kin.ang_zb: ang_b0[2], self.kin.r_sx: r_s0[0], self.kin.r_sy: r_s0[1],
                                              self.kin.r_sz: r_s0[2]})


        v_com = np.zeros((shp, 3, self.nDoF+1))
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        vcm = j_vel_com_num
        for j in range(shp):
            for i in range(len(qd_s)):
                j_vel_com_num = msubs(j_vel_com_num, {qd_s[i]: omega_s[i, j]})
            for k in range(len(qd_m)):
                j_vel_com_num = msubs(j_vel_com_num, {qd_m[k]: qdm_numeric[k, j]})
            v_com[j, :, :] = j_vel_com_num
            j_vel_com_num = vcm
        return v_com

    def kinetic_energy(self):
        j_I = self.momentOfInertia_transform()
        w, com_vel, _ = self.kin.velocities()
        K = 0
        for i in range(self.nDoF + 1):
            K += 0.5*self.m[i]*com_vel[:, i].dot(com_vel[:, i]) + 0.5*w[:, i].dot(j_I[i] @ w[:, i])
        return K

    def get_dyn_para(self):
        K = self.kinetic_energy()
        q, qd = self.kin.q, self.kin.qd
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

    def get_positions(self):
        solver = Solver()
        m, I = self.mass, self.I_numeric
        l = self.kin.l_numeric[1:]  # cutting out satellite length l0
        r_s0, ang_s0, ang_b, q0, b0 = self.kin.r_s0, self.kin.ang_s0, self.kin.ang_b, self.kin.q0, self.kin.b0
        t = np.linspace(0, 60, 91)
        dt = t[1] - t[0]
        ss = np.floor(len(t) / 3)
        cc = np.pi/60
        y1, y2, y3 = np.linspace(0, cc, ss), np.ones(int(ss)) * cc, np.linspace(cc, 0, ss)
        q1_dot = np.zeros(len(t) - 1)
        q2_dot = np.hstack((y1, y2, y3))
        q3_dot = np.zeros(len(t) - 1)
        qdm_numeric = np.vstack((q1_dot, q2_dot, q3_dot))

        omega_s = self.calculate_spacecraft_ang_vel(m, l, I, b0, ang_s0, ang_b, r_s0, q0, qdm_numeric)
        v_com = self.calculate_spacecraft_lin_vel(m, l, I, b0, ang_s0, ang_b, r_s0, q0, qdm_numeric)
        vs = v_com[:, :, 0]
        vs = np.transpose(vs)
        v_com1 = v_com[:, :, 1]
        v_com2 = v_com[:, :, 2]
        # self.kin.q[i].diff(): q_dot[i]}
        r_s = solver.numerical_integration(vs, r_s0, dt)  # position vector of spacecraft COM as a function of time
        r_s = np.c_[r_s0.reshape(3, -1), r_s]
        qm_numeric = solver.numerical_integration(qdm_numeric, q0, dt)
        q = np.c_[q0, qm_numeric]
        ang_s = solver.numerical_integration(omega_s, ang_s0,
                                             dt)  # angular position of spacecraft COM as a function of time
        ang_s = np.c_[ang_s0.reshape(3, -1), ang_s]
        return r_s, ang_s, q, qdm_numeric, t


class Solver(object):

    def __init__(self):
        pass

    def numerical_integration(self, *args):
        derivative, init_pos, dt = args
        # derivative is a 3 x t matrix whose integration is to be carried out
        shp = derivative.shape
        integrand = np.zeros(shp)
        for i in range(shp[1]):
            integrand[:, i] = derivative[:, i] * dt + init_pos
            init_pos = integrand[:, i]
        return integrand


if __name__ == '__main__':

    nDoF = 3
    kin = kinematics(nDoF=nDoF, robot='3DoF')
    dyn = dynamics(nDoF=nDoF, robot='3DoF')
    solver = Solver()

    # T_joint, T_i_i1 = kin.fwd_kin_symbolic(kin.qm)
    # j_T_full, pv_origins, pv_com, j_com_vec = kin.position_vectors()
    # a, b = kin.ab_vectors()
    # omega, cm_vel, joint_velocity = kin.velocities()
    # com_pv = dyn.com_pos_vect()
    omega, vel_com = dyn.velocities_frm_momentum_conservation()
    # omega, com_vel, joint_velocity = kin.velocities()

    # kin_energy = dyn.kinetic_energy()
    # M, C = dyn.get_dyn_para()
    # M, C, G = dyn.get_dyn_para(kin.q, kin.qd)  # Symbolic dynamic parameters
    # M, C, G = dyn.dyn_para_numeric(lp, qp, q_dot)  # Numeric values dynamic parameters
    print('hi')