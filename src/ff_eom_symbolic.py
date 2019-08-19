import numpy as np
from sympy import *
from sympy.physics.mechanics import *
from sympy.tensor.array import Array
from decimal import getcontext
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt
from sympy.physics.vector import ReferenceFrame, Vector
from sympy.physics.vector import time_derivative

# Set the precision.
getcontext().prec = 3
nDoF = 6
# msubs(omega, {self.ang_xs:0, self.ang_ys:0, self.ang_zs:0, self.ang_xb:0, self.ang_yb:0, self.ang_zb:0})
# Ms =  msubs(M, {kin.ang_xs:0, kin.ang_ys:0, kin.ang_zs:0, kin.ang_xb:0, kin.ang_yb:0, kin.ang_zb:0, kin.r_sx:0, kin.r_sy:0, kin.r_sz:0, kin.b0x:0, kin.b0y:0, kin.b0z:0, dyn.Is_xx:0, dyn.Is_yy:0, dyn.Is_zz:0, dyn.m[0]:0})


class kinematics():

    def __init__(self, nDoF=2):
        self.nDoF = nDoF
        self.qm, self.qdm, self.l, self.r = self.initializing(self.nDoF)  # qm = manipulator joint angles
                                    # qdm = manipulator joint velocities, l = link lengths, r = COM vectors from joints
        # DH parameters:
        self.a = Array([0, *self.l])
        self.d = Array([0.0, 0.0])
        self.alpha = Array([0.0, 0.0])

        self.r_sx, self.r_sy, self.r_sz, = dynamicsymbols('r_sx r_sy r_sz')  # r_s = satellite pos_vec wrt inertial
        self.ang_xs, self.ang_ys, self.ang_zs = dynamicsymbols("ang_xs ang_ys ang_zs ")
        self.r_sxd, self.r_syd, self.r_szd, = dynamicsymbols('r_sx r_sy r_sz', 1)  # satellite linear vel wrt inertial
        self.w_sxd, self.w_syd, self.w_szd = dynamicsymbols("ang_xs ang_ys ang_zs ", 1)  # satellite angular vel wrt inertial
        self.q = Matrix([self.r_sx, self.r_sy, self.r_sz, self.ang_xs, self.ang_ys, self.ang_zs, *self.qm])
        self.qd = [self.r_sxd, self.r_syd, self.r_szd, self.w_sxd, self.w_syd, self.w_szd, *self.qdm]

        self.rx = symbols(["r%d" % x for x in range(1, nDoF+1)])  # x component of COM vector of the links

        self.q_i, self.alpha_i, self.a_i, self.d_i = symbols("q_i alpha_i a_i d_i")
        self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z = symbols("ang_xb ang_yb ang_zb b0x b0y b0z")
        self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z = symbols("ang_x ang_y ang_z r0x, r0y, r0z")
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
            ang_x, ang_y, ang_z, r0x, r0y, r0z = self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z
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

    def fwd_kin_symbolic(self, q):
        T = self.robot_DH_matrix()
        T_joint, T_i_i1 = [], []  # T_i_i1 is the 4x4 transformation matrix relating i+1 frame to i
        t = eye(4)
        for i in range(len(q)):
            temp = msubs(T, {self.alpha_i: self.alpha[i], self.a_i: self.a[i], self.d_i: self.d[i], self.q_i: q[i]})
            t = t*temp
            T_joint.append(t)  # joint transformation matrix wrt base
            T_i_i1.append(temp)
        tmp = eye(4)
        tmp[0, 3] = self.a[-1]
        T_ee = t @ tmp
        T_joint.append(T_ee)
        return T_joint, T_i_i1

    def position_vectors(self,): # position vectors of COM of each link wrt inertial CS, {j}
        # {s}, {ji} are respectively the CS of spacecraft at its COM and joint CS of the manipulator
        # q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        j_T_s = self.euler_transformations([self.ang_xs, self.ang_ys, self.ang_zs, self.r_sx, self.r_sy, self.r_sz])
        s_T_j1 = self.euler_transformations([self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z])  # a constant 4 x 4 matrix
        j_T_j1 = j_T_s @ s_T_j1  # transformation from inertial to robot base
        T_joint, T_i_i1 = self.fwd_kin_symbolic(self.qm)  #
        j_T_full = []  # j_T_full is n x 4 x 4 transf. matrices # j_T_full = [0_T_s, 0_T_j0, 0_T_j1, 0_T_j2,..., 0_T_ee]
        # containing satellite, robot base and each of the joint CS
        j_T_full.extend([j_T_s, j_T_j1])
        pv_origins = zeros(3, self.nDoF+3)  # position vector of the origins of all coordinate system wrt inertial {j}
        pv_com = zeros(3, self.nDoF+1)  # position vector of the COM of spacecraft + each of the links wrt inertial {j}
        for i in range(2, 3+self.nDoF):
            j_T_full.append(j_T_j1 @ T_joint[i - 2])
        for i in range(self.nDoF+3):  # includes end-eff coordinate system
            pv_origins[:, i] = j_T_full[i][0:3, 3]
        kk = 1
        for i in range(2, len(j_T_full) - 1):
            trans_temp = pv_origins[:, i]
            rot_temp = j_T_full[i][0:3, 0:3]
            pv_com[:, i-1] = trans_temp[0] + 0.5 * self.a[kk] * rot_temp[0, 0], \
                             trans_temp[1] + 0.5 * self.a[kk] * rot_temp[1, 0], 0
            kk += 1
        return j_T_full, pv_origins, pv_com

    def ab_vectors(self):
        # assuming COM of links be exactly half of link lengths
        # a_i is the vector pointing from joint i to COM i described wrt the joint CS
        # b_i is the vector pointing from COM i to joint i+1 described wrt the COM CS
        a = zeros(3, self.nDoF)
        b = zeros(3, self.nDoF)
        b[:, 0] = transpose(Matrix([[self.b0x, self.b0y, self.b0z]]))
        for i in range(1, self.nDoF):
            b[0, i] = 0.5 * self.a[i]
        for i in range(0, self.nDoF):
            a[0, i] = 0.5 * self.a[i+1]
        return a, b

    def velocities(self):
        j_T_full, pv_origins, pv_com = self.position_vectors()
        omega = zeros(3, self.nDoF+2)
        joint_velocity = zeros(3, self.nDoF+2)
        com_vel = zeros(3, self.nDoF+1)
        b = Matrix([[self.b0x], [self.b0y], [self.b0z]])
        omega[:, 0] = Matrix([[self.w_sxd], [self.w_syd], [self.w_szd]])  # 0_w_s = ang vel of satellite wrt 0
        omega[:, 1] = Matrix([[self.w_sxd], [self.w_syd], [self.w_szd]])  # 0_w_j0 = ang vel of robot base
        joint_velocity[:, 0] = Matrix([[self.r_sxd], [self.r_syd], [self.r_szd]])  # satellite linear vel of COM
        joint_velocity[:, 1] = joint_velocity[:, 0] +\
                               omega[:, 1].cross((j_T_full[0][0:3, 0:3] @ b))  # lin vel of robot_base ({j0})
        joint_velocity[:, 2] = joint_velocity[:, 1]  # linear vel of {j1}
        com_vel[:, 0] = joint_velocity[:, 0]

        for i in range(2, 2+self.nDoF):
            temp = j_T_full[i][0:3, 2] * self.qdm[i - 2]
            omega[:, i] = omega[:, i-1] + temp
        for i in range(3, 3+self.nDoF - 1):  # not considering end-eff vel
            l = Matrix([[self.a[i-2]], [0], [0]])
            joint_velocity[:, i] = joint_velocity[:, i - 1] + omega[:, i].cross((j_T_full[i - 1][0:3, 0:3] @ l))
        for i in range(1, 1+self.nDoF):
            com_vel[:, i] = joint_velocity[:, i+1] + omega[:, i+1].cross((j_T_full[i+1][0:3, 0:3] @ self.r[i - 1]))
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

    def __init__(self, nDoF=2):
        self.nDoF = nDoF
        self.Is_xx, self.Is_yy, self.Is_zz = symbols('Is_xx, Is_yy, Is_zz')
        self.Ixx = symbols(["Ixx%d" % x for x in range(1, nDoF+1)])  # x component of MOI of the links about its COm
        self.Iyy = symbols(["Iyy%d" % x for x in range(1, nDoF+1)])  # y component of MOI of the links about its COm
        self.Izz = symbols(["Izz%d" % x for x in range(1, nDoF+1)])  # z component of MOI of the links about its COm
        self.m = symbols(["m%d" % x for x in range(nDoF+1)])   # mass of space-craft and each of the links
        self.tau, self.I = self.initializing(nDoF)

        self.kin = kinematics()
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

    def pos_vec_frm_momentum_consevation(self):
        a, b = self.kin.ab_vectors()
        j_T_full, _, _ = self.kin.position_vectors()  # j_T_full = [0_T_s, 0_T_j0, 0_T_j1, 0_T_j2, ......, 0_T_ee]
        j_T_full.remove(j_T_full[1])  # j_T_full = [0_T_s, 0_T_j1, 0_T_j2, ......, 0_T_ee]
        k11 = self.mass_frac()
        com_vec = zeros(3, self.nDoF+1)  # vector from inertial to COM of spacecraft + each of the links
        r0 = zeros(3, 1)
        for i in range(self.nDoF):
            Rb = j_T_full[i][0:3, 0:3]  # j_T_full = [0_T_s, 0_T_j1, 0_T_j2, ......, 0_T_ee]
            Ra = j_T_full[i+1][0:3, 0:3]
            r0 += k11[i] * (Rb @ b[:, i] + Ra @ a[:, i])
        com_vec[:, 0] = r0
        for i in range(1, nDoF+1):
            Rb = j_T_full[i-1][0:3, 0:3]  # j_T_full = [0_T_s, 0_T_j1, 0_T_j2, ......, 0_T_ee]
            Ra = j_T_full[i][0:3, 0:3]
            com_vec[:, i] = com_vec[:, i-1] + (Rb @ b[:, i-1] + Ra @ a[:, i-1])
        return com_vec

    def velocities_frm_momentum_consevation(self):
        a, b = self.kin.ab_vectors()
        j_T_full, _, _ = self.kin.position_vectors()  # j_T_full = [0_T_s, 0_T_j0, 0_T_j1, 0_T_j2, ......, 0_T_ee]
        j_T_full.remove(j_T_full[1])  # j_T_full = [0_T_s, 0_T_j1, 0_T_j2, ......, 0_T_ee]
        omega, _, _ = self.kin.velocities()  # omega = [0_w_s, 0_w_j0, 0_w_j1, ..., ] # no end-eff ang_vel here
        omega.col_del(1)  # omega = [0_w_s, 0_w_j1, 0_w_j2, ..., ]
        vel_com = zeros(3, self.nDoF+1)
        j_T_derivative = list()
        for i in range(self.nDoF+1):
            j_w_x = self.kin.skew_matrix(omega[:, i])  # skew symmetric matrix of omega.
            j_T_derivative.append(j_w_x @ j_T_full[i][0:3, 0:3])
        k11 = self.mass_frac()
        j_v_s = zeros(3, 1)
        for i in range(self.nDoF):
            j_v_s += k11[i] * (j_T_derivative[i] @ b[:, i] + j_T_derivative[i+1] @ a[:, i])  # spacecraft lin vel
        vel_com[:, 0] = j_v_s
        for i in range(1, nDoF+1):
            vel_com[:, i] = vel_com[:, i-1] + j_T_derivative[i-1] @ b[:, i-1] + j_T_derivative[i] @ a[:, i-1]
        return vel_com

    def kinetic_energy(self):
        j_T_full, _, _ = self.kin.position_vectors()
        # Transform MOI from COM CS to inertial
        R = j_T_full[0][0:3, 0:3]
        j_I = list()
        j_I.append(R @ self.I[0] @ R.T)
        for i in range(1, len(self.I)):
            R = j_T_full[i + 1][0:3, 0:3]
            j_I.append(R @ self.I[i] @ R.T)

        j_ang_vel, _, _ = self.kin.velocities()
        j_vel_com = self.velocities_frm_momentum_consevation()
        K = 0
        for i in range(self.nDoF + 1):
            K += 0.5*self.m[i]*j_vel_com[:, i].dot(j_vel_com[:, i]) + 0.5*j_ang_vel[:, i].dot(j_I[i] @ j_ang_vel[:, i])
        return K

    def get_dyn_para(self):
        K = self.kinetic_energy()
        q, qd = self.kin.q, self.kin.qd
        # P = self.potential_energy()
        L = K   # Lagrangian. Potential energy at space is insignificant (microgravity envrnt)
        M = transpose(Matrix([[K]]).jacobian(qd)).jacobian(qd) #.applyfunc(trigsimp)  # Mass matrix
        C = transpose(Matrix([[K]]).jacobian(qd)).jacobian(q) * Matrix(qd) - transpose(Matrix([[K]]).jacobian(q))  # Coriolis vector
        # C = C.applyfunc(trigsimp)
        # G = transpose(Matrix([[P]]).jacobian(q)).applyfunc(trigsimp)  # Gravity vector
        # LM = LagrangesMethod(L, q)
        # LM.form_lagranges_equations()
        # print LM.mass_matrix.applyfunc(trigsimp)
        # Matrix([P]).applyfunc(trigsimp)
        return M, C

    # def dyn_para_numeric(self, lp, qp, q_dot):
    #     M, C = self.M, self.C,
    #     for i in range(len(qp)):
    #         M = msubs(M, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i]})
    #         C = msubs(C, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i], self.kin.q[i].diff(): q_dot[i]})
    #         # G = msubs(G, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i], self.g: 9.81})
    #     return M, C,


if __name__ == '__main__':

    nDoF = 2
    kin = kinematics(nDoF=nDoF)
    dyn = dynamics(nDoF=nDoF)
    lp, qp, q_dot = [1, 1], [0, np.pi/2], [0.1, 0.2]
    j_T_full, pv_origins, pv_com = kin.position_vectors()
    ab_vec = kin.ab_vectors()
    r0 = dyn.pos_vec_frm_momentum_consevation()
    vel_com = dyn.velocities_frm_momentum_consevation()
    # T_joint, T_i_i1 = kin.fwd_kin_symbolic(qp)
    # omega, cm_vel, joint_velocity = kin.velocities()
    kin_energy = dyn.kinetic_energy()
    M, C = dyn.get_dyn_para()
    # M, C, G = dyn.get_dyn_para(kin.q, kin.qd)  # Symbolic dynamic parameters
    # M, C, G = dyn.dyn_para_numeric(lp, qp, q_dot)  # Numeric values dynamic parameters
    print('hi')

