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
        self.a = Array([0, self.l[0], self.l[1]])
        self.d = Array([0.0, 0.0])
        self.alpha = Array([0.0, 0.0])

        self.r_sx, self.r_sy, self.r_sz, = dynamicsymbols('r_sx r_sy r_sz')  # r_s = satellite pos_vec wrt inertial
        self.w_sx, self.w_sy, self.w_sz = dynamicsymbols('w_sx w_sy w_sz')  # w_s = satellite ang position wrt inertial
        self.r_sxd, self.r_syd, self.r_szd, = dynamicsymbols('r_sx r_sy r_sz', 1)  # satellite linear vel wrt inertial
        self.w_sxd, self.w_syd, self.w_szd = dynamicsymbols('w_sx w_sy w_sz', 1)  # satellite angular vel wrt inertial
        self.q = Matrix([self.r_sx, self.r_sy, self.r_sz, self.w_sx, self.w_sy, self.w_sz, *self.qm])
        self.qd = [self.r_sxd, self.r_syd, self.r_szd, self.w_sxd, self.w_syd, self.w_szd, *self.qdm]

        self.q_i = Symbol("q_i")
        self.alpha_i = Symbol("alpha_i")
        self.a_i = Symbol("a_i")
        self.d_i = Symbol("d_i")
        self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z = symbols("ang_x ang_y ang_z r0x, r0y, r0z")
        # self.b0 = np.array([0.2, 0.3, 0.])  # vector from the spacecraft COM to the base of the robot in spacecraft CS

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
        self.ang_xs, self.ang_ys, self.ang_zs, self.r0x, self.r0y, self.r0z = symbols("ang_xs ang_ys ang_zs r0x, r0y, r0z")
        self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z = symbols("ang_xb ang_yb ang_zb b0x b0y b0z")
        j_T_s = self.euler_transformations([self.ang_xs, self.ang_ys, self.ang_zs, self.r0x, self.r0y, self.r0z])
        s_T_j1 = self.euler_transformations([self.ang_xb, self.ang_yb, self.ang_zb, self.b0x, self.b0y, self.b0z])  # a constant 4 x 4 matrix
        j_T_j1 = j_T_s @ s_T_j1  # transformation from inertial to robot base
        T_joint, T_i_i1 = self.fwd_kin_symbolic(self.qm)  #
        j_T_full = []  # j_T_full is n x 4 x 4 transf. matrices
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


class dynamics():

    def __init__(self, nDoF=2):
        self.nDoF = nDoF
        self.tau, self.I = self.initializing(nDoF)
        self.g = symbols('g', positive=True)
        self.mn = [3, 1] #############################
        self.grav = transpose(Matrix([[0, self.g, 0]]))

        self.kin = kinematics()
        # self.M, self.C, self.G = self.get_dyn_para(self.kin.q, self.kin.qd)

    def initializing(self, nDoF):
        # q = dynamicsymbols('q:{0}'.format(nDoF))
        # r = dynamicsymbols('r:{0}'.format(nDoF))
        # for i in range(1, nDoF+1):
        Izz = symbols(["I%d" % x for x in range(1, nDoF+1)])  # z component of MOI of the links about its COm
        self.m = symbols(["m%d" % x for x in range(nDoF+1)])   # mass of space-craft and each of the links
        tau = symbols(["tau%d" % x for x in range(1, nDoF+1)])   # mass of space-craft and each of the links
        I = []
        [I.append(zeros(3)) for i in range(nDoF+1)]  # MOI matrix for the satellite and each of the links
        self.Is_xx, self.Is_yy, self.Is_zz = symbols('Is_xx, Is_yy, Is_zz')
        I[0][0, 0], I[0][1, 1], I[0][2, 2] = self.Is_xx, self.Is_yy, self.Is_zz
        for i in range(nDoF):
            I[i+1][2, 2] = Izz[i]
        return tau, I

    def kinetic_energy(self):
        j_T_full, _, _ = self.kin.position_vectors()
        # Transform MOI from COM CS to inertial
        R = j_T_full[0][0:3, 0:3]
        j_I = list()
        j_I.append(R @ self.I[0] @ R.T)
        for i in range(1, len(self.I)):
            R = j_T_full[i + 1][0:3, 0:3]
            j_I.append(R @ self.I[i] @ R.T)

        w, com_vel, _ = self.kin.velocities()
        K = 0
        for i in range(self.nDoF + 1):
            K += 0.5*self.m[i]*com_vel[:, i].dot(cm_vel[:, i]) + 0.5*w[:, i].dot(j_I[i] @ w[:, i])
        return K

    def get_dyn_para(self):
        K = self.kinetic_energy()
        q, qd = kin.q, kin.qd
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

    kin = kinematics()
    dyn = dynamics()
    lp, qp, q_dot = [1, 1], [0, np.pi/2], [0.1, 0.2]
    T_joint, T_i_i1 = kin.fwd_kin_symbolic(qp)
    j_T_full, pv_origins, pv_com = kin.position_vectors()
    omega, cm_vel, joint_velocity = kin.velocities()
    kin_energy = dyn.kinetic_energy()
    M, C = dyn.get_dyn_para()
    # M, C, G = dyn.get_dyn_para(kin.q, kin.qd)  # Symbolic dynamic parameters
    # M, C, G = dyn.dyn_para_numeric(lp, qp, q_dot)  # Numeric values dynamic parameters
    print('hi')

