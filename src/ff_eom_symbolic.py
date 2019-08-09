import numpy as np
from sympy import *
from sympy.physics.mechanics import *
from sympy.tensor.array import Array
from decimal import getcontext
# import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize as opt
from sympy.physics.vector import ReferenceFrame, Vector
from sympy.physics.vector import time_derivative


# Set the precision.
getcontext().prec = 3
nDoF = 6


class kinematics():

    def __init__(self):
        self.q1, self.q2 = dynamicsymbols('q1 q2')  # r_s = satellite pos_vec from inertial, qi=joint ang
        self.r_sx, self.r_sy, self.r_sz, = dynamicsymbols('r_sx r_sy r_sz')  # r_s = satellite pos_vec wrt inertial
        self.w_sx, self.w_sy, self.w_sz =  dynamicsymbols('w_sx w_sy w_sz')  # w_s = satellite ang position wrt inertial
        self.q1d, self.q2d = dynamicsymbols('q1 q2', 1)
        self.r_sxd, self.r_syd, self.r_szd, = dynamicsymbols('r_sx r_sy r_sz', 1)  # satellite vel wrt inertial
        self.w_sxd, self.w_syd, self.w_szd = dynamicsymbols('w_sx w_sy w_sz', 1)  # satellite ang vel wrt inertial
        self.q = [self.r_sx, self.r_sy, self.r_sz, self.w_sx, self.w_sy, self.w_sz, self.q1, self.q2]
        self.qd = [self.r_sxd, self.r_syd, self.r_szd, self.w_sxd, self.w_syd, self.w_szd, self.q1d, self.q2d]
        self.l1, self.l2 = symbols('l_1 l_2', positive=True)
        self.l = [self.l1, self.l2]
        self.ln = [1.5, 1.0]  ############################

        # COM vectors
        # self.r1, self.r2 = symbols('r1 r2')
        # self.r11 = zeros(3, 1)
        # # self.r11[0] = self.r1
        # self.r11[0] = 1  ################################
        # self.r22 = zeros(3, 1)
        # # self.r22[0] = self.r2
        # self.r22[0] = 1  #################################
        # self.r = zeros(3, 2)
        # self.r[:, 0] = self.r11
        # self.r[:, 1] = self.r22

        self.a = Array([0, self.l[0], self.l[1]])
        self.d = Array([0.0, 0.0])
        self.alpha = Array([0.0, 0.0])
        self.T_eff = eye(4)
        self.T_eff[0, 3] = self.l[-1]

        self.q_i = Symbol("q_i")
        self.alpha_i = Symbol("alpha_i")
        self.a_i = Symbol("a_i")
        self.d_i = Symbol("d_i")
        self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z = symbols("ang_x ang_y ang_z r0x, r0y, r0z")

    def euler_transformations(self):
        ang_x, ang_y, ang_z, r0x, r0y, r0z = self.ang_x, self.ang_y, self.ang_z, self.r0x, self.r0y, self.r0z
        cx, cy, cz = cos(ang_x), cos(ang_y), cos(ang_z)
        sx, sy, sz = sin(ang_x), sin(ang_y), sin(ang_z)
        T = Matrix([[cy*cz, -cy*sz, sy, r0x],
                    [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy, r0y],
                    [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy, r0z],
                    [0, 0, 0, 1]])
        return T

    def fwd_kin_symbolic(self, q):
        T = Matrix([[cos(self.q_i), -sin(self.q_i), 0, self.a_i],
                    [sin(self.q_i) * cos(self.alpha_i), cos(self.q_i) * cos(self.alpha_i), -sin(self.alpha_i), -sin(self.alpha_i) * self.d_i],
                    [sin(self.q_i) * sin(self.alpha_i), cos(self.q_i) * sin(self.alpha_i), cos(self.alpha_i), cos(self.alpha_i) * self.d_i],
                    [0, 0, 0, 1]])
        T_joint, T_i_i1 = [], []  # T_i_i1 is the 4x4 transformation matrix relating i+1 frame to i
        t = eye(4)
        for i in range(len(q)):
            temp = T.subs(self.alpha_i, self.alpha[i]).subs(self.a_i, self.a[i]).subs(self.d_i, self.d[i]).subs(self.q_i, q[i])
            t = t*temp
            T_joint.append(t)
            T_i_i1.append(temp)
        return T_joint, T_i_i1

    def fwd_kin_numeric(self, lp, qp):  # provide the values of link lengths and joint angles to get the end-eff pose
        T_joint, _ = self.fwd_kin_symbolic(self.q)
        T_0_eff = T_joint[-1] * self.T_eff
        # qp, lp = [0, np.pi/2], [1, 1]
        for i in range(len(self.q)):
            T_0_eff = T_0_eff.subs(self.q[i], qp[i]).subs(self.l[i], lp[i])
        Rot_0_eff = T_0_eff[0:3, 0:3]
        pos_0_eff = T_0_eff[0:3, 3]
        return pos_0_eff, Rot_0_eff

    def inv_kin(self, X):
        a = X[0]**2 + X[1]**2 - self.ln[0]**2 - self.ln[1]**2
        b = 2 * self.ln[0] * self.ln[1]
        q2 = np.arccos(a/b)
        c = np.arctan2(X[1], X[0])
        q1 = c - np.arctan2(self.ln[1] * np.sin(q2), (self.ln[0] + self.ln[1]*np.cos(q2)))
        return q1, q2

    def inv_kin_optfun(self, q):
        # a = np.array(two_R.end_effec_pose(q[:, i])).astype(np.float64)
        pos_0_eff, _ = self.fwd_kin_numeric(self.l, q)
        pos_0_eff = np.array(pos_0_eff[0:2]).astype(np.float64)
        k = pos_0_eff.reshape(-1) - self.T_desired
        # k = np.reshape(k, (16, 1))
        k = k.transpose() @ k
        return k

    def inv_kin2(self, q_current, T_desired):  # Implements Min. (F(q) - T_desired) = 0
        x0 = q_current
        self.T_desired = T_desired
        final_theta = opt.minimize(self.inv_kin_optfun, x0,
                                   method='BFGS', )  # jac=self.geometric_jacobian(T_joint, T_current))
        # print 'res \n', final_theta
        # final_theta = np.insert(final_theta.x, self.numJoints, 0)  # Adding 0 at the end for the fixed joint

        return final_theta.x

    def velocities(self, q):
        omega = Matrix.zeros(3, len(q)+1)
        joint_velocity = Matrix.zeros(3, len(q)+1)
        cm_vel = Matrix.zeros(3, len(q))
        _, t_i_i1 = self.fwd_kin_symbolic(q)  # T_i_i1 is the 4x4 transformation matrix of i+1 frame wrt to i
        for i in range(len(q)):
            R = t_i_i1[i][0:3, 0:3].transpose()
            omega[:, i+1] = R * omega[:, i] + Matrix([[0], [0], [self.q[i]]])
            # omega[:, i+1] = R * omega[:, i] + Matrix([[0], [0], [self.qd[i]]])
            joint_velocity[:, i+1] = R * (joint_velocity[:, i] + omega[:, i].cross(t_i_i1[i][0:3, 3]))
        omega, joint_velocity = omega[:, 1:], joint_velocity[:, 1:]
        for i in range(len(q)):
            # cm_vel[:, i] = joint_velocity[:, i] + omega[:, i].cross(t_i_i1[i][0:3, 3]/2)
            cm_vel[:, i] = joint_velocity[:, i] + omega[:, i].cross(self.r[:, i])
        return omega, cm_vel, joint_velocity


class dynamics():

    def __init__(self):
        self.tau_1, self.tau_2, self.I1_zz, self.I2_zz, self.m1, self.m2 = symbols('tau_1 tau_2 I1_zz, I2_zz, m1, m2')
        self.g = symbols('g', positive=True)
        # self.m = [self.m1, self.m2]
        self.m = [3, 1] #############################
        self.grav = transpose(Matrix([[0, self.g, 0]]))

        # Inertia tensor wrt centre of mass of each link
        self.I1 = zeros(3, 3)
        # self.I1[2, 2] = self.I1_zz
        self.I1[2, 2] = 2   ###########################
        self.I2 = zeros(3, 3)
        # self.I2[2, 2] = self.I2_zz
        self.I2[2, 2] = 1.5   ##########################
        self.I = [self.I1, self.I2]

        self.kin = kinematics()
        self.M, self.C, self.G = self.get_dyn_para(self.kin.q, self.kin.qd)

    def kinetic_energy(self, q):
        w, cm_vel, _ = self.kin.velocities(q)
        K = 0
        for i in range(len(q)):
            K += 0.5*self.m[i]*cm_vel[:, i].dot(cm_vel[:, i]) + 0.5*w[:, i].dot(self.I[i]*w[:, i])
        return K

    def potential_energy(self, q):
        T_joint, _ = self.kin.fwd_kin_symbolic(q)  # T_joint is the 4x4 transformation matrix relating i_th frame  wrt to 0
        P = 0
        for i in range(len(q)):
            r_0_cm = T_joint[i][0:3, 0:3]*self.kin.r[:, i] + T_joint[i][0:3, 3]
            P += self.m[i]*self.grav.dot(r_0_cm)
        return P

    def get_dyn_para(self, q, qd):
        K = self.kinetic_energy(q)
        P = self.potential_energy(q)
        L = K - P  # Lagrangian
        M = transpose(Matrix([[K]]).jacobian(qd)).jacobian(qd).applyfunc(trigsimp)  # Mass matrix
        C = transpose(Matrix([[K]]).jacobian(qd)).jacobian(q) * Matrix(qd) - transpose(Matrix([[K]]).jacobian(q))  # Coriolis vector
        C = C.applyfunc(trigsimp)
        G = transpose(Matrix([[P]]).jacobian(q)).applyfunc(trigsimp)  # Gravity vector
        # LM = LagrangesMethod(L, q)
        # LM.form_lagranges_equations()
        # print LM.mass_matrix.applyfunc(trigsimp)
        # Matrix([P]).applyfunc(trigsimp)
        return M, C, G

    def dyn_para_numeric(self, lp, qp, q_dot):
        M, C, G = self.M, self.C, self.G
        for i in range(len(qp)):
            M = msubs(M, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i]})
            C = msubs(C, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i], self.kin.q[i].diff(): q_dot[i]})
            G = msubs(G, {self.kin.q[i]: qp[i], self.kin.l[i]: lp[i], self.g: 9.81})
        return M, C, G

    def round2zero(self, m, e):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if (isinstance(m[i,j], Float) and m[i, j] < e):
                    m[i, j] = 0


if __name__ == '__main__':

    kin = kinematics()
    # dyn = dynamics()
    lp, qp, q_dot = [1, 1], [0, np.pi/2], [0.1, 0.2]
    # M, C, G = dyn.get_dyn_para(kin.q, kin.qd)  # Symbolic dynamic parameters
    # M, C, G = dyn.dyn_para_numeric(lp, qp, q_dot)  # Numeric values dynamic parameters
    print('hi')

