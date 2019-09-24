import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
from eom_symbolic import dynamics, kinematics
from sympy import *
from sat_manip_simulation import Simulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=3)


class InverseKinematics():

    def __init__(self, nDoF, robot='3DoF'):
        self.t = Symbol('t')
        self.nDoF = nDoF
        self.kin = kinematics(nDoF, robot)
        self.dyn = dynamics(nDoF, robot)
        self.sat_manip_sim = Simulation()
        self.m, self.I, self.l = self.dyn.mass, self.dyn.I_numeric, self.kin.l_numeric[
                                                                    1:]  # cutting out satellite length l0
        self.ang_b0, self.b0 = self.kin.ang_b, self.kin.b0
        self.ang_s0 = self.kin.ang_s0
        pv_com, pv_eef, _ = self.dyn.com_pos_vect()
        self.pv_com_num = self.dyn.substitute(pv_com, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.pv_eef = self.dyn.substitute(pv_eef, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        L = self.dyn.ang_momentum_conservation()
        self.L_num = self.dyn.substitute(L, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        j_omega, _, j_vel_eef = self.dyn.velocities_frm_momentum_conservation()
        self.omega_eef = self.dyn.substitute(j_omega[:, -1], m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.vel_eef = self.dyn.substitute(j_vel_eef, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.qd = self.kin.qd[3:]
        self.qd_s, self.qd_m = self.qd[0:3], self.qd[3:]
        self.J_p = self.vel_eef.jacobian(self.qd)
        self.J_w = self.omega_eef.jacobian(self.qd)
        j_T_full = self.kin.fwd_kin_symb_full()
        self.R_eef = j_T_full[-1][0:3, 0:3]  # rotation matrix of the eef CS wrt inertial
        # t1 = diff(self.R_eef, self.t)
        # sh = t1.shape
        # l = list()
        # for i in range(t1.shape[0]):
        #     a1 = t1[i, :].jacobian(Matrix(self.qd))
        #     l.append(a1)
        # rt = Matrix(l)
        # t2 = t1.reshape(sh[0]*sh[1], 1)
        # J_R = t2.jacobian(self.qd)
        # self.J_R = J_R.reshape(sh[0], sh[1])

        # self.J_star = self.generalized_jacobian_symb()

    def euler_transformations(self, *args):
        ang_x, ang_y, ang_z, r0x, r0y, r0z = args
        cx, cy, cz = cos(ang_x), cos(ang_y), cos(ang_z)
        sx, sy, sz = sin(ang_x), sin(ang_y), sin(ang_z)
        T = Matrix([[cy*cz, -cy*sz, sy, r0x],
                    [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy, r0y],
                    [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy, r0z],
                    [0, 0, 0, 1]])
        return T

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def spacecraft_com_num(self, ang_s, q):
        r_s0 = self.dyn.substitute(self.pv_com_num[:, 0], ang_s0=ang_s, q0=q)
        r_s0 = np.array(r_s0).astype(np.float64)
        r_s0 = r_s0.reshape((3, 1))
        return r_s0

    def manip_eef_pos_num(self, ang_s, q):
        eef_pos = self.dyn.substitute(self.pv_eef, ang_s0=ang_s, q0=q)
        eef_pos = np.array(eef_pos).astype(np.float64)
        return eef_pos

    def manip_eef_rotmat_num(self, ang_s, q):
        eef_rotmat = self.dyn.substitute(self.R_eef, ang_s0=ang_s, q0=q)
        eef_rotmat = np.array(eef_rotmat).astype(np.float64)
        return eef_rotmat

    def path(self, eef_des_pos, q0):  # q0 is current joint angles which is used to calculate current end_eff position
        pv_eef = self.manip_eef_pos_num(self.ang_s0, q0)
        init_pos = np.squeeze(pv_eef)
        points = self.discretize(init_pos, eef_des_pos, step_size=0.45)  # step_size is inversely proportional to vel
        points = np.insert(points, 0, init_pos, axis=0)
        points = np.insert(points, len(points), eef_des_pos, axis=0)
        return points

    # Method 1: Fails
    def jacobians(self, ang_s, q):
        omega_eef_num = self.dyn.substitute(self.omega_eef, ang_s0=ang_s, q0=q)
        Jw_s, Jw_m = omega_eef_num.jacobian(self.qd_s), omega_eef_num.jacobian(
            self.qd_m)  # Jv_s and Jv_m are both functions of ang_s, q_i
        Jw_s, Jw_m = np.array(Jw_s).astype(np.float64), np.array(Jw_m).astype(np.float64),

        vel_eef_num = self.dyn.substitute(self.vel_eef, ang_s0=ang_s, q0=q)
        Jv_s, Jv_m = vel_eef_num.jacobian(self.qd_s), vel_eef_num.jacobian(
            self.qd_m)  # Jv_s and Jv_m are both functions of ang_s, q_i
        Jv_s, Jv_m = np.array(Jv_s).astype(np.float64), np.array(Jv_m).astype(np.float64),
        return Jv_s, Jv_m, Jw_s, Jw_m

    def generalized_jacobian(self, ang_s, q):
        Jv_s, Jv_m, Jw_s, Jw_m = self.jacobians(ang_s, q)
        L_num = self.dyn.substitute(self.L_num, ang_s0=ang_s, q0=q)
        Ls, Lm = L_num.jacobian(self.qd_s), L_num.jacobian(self.qd_m)
        Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)
        a1 = np.linalg.solve(Ls, Lm)
        Jv = Jv_m - Jv_s @ a1
        Jw = Jw_m - Jw_s @ a1
        return Jv, Jw

    def laplace_cost_and_grad(self, X, r_eef_des, eul_ang_des):
        ang_s, q = np.array([0., 0., X[0]]), X[1:]
        # ang_s, q = X[0:3], X[3:]
        Wr, Ww = np.eye(3), np.eye(3)
        pv_eef = self.manip_eef_pos_num(ang_s, q)
        diff1 = np.squeeze(pv_eef) - r_eef_des
        tmp1 = np.dot(Wr, diff1)

        eef_rotmat = self.manip_eef_rotmat_num(ang_s, q)
        r = R.from_dcm(eef_rotmat)
        angles = r.as_euler('xyz',)
        diff2 = eul_ang_des - angles
        tmp2 = np.dot(Ww, diff2)

        J_p = self.dyn.substitute(self.J_p, ang_s0=ang_s, q0=q)
        J_w = self.dyn.substitute(self.J_w, ang_s0=ang_s, q0=q)
        J_p, J_w = np.array(J_p).astype(np.float64), np.array(J_w).astype(np.float64),
        # Jv, Jw = self.generalized_jacobian(ang_s, q)

        nll = 0.5 * (X.T @ X + np.dot(diff1, tmp1) + np.dot(diff2, tmp2))
        ad = np.dot(J_p.T, tmp1) + np.dot(J_w.T, tmp2)
        grad_nll = X + ad[2:]
        return nll, grad_nll

    def inv_kin_ash(self, X0, r_eef_des, eul_ang_des):
        cost_grad = lambda X: self.laplace_cost_and_grad(X, r_eef_des, eul_ang_des)
        cost = lambda X: cost_grad(X)[0]
        grad = lambda X: cost_grad(X)[1]
        res = opt.minimize(cost, X0, method='BFGS', options={'maxiter': 150, 'disp': True})
        # res = opt.minimize(cost, X0, method='BFGS', )
        post_mean = res.x
        post_cov = res.hess_inv
        return post_mean, #post_cov

    def call_optimize2(self, target, q):
        points = self.path(target, q)  # q = initial joint values to compute the position of end_eff
        pr, pc = points.shape
        X0 = np.array([0.05, *q], dtype=float)
        # X0 = np.array([0.05, 0.07, 0.0, 0.0, 0.12, 0.2])
        X = np.zeros((X0.shape[0], pr))
        for i in range(pr):
            a = self.inv_kin_ash(X0, points[i, :], np.array([0, 0, 0]))
            X[:, i] = a[0]
            X0 = X[:, i]
        return X

    # Method 2: Fails
    def jac_pseudo_inv(self, X, path):
        lmda = 0.002
        ang_s, q = np.array([0., 0., X[0]]), X[1:]
        pv_eef = np.squeeze(self.manip_eef_pos_num(ang_s, q, ))
        dx = path - pv_eef
        J_star_num = self.generalized_jacobian(ang_s, q)
        J_star_num = np.array(J_star_num).astype(np.float64)
        a1 = J_star_num.transpose() @ J_star_num
        a2 = a1 + lmda ** 2 * np.eye(J_star_num.shape[0])
        a3 = np.linalg.solve(a2, J_star_num)
        dX = a3 @ dx
        return dX

    def analytic_inv_kin(self, target, q):
        points = self.path(target, q)  # q = initial joint values to compute the position of end_eff
        pr, pc = points.shape
        X0 = np.array([0.05, *q], dtype=float)
        # X0 = np.array([0.05, 0.07, 0.0, 0.0, 0.12, 0.2])
        X = np.zeros((X0.shape[0], pr))
        for i in range(pr):
            dX = self.jac_pseudo_inv(X0, points[i, :])
            X[:, i] = X0 + dX
            X0 = X[:, i]
        return X

    # Method 3:
    def jac(self, X, a):
        ang_s, q = np.array([0., 0., X[0]]), X[1:]
        qd = self.qd[2:]
        vel_eef_num = self.dyn.substitute(self.vel_eef, ang_s0=ang_s, q0=q)
        J = vel_eef_num.jacobian(qd),
        return np.array(J[0]).astype(np.float64)

    def inv_kin_optim_func(self, X, r_eef_des, ):
        ang_s, q = np.array([0., 0., X[0]]), X[1:]
        # ang_s, q = X[0:3], X[3:]
        pv_eef = self.manip_eef_pos_num(ang_s, q, )
        temp = r_eef_des - np.squeeze(pv_eef)
        return temp.dot(temp)

    def inv_kin(self, X0, eef_des_pos):
        results = opt.minimize(self.inv_kin_optim_func, X0, args=eef_des_pos, method='BFGS',
                               options={'maxiter': 150, 'disp': True})
        return results.x

    def call_optimize(self, target, q):
        points = self.path(target, q)  # q = initial joint values to compute the position of end_eff
        pr, pc = points.shape
        X0 = np.array([0.05, *q])
        # X0 = np.array([0.05, 0.07, 0.0, 0.0, 0.12, 0.2])
        X = np.zeros((X0.shape[0], pr))
        for i in range(pr):
            X[:, i] = self.inv_kin(X0, points[i, :])
            X0 = X[:, i]
        return X

    def animation(self, pos, size, color, rot_ang, q, path, pv_com=None, ax=None):
        # rot_ang is a 3 x t vector of the rotation angles of the spacecraft. q is manipulator angles
        if not ax:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
        for i in range(rot_ang.shape[1]):
            temp = [(pos[:, i][0], pos[:, i][1], pos[:, i][2])]
            qi = q[:, i]
            plt.cla()
            if isinstance(pv_com, (list, tuple, np.ndarray)):
                ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
            for p, s, c in zip(temp, size, color):
                self.sat_manip_sim.satellite_namipulator(rot_ang[:, i], qi, pos=p, size=s, ax=ax, color=c)
                ax.scatter(path[:, 0], path[:, 1], path[:, 2], 'r-', lw=4)
            plt.pause(0.3)
            # plt.savefig("/home/ar0058/Ash/repo/model_predictive_control/src/animation/%02d.png" % i)
            # print('hi')


if __name__ == '__main__':
    nDoF = 3
    IK = InverseKinematics(nDoF, robot='3DoF')
    target_loc = np.array([1.0, -2.0, 0.0])
    q0 = Array([[0.], [np.pi / 2], [0.]])
    ang_s0 = IK.kin.ang_s0
    r_s0 = IK.spacecraft_com_num(ang_s0, q0)
    # IK.plot(target_loc, Array([[0.], [0], [0.]]))
    # XX = IK.analytic_inv_kin(target_loc, q0)
    X = IK.call_optimize(target_loc, q0)
    print('###########################')
    # XX = IK.call_optimize2(target_loc, q0)


    ff = input('Enter')
    if ff == 1:
        P = X
    else:
        P = XX
    z = np.zeros(P.shape[1])
    ang_s, q = np.vstack((z, z, P[0, :])), P[1:, :]
    # ang_s, q = X[0:3, :], X[3:, :]
    # plt.figure()
    # plt.plot(X[0, :], label='satellite_z_rotation')
    # # plt.plot(X[1, :], label='satellite_y_rotation')
    # # plt.plot(X[2, :], label='satellite_z_rotation')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(X[1, :], label='q1')
    # plt.plot(X[2, :], label='q2')
    # plt.plot(X[3, :], label='q3')
    # plt.legend()

    r_s = np.zeros((3, ang_s.shape[1]))
    for i in range(ang_s.shape[1]):
        r_s[:, i] = np.squeeze(IK.spacecraft_com_num(ang_s[:, i], q[:, i]))

    q = np.c_[q0, q]
    r_s = np.c_[r_s0, r_s]
    ang_s = np.c_[ang_s0, ang_s]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(target_loc[0], target_loc[1], target_loc[2], lw=5)
    points = IK.path(target_loc, np.squeeze(q0))

    IK.animation(r_s, IK.kin.size, 'green', ang_s, q, points, ax=ax)
    print('hi')

    plt.show()