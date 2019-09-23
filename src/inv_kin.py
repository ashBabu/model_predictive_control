import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from eom_symbolic import dynamics, kinematics
from sympy import *
from sat_manip_simulation import Simulation
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3)


class InverseKinematics():

    def __init__(self, nDoF, robot='3DoF'):
        self.t = Symbol('t')
        self.nDoF = nDoF
        self.kin = kinematics(nDoF, robot)
        self.dyn = dynamics(nDoF, robot)
        self.sat_manip_sim = Simulation()
        self.m, self.I, self.l = self.dyn.mass, self.dyn.I_numeric, self.kin.l_numeric[1:]  # cutting out satellite length l0
        self.ang_b0, self.b0 = self.kin.ang_b, self.kin.b0
        self.ang_s0 = self.kin.ang_s0
        pv_com, pv_eef, _ = self.dyn.com_pos_vect()
        L = self.dyn.ang_momentum_conservation()
        self.L_num = self.dyn.substitute(L, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.pv_com_num = self.dyn.substitute(pv_com, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.pv_eef_num = self.dyn.substitute(pv_eef, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.vel_eef = diff(self.pv_eef_num, self.t)
        self.qd = self.kin.qd[3:]
        self.qd_s, self.qd_m = self.qd[0:3], self.qd[3:]
        # self.J_star = self.generalized_jacobian_symb()

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def jacobians(self, ang_s, q):
        vel_eef_num = self.dyn.substitute(self.vel_eef, ang_s0=ang_s, q0=q)
        Js, Jm = vel_eef_num.jacobian(self.qd_s), vel_eef_num.jacobian(self.qd_m)  # Js and Jm are both functions of ang_s, q_i
        return Js, Jm

    def generalized_jacobian(self, ang_s, q):
        Js, Jm = self.jacobians(ang_s, q)
        Js, Jm = np.array(Js).astype(np.float64), np.array(Jm).astype(np.float64),
        L_num = self.dyn.substitute(self.L_num, ang_s0=ang_s, q0=q)
        Ls, Lm = L_num.jacobian(self.qd_s), L_num.jacobian(self.qd_m)
        Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)
        a1 = np.linalg.solve(Ls, Lm)
        return Jm - Js @ a1

    def jac_pseudo_inv(self, X, path):
        lmda = 0.002
        ang_s, q = np.array([0., 0., X[0]]), X[1:]
        pv_eef_num = np.squeeze(self.manip_eef_pos_num(ang_s, q, ))
        dx = path - pv_eef_num
        J_star_num = self.generalized_jacobian(ang_s, q)
        J_star_num = np.array(J_star_num).astype(np.float64)
        a1 = J_star_num.transpose() @ J_star_num
        a2 = a1 + lmda**2 * np.eye(J_star_num.shape[0])
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


    def generalized_jacobian2(self, ang_s, q):
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        m, I = self.dyn.mass, self.dyn.I_numeric
        l = self.kin.l_numeric[1:]  # cutting out satellite length l0
        ang_b0, b0 = self.kin.ang_b, self.kin.b0
        j_omega, j_vel_com = self.dyn.velocities_frm_momentum_conservation()
        pv_com, pv_eef, pv_origin = self.dyn.com_pos_vect()
        pv_com_num0 = self.dyn.substitute(pv_com, m=m, l=l, I=I, b=b0, ang_b0=ang_b0, ang_s0=ang_s, q0=q)
        pv_com_num0 = np.array(pv_com_num0).astype(np.float64)
        r_s0 = pv_com_num0[:, 0].reshape((3, 1))
        vel_eef = diff(pv_eef, Symbol('t'))

        vel_eef_num = self.dyn.substitute(vel_eef, m=m, l=l, I=I, b=b0, ang_b0=ang_b0, ang_s0=ang_s, q0=q, r_s0=r_s0)
        j_omega_num = self.dyn.substitute(j_omega[:, -1], m=m, l=l, I=I, b=b0, ang_b0=ang_b0, ang_s0=ang_s, q0=q, r_s0=r_s0)
        # tt1, tt
        Jvw, Jvq = vel_eef_num.jacobian(qd_s), vel_eef_num.jacobian(qd_m)
        Jvw, Jvq = np.array(Jvw).astype(np.float64), np.array(Jvq).astype(np.float64)
        Jww, Jwq = j_omega_num.jacobian(qd_s), j_omega_num.jacobian(qd_m)
        Jww, Jwq = np.array(Jww).astype(np.float64), np.array(Jwq).astype(np.float64)

        L = self.dyn.ang_momentum_conservation()
        L_num = self.dyn.substitute(L, m=m, l=l, I=I, b=b0, ang_b0=ang_b0, ang_s0=ang_s, q0=q, r_s0=r_s0)
        Ls, Lm = L_num.jacobian(qd_s), L_num.jacobian(qd_m)
        Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)

        z1 = np.linalg.solve(Ls, Lm)  # omega_s = z1 x q_dot
        Jv, Jw = Jvq - Jvw @ z1, Jwq - Jww @ z1
        return np.vstack((Jv, Jw))

    def spacecraft_com_num(self, ang_s, q):
        r_s0 = self.dyn.substitute(self.pv_com_num[:, 0], ang_s0=ang_s, q0=q)
        r_s0 = np.array(r_s0).astype(np.float64)
        r_s0 = r_s0.reshape((3, 1))
        return r_s0

    def manip_eef_pos_num(self, ang_s, q):
        eef_pos = self.dyn.substitute(self.pv_eef_num, ang_s0=ang_s, q0=q)
        eef_pos = np.array(eef_pos).astype(np.float64)
        return eef_pos

    def path(self, eef_des_pos, q0):  # q0 is current joint angles which is used to calculate current end_eff position
        pv_eef_num = self.manip_eef_pos_num(self.ang_s0, q0)
        init_pos = np.squeeze(pv_eef_num)
        points = self.discretize(init_pos, eef_des_pos, step_size=0.45)  # step_size is inversely proportional to vel
        points = np.insert(points, 0, init_pos, axis=0)
        points = np.insert(points, len(points), eef_des_pos, axis=0)
        return points

    def plot(self, target, q):
        size = self.kin.size
        pos = self.spacecraft_com_num(self.ang_s0, q)
        rot_ang = self.ang_s0.reshape(3, 1)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.sat_manip_sim.call_plot(pos, size, 'green', rot_ang, q, ax=ax)
        ax.scatter(target[0], target[1], target[2], lw=5)
        points = self.path(target, np.squeeze(q))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 'r-', lw=4)

    def inv_kin_optim_func(self, X, r_eef_des,):
        ang_s, q = np.array([0., 0., X[0]]), X[1:]
        # ang_s, q = X[0:3], X[3:]
        pv_eef_num = self.manip_eef_pos_num(ang_s, q,)
        temp = r_eef_des - np.squeeze(pv_eef_num)
        return temp.dot(temp)

    def inv_kin(self, X0, eef_des_pos):
        results = opt.minimize(self.inv_kin_optim_func, X0, args=eef_des_pos, method='BFGS',
                              options={'maxiter': 150, 'disp': True})
        return results.x

    def call_optimize(self, target, q):
        points = self.path(target, q)  # q = initial joint values to compute the position of end_eff
        pr, pc = points.shape
        X0 = np.array([0.05, 0.0, 0.12, 0.2])
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
                self.sat_manip_sim.satellite_namipulator(rot_ang[:, i], qi,  pos=p, size=s, ax=ax, color=c)
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
    XX = IK.analytic_inv_kin(target_loc, q0)
    X = IK.call_optimize(target_loc, q0)
    z = np.zeros(X.shape[1])
    ang_s, q = np.vstack((z, z, X[0, :])), X[1:, :]
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