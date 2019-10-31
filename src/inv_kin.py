import numpy as np
import scipy.optimize as opt
import time
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
        self.m, self.I, self.l = self.dyn.mass, self.dyn.I_num, self.kin.l_num[1:]  # cutting out satellite length l0
        self.ang_b0, self.b0 = self.kin.ang_b, self.kin.b0
        self.ang_s0 = self.kin.ang_s0
        pv_com, pv_eef, _ = self.dyn.com_pos_vect()
        self.pv_com_num = self.dyn.substitute(pv_com, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.pv_eef_num = self.dyn.substitute(pv_eef, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        L = self.dyn.ang_momentum_conservation()
        self.L_num = self.dyn.substitute(L, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        j_omega, _, j_vel_eef = self.dyn.velocities_frm_momentum_conservation()
        self.omega_eef = self.dyn.substitute(j_omega[:, -1], m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.vel_eef = self.dyn.substitute(j_vel_eef, m=self.m, l=self.l, I=self.I, b=self.b0, ang_b0=self.ang_b0)
        self.qd = self.kin.qd[3:]
        self.qd_s, self.qd_m = self.qd[0:3], self.qd[3:]
        self.lmda1, self.lmda2 = 2, 0.5  # optimization weights

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
        eef_pos = self.dyn.substitute(self.pv_eef_num, ang_s0=ang_s, q0=q)
        eef_pos = np.array(eef_pos).astype(np.float64)
        return np.squeeze(eef_pos)

    def path(self, eef_des_pos, q0):  # q0 is current joint angles which is used to calculate current end_eff position
        pv_eef_num = self.manip_eef_pos_num(self.ang_s0, q0)
        init_pos = np.squeeze(pv_eef_num)
        points = self.discretize(init_pos, eef_des_pos, step_size=0.15)  # step_size is inversely proportional to vel
        points = np.insert(points, 0, init_pos, axis=0)
        points = np.insert(points, len(points), eef_des_pos, axis=0)
        return points

    # Method 1: works but needs refinement
    def jacobians(self, ang_s, q):
        omega_eef_num = self.dyn.substitute(self.omega_eef, ang_s0=ang_s, q0=q)
        Jw_s, Jw_m = omega_eef_num.jacobian(self.qd_s), omega_eef_num.jacobian(self.qd_m)  # Jv_s and Jv_m are both functions of ang_s, q_i
        Jw_s, Jw_m = np.array(Jw_s).astype(np.float64), np.array(Jw_m).astype(np.float64),

        vel_eef_num = self.dyn.substitute(self.vel_eef, ang_s0=ang_s, q0=q)
        Jv_s, Jv_m = vel_eef_num.jacobian(self.qd_s), vel_eef_num.jacobian(self.qd_m)  # Jv_s and Jv_m are both functions of ang_s, q_i
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
        return np.vstack((Jv, Jw)), Ls, Lm

    # Method 1: Directly calculating as given in Umetani and Yoshida equation 22
    # Cant handle joint limits

    def dir(self, X, point):
        ang_s, q = X[0:3], X[3:]
        r_eef_current = self.manip_eef_pos_num(ang_s, q)
        dx = point - r_eef_current
        dx = np.hstack((dx, 0., 0., 0.))
        J, Ls, Lm = self.generalized_jacobian(ang_s, q)
        dq = np.linalg.pinv(J) @ dx
        dphi = -np.linalg.solve(Ls, Lm) @ dq
        return dq, dphi

    def call_dir(self, target, q0, ang_s0):
        q0 = np.squeeze(np.array(q0).astype(np.float64))
        ang_s0 = np.squeeze(np.array(ang_s0).astype(np.float64))
        points = self.path(target, q0)
        sh = points.shape[0]
        X0 = np.hstack((ang_s0, q0))
        q, ang_s = np.zeros((3, sh+1)), np.zeros((3, sh+1))
        q[:, 0], ang_s[:, 0] = q0, ang_s0
        for i in range(1, sh+1):
            dq, dphi = self.dir(X0, points[i-1, :])
            q[:, i] = q[:, i-1] + dq
            ang_s[:, i] = ang_s[:, i-1] + dphi
            X0 = np.hstack((ang_s[:, i], q[:, i]))
        return ang_s, q

    # Method 2: Using optimization and hence can handle bounds

    def cost(self, dq, r_eef_current, eef_des_pos, J):
        dx = eef_des_pos - r_eef_current
        dx = np.hstack((dx, 0., 0., 0.))
        t1 = dx - J @ dq
        g = np.array([0.5, 0.1, 0.5])
        W = np.eye(dq.shape[0]) #@ np.diag(g)
        # W = self.lmda1 * np.eye(dq.shape[0])
        cost = 0.5 * (dq.T @ W @ dq + self.lmda2 * t1.T @ t1)
        return cost

    def jac_cost(self, dq, r_eef_current, eef_des_pos, J):
        # r_eef_current = self.manip_eef_pos_num(ang_s, q)
        dx = eef_des_pos - r_eef_current
        dx = np.hstack((dx, 0., 0., 0.))
        t1 = dx - J @ dq
        g = np.array([0.5, 2, 0.8])
        W = np.eye(dq.shape[0]) #@ np.diag(g)
        # W = self.lmda1 * np.eye(dq.shape[0])
        jac = W @ dq - self.lmda2 * J.T @ t1
        return jac

    def inv_kin(self, dq0, r_eef_current, eef_des_pos, J):
        # results = opt.minimize(self.inv_kin_optim_func, X0, args=eef_des_pos, method='BFGS',
        #                        options={'maxiter': 150, 'disp': True})
        bnds = ((-np.pi, np.pi), (-np.pi/6, np.pi), (-np.pi/4, np.pi/2))
        bnds1 = ((None, None), (None, None), (None, None), (None, None), (None, None), (None, None))
        results = opt.minimize(self.cost, dq0, args=(r_eef_current, eef_des_pos, J), method='SLSQP', jac=self.jac_cost,
                               options={'maxiter': 150, 'disp': True}, bounds=bnds)
        # results = opt.fmin_slsqp(func=self.inv_kin_optim_func,
        #                           x0=X0, eqcons=[self.constraints[0],self.constraints[1], self.constraints[2]],
        #                           args=eef_des_pos, iprint=0)
        return results.x

    def call_optimize(self, target, ang_s0, q0):
        q0 = np.squeeze(np.array(q0).astype(np.float64))
        ang_s0 = np.squeeze(np.array(ang_s0).astype(np.float64))
        dq0 = np.array([0.00, 0.03, 0.1])
        points = self.path(target, q0)  # q = initial joint values to compute the position of end_eff
        pr, pc = points.shape
        q, ang_s = np.zeros((3, pr + 1)), np.zeros((3, pr + 1))
        q[:, 0], ang_s[:, 0] = q0, ang_s0
        J, Ls, Lm = self.generalized_jacobian(ang_s0, q0)
        r_eef_current = self.manip_eef_pos_num(ang_s0, q0)
        for i in range(1, pr+1):
            dq = self.inv_kin(dq0, r_eef_current, points[i-1, :], J)
            q[:, i] = q[:, i - 1] + dq
            ang_s[:, i] = ang_s[:, i - 1] - np.linalg.solve(Ls, Lm) @ dq
            ang_s0, q0, dq0 = ang_s[:, i], q[:, i], dq
            J, Ls, Lm = self.generalized_jacobian(ang_s0, q0)
            r_eef_current = self.manip_eef_pos_num(ang_s0, q0)
        return ang_s, q

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
                ax.view_init(elev=85., azim=-58)
            plt.pause(0.05)
            # plt.savefig("/home/ar0058/Ash/repo/model_predictive_control/src/animation/inv_kinematics_direct/%02d.png" % i)
            # print('hi')


if __name__ == '__main__':
    nDoF = 3
    IK = InverseKinematics(nDoF, robot='3DoF')
    target_loc = np.array([1.0, -2.0, 0.0])
    q0 = Array([[0.], [np.pi / 2], [-0.0]])
    # q0 = Array([[0.], [np.pi / 2.5], [-0.03]])
    ang_s0 = IK.kin.ang_s0
    r_s0 = IK.spacecraft_com_num(ang_s0, q0)

    f1 = int(input('Enter 1: for optimized result 2: analytical'))
    if f1 == 1:
        A, Q = IK.call_optimize(target_loc, ang_s0, q0)  # optimization method A = spacecraft angles and Q = joint angles
        # z = np.zeros(X.shape[1])
        # ang_s, q = np.vstack((z, z, X[0, :])), X[1:, :]
        # A, Q = ang_s, q
    else:
        A, Q = IK.call_dir(target_loc, q0, ang_s0)  # direct method

    r_s = np.zeros((3, A.shape[1]))
    for i in range(A.shape[1]):
        r_s[:, i] = np.squeeze(IK.spacecraft_com_num(A[:, i], Q[:, i]))

    q = np.c_[q0, Q]
    r_s = np.c_[r_s0, r_s]
    ang_s = np.c_[ang_s0, A]

    if f1 == 1:
        # ang_s, q = X[0:3, :], X[3:, :]

        plt.figure()
        plt.plot(A[0, :], label='satellite_x_rotation')
        plt.plot(A[1, :], label='satellite_y_rotation')
        plt.plot(A[2, :], label='satellite_z_rotation')
        plt.legend()

        plt.figure()
        plt.plot(Q[0, :], label='q1')
        plt.plot(Q[1, :], label='q2')
        plt.plot(Q[2, :], label='q3')
        plt.legend()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(target_loc[0], target_loc[1], target_loc[2], lw=5)
        points = IK.path(target_loc, np.squeeze(q0))

        IK.animation(r_s, IK.kin.size, 'green', A, Q, points, ax=ax)
    else:
        plt.figure()
        plt.plot(A[0, :], label='satellite_x_rotation')
        plt.plot(A[1, :], label='satellite_y_rotation')
        plt.plot(A[2, :], label='satellite_z_rotation')
        plt.legend()

        plt.figure()
        plt.plot(Q[0, :], label='q1')
        plt.plot(Q[1, :], label='q2')
        plt.plot(Q[2, :], label='q3')
        plt.legend()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(target_loc[0], target_loc[1], target_loc[2], lw=5)
        points = IK.path(target_loc, np.squeeze(q0))

        IK.animation(r_s, IK.kin.size, 'green', A, Q, points, ax=ax)

    end_eff_pos = np.zeros((3, q.shape[1]))
    for i in range(q.shape[1]):
        end_eff_pos[:, i] = IK.manip_eef_pos_num(ang_s[:, i], q[:, i])

    import pickle
    # writing to file
    with open('joint_angles.pickle', 'wb') as jq:
        jq.write(pickle.dumps(q))
    with open('spacecraft_angles.pickle', 'wb') as sq:
        sq.write(pickle.dumps(ang_s))

    with open('end_eff_cart_coord.pickle', 'wb') as eef:
        eef.write(pickle.dumps(end_eff_pos))

    # reading file
    # with open('joint_angles.pickle', 'rb') as LmdR:
    #     jjq = pickle.loads(LmdR.read())
    print('hi')
    plt.show()