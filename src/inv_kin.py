import numpy as np
from eom_symbolic import dynamics, kinematics
from sympy import *
from sat_manip_simulation import Simulation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class InverseKinematics():

    def __init__(self, nDoF, robot='3DoF'):
        self.nDoF = nDoF
        self.kin = kinematics(nDoF, robot)
        self.dyn = dynamics(nDoF, robot)
        self.sat_manip_sim = Simulation()

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def generalized_jacobian(self, r_s0, ang_s0, q0):
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        m, I = self.dyn.mass, self.dyn.I_numeric
        l = self.kin.l_numeric[1:]  # cutting out satellite length l0
        ang_b0, b = self.kin.ang_b, self.kin.b0
        j_omega, j_vel_com = self.dyn.velocities_frm_momentum_conservation()
        pv_com, pv_eef, pv_origin = self.dyn.com_pos_vect()
        vel_eef = diff(pv_eef, Symbol('t'))
        vel_eef_num = self.dyn.substitute(vel_eef, m, l, I, b, ang_b0, r_s0, ang_s0, q0)
        j_omega_num = self.dyn.substitute(j_omega[:, -1], m, l, I, b, ang_b0, r_s0, ang_s0, q0)
        # tt1, tt
        Jvw, Jvq = vel_eef_num.jacobian(qd_s), vel_eef_num.jacobian(qd_m)
        Jvw, Jvq = np.array(Jvw).astype(np.float64), np.array(Jvq).astype(np.float64)
        Jww, Jwq = j_omega_num.jacobian(qd_s), j_omega_num.jacobian(qd_m)
        Jww, Jwq = np.array(Jww).astype(np.float64), np.array(Jwq).astype(np.float64)

        L = self.dyn.ang_momentum_conservation()
        L_num = self.dyn.substitute(L, m, l, I, b, ang_b0, r_s0, ang_s0, q0)
        Ls, Lm = L_num.jacobian(qd_s), L_num.jacobian(qd_m)
        Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)

        z1 = np.linalg.solve(Ls, Lm)  # omega_s = z1 x q_dot
        Jv, Jw = Jvq - Jvw @ z1, Jwq - Jww @ z1
        return np.vstack((Jv, Jw))

    def path(self, eef_final_pos, q0):
        _, pv_eef, _ = self.dyn.com_pos_vect()
        m, I = self.dyn.mass, self.dyn.I_numeric
        l = self.kin.l_numeric[1:]  # cutting out satellite length l0
        ang_b0, b0 = self.kin.ang_b, self.kin.b0
        r_s0, ang_s0 = self.kin.r_s0, self.kin.ang_s0
        pv_eef_num = self.dyn.substitute(pv_eef, m, l, I, b0, ang_b0, r_s0, ang_s0, q0)
        pv_eef_num = np.array(pv_eef_num).astype(np.float64)
        init_pos = np.squeeze(pv_eef_num)
        points = self.discretize(init_pos, eef_final_pos, step_size=0.1)
        points = np.insert(points, 0, init_pos, axis=0)
        points = np.insert(points, len(points), eef_final_pos, axis=0)
        return points

    def plot(self, target):
        size = self.kin.size
        pos, rot_ang = self.kin.r_s0, self.kin.ang_s0
        pos, rot_ang = rot_ang.reshape((3, 1)), pos.reshape((3, 1))
        q = np.array([[0.], [np.pi/2], [0.]])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.sat_manip_sim.call_plot(pos, size, 'green', rot_ang, q, ax=ax)
        ax.scatter(target[0], target[1], target[2], lw=5)
        points = self.path(target, np.squeeze(q))
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 'r-', lw=4)

    def inv_kin(self, v_ee, w_ee):
        dt = 0.1

        J_genralised = self.generalized_jacobian(r_s0, ang_s0, q0)
        q_dot = np.linalg.solve(J_genralised, np.vstack((v_ee, w_ee)))
        print('hi')


if __name__ == '__main__':
    nDoF = 3
    IK = InverseKinematics(nDoF, robot='3DoF')
    target_location = np.array([0.5, 3, 0])
    # IK.generalized_jacobian
    # points = IK.path(target_location)
    IK.plot(target_location)
    plt.show()