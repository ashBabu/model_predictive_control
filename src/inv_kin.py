import numpy as np
from eom_symbolic import dynamics, kinematics
from sympy import *

class InverseKinematics():

    def __init__(self, nDoF, robot='3DoF'):
        self.nDoF = nDoF
        self.kin = kinematics(nDoF, robot='3DoF')
        self.dyn = dynamics(nDoF, robot='3DoF')

    def generalized_jacobian(self):
        qd = self.kin.qd[3:]
        qd_s, qd_m = qd[0:3], qd[3:]
        m, I = self.dyn.mass, self.dyn.I_numeric
        l = self.kin.l_numeric[1:]  # cutting out satellite length l0
        r_s0, ang_s0, ang_b0, q0, b0 = self.kin.r_s0, self.kin.ang_s0, self.kin.ang_b, self.kin.q0, self.kin.b0

        j_omega, j_vel_com = self.dyn.velocities_frm_momentum_conservation()
        pv_com, pv_eef, pv_origin = self.dyn.com_pos_vect()
        vel_eef = diff(pv_eef, Symbol('t'))
        vel_eef_num = self.dyn.substitute(vel_eef, m, l, I, b0, ang_s0, ang_b0, r_s0, q0, )
        j_omega_num = self.dyn.substitute(j_omega[:, -1], m, l, I, b0, ang_s0, ang_b0, r_s0, q0, )
        # tt1, tt
        Jvw, Jvq = vel_eef_num.jacobian(qd_s), vel_eef_num.jacobian(qd_m)
        Jvw, Jvq = np.array(Jvw).astype(np.float64), np.array(Jvq).astype(np.float64)
        Jww, Jwq = j_omega_num.jacobian(qd_s), j_omega_num.jacobian(qd_m)
        Jww, Jwq = np.array(Jww).astype(np.float64), np.array(Jwq).astype(np.float64)

        L = self.dyn.ang_momentum_conservation()
        L_num = self.dyn.substitute(L, m, l, I, b0, ang_s0, ang_b0, r_s0, q0)
        Ls, Lm = L_num.jacobian(qd_s), L_num.jacobian(qd_m)
        Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)

        z1 = np.linalg.solve(Ls, Lm)  # omega_s = z1 x q_dot
        Jv, Jw = Jvq - Jvw @ z1, Jwq - Jww @ z1
        return Jv, Jw

    def inv_kin(self, v_ee, w_ee):
        Jv, Jw = self.generalized_jacobian()
        # q_dot =
        print('hi')


if __name__ == '__main__':
    nDoF = 3
    IK = InverseKinematics(nDoF, robot='3DoF')
    IK.generalized_jacobian()