import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from eom_symbolic import dynamics, kinematics
from sympy import *
from sympy.physics.mechanics import *


class SolveEom:
    def __init__(self, nDoF):
        self.nDoF = nDoF
        self.dyn = dynamics(nDoF, robot='3DoF')
        self.kin = kinematics(nDoF, robot='3DoF')
        self.M, self.C = self.dyn.get_dyn_para()

        self.M_temp = self.M
        for i in range(len(self.dyn.m)):
            self.M_temp = msubs(self.M_temp, {self.dyn.m[i]: self.dyn.mass[i]})
        for i in range(len(self.kin.l)):
            self.M_temp = msubs(self.M_temp, {self.kin.l[i]: self.kin.l_numeric[i+1]})
        for i in range(len(self.dyn.I_full)):
            self.M_temp = msubs(self.M_temp, {self.dyn.I_full[i]: self.dyn.I_numeric[i]})
        for i in range(len(self.kin.q0)):
            self.M_temp = msubs(self.M_temp, {self.kin.qm[i]: self.kin.q0[i]})
        self.M_temp = msubs(self.M_temp, {self.kin.b0x: self.kin.b0[0], self.kin.b0y: self.kin.b0[1],
                                          self.kin.b0z: self.kin.b0[2], self.kin.ang_xs: self.kin.ang_s0[0],
                                          self.kin.ang_ys: self.kin.ang_s0[1], self.kin.ang_zs: self.kin.ang_s0[2],
                                          self.kin.ang_xb: self.kin.ang_b[0], self.kin.ang_yb: self.kin.ang_b[1],
                                          self.kin.ang_zb: self.kin.ang_b[2], self.kin.r_sx: self.kin.r_s0[0],
                                          self.kin.r_sy: self.kin.r_s0[1], self.kin.r_sz: self.kin.r_s0[2]})

        self.C_temp = self.C
        for i in range(len(self.dyn.m)):
            self.C_temp = msubs(self.C_temp, {self.dyn.m[i]: self.dyn.mass[i]})
        for i in range(len(self.kin.l)):
            self.C_temp = msubs(self.C_temp, {self.kin.l[i]: self.kin.l_numeric[i+1]})
        for i in range(len(self.dyn.I_full)):
            self.C_temp = msubs(self.C_temp, {self.dyn.I_full[i]: self.dyn.I_numeric[i]})
        for i in range(len(self.kin.q0)):
            self.C_temp = msubs(self.C_temp, {self.kin.qm[i]: self.kin.q0[i]})
        self.C_temp = msubs(self.C_temp, {self.kin.b0x: self.kin.b0[0], self.kin.b0y: self.kin.b0[1],
                                          self.kin.b0z: self.kin.b0[2], self.kin.ang_xs: self.kin.ang_s0[0],
                                          self.kin.ang_ys: self.kin.ang_s0[1], self.kin.ang_zs: self.kin.ang_s0[2],
                                          self.kin.ang_xb: self.kin.ang_b[0], self.kin.ang_yb: self.kin.ang_b[1],
                                          self.kin.ang_zb: self.kin.ang_b[2], self.kin.r_sx: self.kin.r_s0[0],
                                          self.kin.r_sy: self.kin.r_s0[1], self.kin.r_sz: self.kin.r_s0[2]})

    # function that returns dy/dt
    def model(self, X, t):
        pos, vel = X[0:self.nDoF+6], X[self.nDoF+6:]
        Mt = self.M_temp
        Ct = self.C_temp
        for i in range(len(pos)):
            Mt = msubs(Mt, {self.kin.q[i]: pos[i]})
            Ct = msubs(Ct, {self.kin.q[i]: pos[i], self.kin.q[i].diff(): vel[i], })
        M, C = np.array(Mt).astype(np.float64), np.array(Ct).astype(np.float64)  # convert from sympy to numpy array

        x1dot = vel
        x2dot = np.linalg.solve(M, C)
        # dydt = np.array([x1dot, x2dot])
        dydt = np.hstack((x1dot, np.squeeze(x2dot)))
        return dydt

    def solve_ode(self, t):
        pos = np.hstack((self.kin.r_s0, self.kin.ang_s0, self.kin.q0))  # [ r_s, ang_s, q1, q2, ...]
        vel = 0.01 * np.ones(len(pos))
        y0 = np.hstack((pos, vel))
        y1 = odeint(self.model, y0, t, )
        return y1


if __name__ == '__main__':
    nDoF = 3
    t = np.linspace(0, 60, 90)
    seom = SolveEom(nDoF)
    y1 = seom.solve_ode(t)
    r_sx, r_sy, r_sz = y1[:, 0], y1[:, 1], y1[:, 2]
    ang_sx, ang_sy, ang_sz = y1[:, 3], y1[:, 4], y1[:, 5]

    v_sx, v_sy, v_sz = y1[:, 6+nDoF], y1[:, 6+nDoF+1], y1[:, 6+nDoF+2]

    w_sx, w_sy, w_sz  = y1[:, 6+nDoF+3], y1[:, 6+nDoF+4], y1[:, 6+nDoF+5]

    # f = open("data", "w")
    # f.write("# t y\n")  # column names
    np.savetxt('solution.txt', y1, fmt="%.3f", )

    # loading:
    # x, y = np.loadtxt('solution.txt', unpack=True)

    plt.plot(t, r_sx, 'r-', linewidth=2, label='x_position')
    plt.plot(t, r_sy, 'b--', linewidth=2, label='y_position')
    plt.plot(t, ang_sz, 'g:', linewidth=2,label='z_angle')

    plt.plot(t, v_sx, 'r^', linewidth=2, label='x_vel')
    plt.plot(t, v_sy, 'bo', linewidth=2, label='y_vel')
    plt.plot(t, w_sz, 'g*', linewidth=2,label='z_ang_vel')
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()
    # Ms = msubs(seom.M, {seom.kin.ang_xs:0, seom.kin.ang_ys:0, seom.kin.ang_zs:0, seom.kin.ang_xb:0, seom.kin.ang_yb:0, seom.kin.ang_zb:0, seom.kin.r_sx:0, seom.kin.r_sy:0, seom.kin.r_sz:0, seom.kin.b0x:0, seom.kin.b0y:0, seom.kin.b0z:0, seom.dyn.Is_xx:0, seom.dyn.Is_yy:0, seom.dyn.Is_zz:0, seom.dyn.m[0]:0})

    print('hi')
