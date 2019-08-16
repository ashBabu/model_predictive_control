import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from ff_eom_symbolic import dynamics, kinematics
from sympy import *
from sympy.physics.mechanics import *
import dill
dill.settings['recurse'] = True
import pickle


class SolveEom:
    def __init__(self, nDoF):
        self.nDoF = nDoF
        self.dyn = dynamics(nDoF)
        self.kin = kinematics(nDoF)
        self.M, self.C = self.dyn.get_dyn_para()

        # Constant parameters
        self.bn = [0.2, 0.3, 0.0]  # vector from spacecraft COM to robot base wrt spacecraft CS
        self.ang_b = [0.0, 0.0, np.pi/4]  # angular position of robot base wrt spacecraft COM CS
        self.Isn = [0.0, 0.0, 0.05]  # spacecraft Is_xx, Is_yy, Is_zz (principal MOM about its COM)
        self.mn = [10.0, 3.0, 2.0]  # mass of spacecraft, lin1, lik2
        self.rn = [0.3, 0.4]  # x component of COM vector for each of the links
        self.ln = [1.5, 1]
        self.Ixxn = [0.0, 0.0]
        self.Iyyn = [0.0, 0.0]
        self.Izzn = [0.04, 0.06]

        self.M_temp = msubs(self.M,
                   {self.kin.b0x: self.bn[0], self.kin.b0y: self.bn[1], self.kin.b0z: self.bn[2],
                    self.kin.ang_xb: self.ang_b[0], self.kin.ang_yb: self.ang_b[1], self.kin.ang_zb: self.ang_b[2],
                    self.dyn.Is_xx: self.Isn[0], self.dyn.Is_yy: self.Isn[1], self.dyn.Is_zz: self.Isn[2],
                    self.dyn.m[0]: self.mn[0], self.dyn.m[1]: self.mn[1], self.dyn.m[2]: self.mn[2],
                    self.kin.rx[0]: self.rn[0], self.kin.rx[1]: self.rn[1],
                    self.kin.l[0]: self.ln[0], self.kin.l[1]: self.ln[1],
                    self.dyn.Ixx[0]: self.Ixxn[0], self.dyn.Ixx[1]: self.Ixxn[1],
                    self.dyn.Iyy[0]: self.Iyyn[0], self.dyn.Iyy[1]: self.Iyyn[1],
                    self.dyn.Izz[0]: self.Izzn[0], self.dyn.Izz[1]: self.Izzn[1],})

        self.C_temp = msubs(self.C,
                   {self.kin.b0x: self.bn[0], self.kin.b0y: self.bn[1], self.kin.b0z: self.bn[2],
                    self.kin.ang_xb: self.ang_b[0], self.kin.ang_yb: self.ang_b[1], self.kin.ang_zb: self.ang_b[2],
                    self.dyn.Is_xx: self.Isn[0], self.dyn.Is_yy: self.Isn[1], self.dyn.Is_zz: self.Isn[2],
                    self.dyn.m[0]: self.mn[0], self.dyn.m[1]: self.mn[1], self.dyn.m[2]: self.mn[2],
                    self.kin.rx[0]: self.rn[0], self.kin.rx[1]: self.rn[1],
                    self.kin.l[0]: self.ln[0], self.kin.l[1]: self.ln[1],
                    self.dyn.Ixx[0]: self.Ixxn[0], self.dyn.Ixx[1]: self.Ixxn[1],
                    self.dyn.Iyy[0]: self.Iyyn[0], self.dyn.Iyy[1]: self.Iyyn[1],
                    self.dyn.Izz[0]: self.Izzn[0], self.dyn.Izz[1]: self.Izzn[1],})
        print('hi')
    # function that returns dy/dt
    def model(self, X, t):
        pos, vel = X[0:8], X[8:]
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

    def solve_ode(self):
        t = np.linspace(0, 3, 10)
        pos = np.array([0.1, 0.2, 0.0, 0.0, 0.0, np.pi/4, np.pi/8, np.pi/6])
        vel = np.array([0.01, 0.02, 0.0, 0.0, 0.0, 0.1*np.pi/4, 0.1*np.pi/8, 0.1*np.pi/6])
        y0 = np.array([0.1, 0.2, 0.0, 0.0, 0.0, np.pi/4, np.pi/8, np.pi/6,
                       0.01, 0.02, 0.0, 0.0, 0.0, 0.1*np.pi/4, 0.1*np.pi/8, 0.1*np.pi/6])
        X0 = [pos, vel]
        y1 = odeint(self.model, y0, t, )
        return y1


if __name__ == '__main__':
    nDoF = 4
    t = np.linspace(0, 3, 10)
    seom = SolveEom(nDoF)
    y1 = seom.solve_ode()
    r_sx = y1[:, 0]
    r_sy = y1[:, 1]
    ang_sz = y1[:, 7]
    v_sx = y1[:, 8]
    v_sy = y1[:, 9]
    w_sz = y1[:, -1]

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

    # dill.dump(M, open("MassMatrix", "wb"))
    # dill.dump(C, open("Coriolis&Centrifugal", "wb"))
    #
    # MassMatrix = dill.load(open("MassMatrix", "rb"))
    # Coriolis = dill.load(open("Coriolis", "rb"))



    # initial condition
    y0 = 5

    # time points
    t = np.linspace(0,20)

    # plot results
    # plt.plot(t,y1,'r-',linewidth=2,label='k=0.1')
    # plt.plot(t,y2,'b--',linewidth=2,label='k=0.2')
    # plt.plot(t,y3,'g:',linewidth=2,label='k=0.5')
    # plt.xlabel('time')
    # plt.ylabel('y(t)')
    # plt.legend()
    # plt.show()