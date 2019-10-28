import numpy as np
from sympy import *
import pickle
from eom_symbolic import kinematics, dynamics
from sympy.physics.mechanics import *
init_printing()

class FreeFloatingMPC(object):
    def __init__(self):
        self.kin, self.dyn = kinematics(), dynamics()
        self.alpha, self.beta, self.gamma = symbols('alpha beta gamma')
        self.alpha_d, self.beta_d, self.gamma_d = symbols('alpha_d beta_d gamma_d')
        self.theta_1, self.theta_2, self.theta_3 = symbols('theta_1 theta_2 theta_3')
        self.theta_1d, self.theta_2d, self.theta_3d = symbols('theta_1d theta_2d theta_3d')
        self.ang_s = [self.alpha, self.beta, self.gamma]
        self.omega_s = [self.alpha_d, self.beta_d, self.gamma_d]
        self.theta = [self.theta_1, self.theta_2, self.theta_3]
        self.theta_d = [self.theta_1d, self.theta_2d, self.theta_3d]

        with open('MassMat.pickle', 'rb') as inM:
            self.MassMat = pickle.loads(inM.read())
        with open('Corioli.pickle', 'rb') as inC:
            self.CoriolisVector = pickle.loads(inC.read())

    def substitute(self, parm,  m=None, l=None, I=None, ang_s=None, omega_s=None,
                   q=None, dq=None):
        if isinstance(m, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(m)):
                parm = msubs(parm, {self.dyn.m[i]: m[i]})
        if isinstance(I, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(I)):
                parm = msubs(parm, {self.dyn.I_full[i]: I[i]})
        if isinstance(l, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(l)):
                parm = msubs(parm, {self.kin.l[i]: l[i]})
        if isinstance(q, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(q)):
                parm = msubs(parm, {self.theta[i]: q[i]})
        if isinstance(dq, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            for i in range(len(dq)):
                parm = msubs(parm, {self.theta_d[i]: dq[i]})
        if isinstance(ang_s, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            parm = msubs(parm, {self.alpha: ang_s[0], self.beta: ang_s[1], self.gamma: ang_s[2]})
        if isinstance(omega_s, (list, tuple, np.ndarray, ImmutableDenseNDimArray)):
            parm = msubs(parm, {self.alpha_d: omega_s[0], self.beta_d: omega_s[1], self.gamma_d: omega_s[2]})
        return parm.evalf()

    def massMatrix(self, M=None, L=None, I=None, ang_s=None, q=None, omega_s=None, dq=None):
        Mass = self.substitute(self.MassMat, m=M, l=L, I=I, ang_s=ang_s, q=q, omega_s=omega_s, dq=dq)
        return Mass

    def coriolis(self, M=None, L=None, I=None, ang_s=None, q=None, omega_s=None, dq=None):
        Coriolis = self.substitute(self.CoriolisVector, m=M, l=L, I=I, ang_s=ang_s, q=q, omega_s=omega_s, dq=dq)
        return Coriolis


if __name__ == '__main__':
    ffmpc = FreeFloatingMPC()

    m, I = ffmpc.dyn.mass, ffmpc.dyn.I_num
    l = ffmpc.kin.l_num[1:]  # cutting out satellite length l0
    # ang_b, b0 = ffmpc.kin.ang_b, ffmpc.kin.b0

    ang_s = Array([0., 0., 0.])
    omega_s = Array([0.1, 0.2, 0.4])
    q = Array([pi / 3 * 0, 5 * pi / 4, pi / 2])
    dq = Array([0.05*pi / 3, 0.04 * 5 * pi / 4, 0.06 * pi / 2])

    MassMatrix = ffmpc.massMatrix(M=m, L=l, I=I, ang_s=ang_s, q=q)
    CoriolisVec = ffmpc.coriolis(M=m, L=l, I=I, ang_s=ang_s, q=q, omega_s=omega_s, dq=dq)
    print(CoriolisVec)
    print('############')
    print(MassMatrix)

    print('hi')

