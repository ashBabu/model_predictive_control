import numpy as np
from eom_symbolic import dynamics, kinematics


class InverseKinematics():

    def __init__(self, nDoF, robot='3DoF'):
        self.nDoF = nDoF
        self.kin = kinematics(nDoF, robot='3DoF')
        self.dyn = dynamics(nDoF, robot='3DoF')

    def inv_kin(self):
        j_omega, j_vel_com = self.dyn.velocities_frm_momentum_conservation()
        print('hi')


if __name__ == '__main__':
    nDoF = 3
    IK = InverseKinematics(nDoF, robot='3DoF')
    IK.inv_kin()