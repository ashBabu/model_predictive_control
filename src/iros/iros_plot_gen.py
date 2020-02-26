import numpy as np
import matplotlib.pyplot as plt
from iros_forward_kin import ForwardKin
from mayavi import mlab
from iros_7DoF_EOM import Dynamics, Kinematics
# from fwd_kin import MayaviRendering
np.set_printoptions(precision=3)


if __name__ == '__main__':
    pi = np.pi
    q1 = np.array([0., 5 * pi / 4, 0., 0., 0., 0., 0.])
    b0 = np.array([-1.05, 1.05, 0], dtype=float)
    nDoF, robot = 7, '7DoF'
    dyn = Dynamics(nDoF=nDoF, robot=robot)
    fwd_kin = ForwardKin(nDoF=nDoF, robot=robot)
    i = 1  # int(input('1:Matplotlib, 2: Mayavi'))
    ang_s1 = np.zeros(3).reshape(-1, 1)
    rs1 = dyn.spacecraft_com_pos(q=q1, ang_s=ang_s1, b0=b0)
    if i == 1:
        # For Matplotlib simulation:
        fwd_kin.simulation(r_s=rs1, ang_s=ang_s1, q=q1, b0=b0)
        plt.show()
    # For Mayavi Rendering:
    elif i == 2:
        MR = MayaviRendering(nDoF=7, robot='7DoF')
        MR.anim(rs=rs1, angs=ang_s1, q=q1, b0=b0, fig_save=False, reverse=False)
        mlab.show()

    print('hi')
