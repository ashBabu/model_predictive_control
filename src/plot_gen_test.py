import numpy as np
import matplotlib.pyplot as plt
from draw_satellite import SatellitePlotter
from fwd_kin import ForwardKinematics
from fwd_kin import MayaviRendering
from dh_plotter import DH_plotter
np.set_printoptions(precision=3)
from mayavi import mlab
from tvtk.api import tvtk

save_dir = '/home/ashith/Ash/repo/model_predictive_control/src/save_dir_fwd_kin/'

# q1 = np.zeros(7)
# DH = DH_plotter(nDoF=7, robot='7DoF')
# T_joint, Ti = DH.robot_DH_matrix(q1)

pi = np.pi
q1 = np.array([(0., 5*pi/4, 0., 0., 0., pi/2, 0.)])
b0 = np.array([-1.05, 1.05, 0])
# r_s, ang_s, q, q_dot, t, pv_com = sim.dyn.get_positions(b0=b0)
# np.save('rs.npy', r_s, allow_pickle=True), np.save('ang_s.npy', ang_s, allow_pickle=True), np.save('q.npy', q, allow_pickle=True), \
# np.save('q_dot.npy', q_dot, allow_pickle=True), np.save('pv_com.npy', pv_com, allow_pickle=True)
t = np.load(save_dir + 'data/t.npy', allow_pickle=True)
r_s, ang_s, q, q_dot, pv_com = np.load(save_dir + 'data/rs.npy', allow_pickle=True), \
                               np.load(save_dir + 'data/ang_s.npy', allow_pickle=True), \
                               np.load(save_dir + 'data/q.npy', allow_pickle=True), \
                               np.load(save_dir + 'data/q_dot.npy', allow_pickle=True), \
                               np.load(save_dir + 'data/pv_com.npy', allow_pickle=True),

fwd_kin =ForwardKinematics(nDoF=7, robot='7DoF')
i = 1  # int(input('1:Matplotlib, 2: Mayavi'))
rs1, ang_s1 = r_s[:, 0].reshape(-1, 1), ang_s[:, 0].reshape(-1, 1)
if i == 1:
    # For Matplotlib simulation:
    fwd_kin.simulation(r_s=rs1, ang_s=ang_s1, q=q1, b0=b0)
    plt.show()
# For Mayavi Rendering:
elif i == 2:
    MR = MayaviRendering(nDoF=7, robot='7DoF')
    MR.anim(rs=rs1, angs=ang_s1, q=q1, b0=b0, fig_save=False, reverse=True)
    mlab.show()

print('hi')
