import numpy as np
from fwd_kin import MayaviRendering
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

save_dir = '/home/ash/Ash/repo/model_predictive_control/src/save_data_inv_kin/animation/'
load_dir = '/home/ash/Ash/repo/model_predictive_control/src/save_data_inv_kin/data/'
save_dir_mpc = '/home/ash/Ash/repo/model_predictive_control/src/save_ffmpc/data/'


class Rendering:
    def __init__(self, nDoF=3, robot='3DoF', image_file='Nasa_blue_marble.jpg'):
        self.MR = MayaviRendering(nDoF=nDoF, robot=robot, image_file=image_file)

    @mlab.animate(delay=200)
    def anim(self, rs=None, angs=None, q=None, ref_path=None, target=None, end_eff_pos=None,
             fig_save=False, reverse=False):
        size = self.MR.kin.size[0]
        if reverse:
            rs, angs, q = np.hstack((rs, np.fliplr(rs))), np.hstack((angs, np.fliplr(angs))), np.hstack((q, np.fliplr(q)))
        mlab.figure(size=(1000, 800), bgcolor=(0., 0., 0.), )
        self.MR.manual_sphere(self.MR.image_file)
        p = [(rs[:, 0][0], rs[:, 0][1], rs[:, 0][2])][0]
        qi = q[:, 0]
        T_combined, x, y, z = self.MR.plot_satellite_namipulator(angs[:, 0], qi, pos=p, size=size)
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        robot_base = mlab.points3d(xx, yy, zz, mode='sphere', scale_factor=0.25, color=(0.5, 0.5, 0.5))
        satellite = mlab.plot3d(x, y, z, tube_radius=.1, color=(0.8, 0.9, 0.5))
        mlab.points3d(0, 0, 0, mode='sphere', scale_factor=0.4)
        xa, ya, za = T_combined[1:, 0, 3], T_combined[1:, 1, 3], T_combined[1:, 2, 3]
        manipulator = mlab.plot3d(xa, ya, za, color=(0.3, 0.4, 0.5), tube_radius=.1)
        mlab.plot3d(ref_path[:, 0], ref_path[:, 1], ref_path[:, 2], tube_radius=0.05, color=(0.4, 0.7, 0.25))
        if fig_save:
            mlab.savefig(save_dir+"animation/%02d.png" % 0)
        if angs.shape[1] > 3:
            n = angs.shape[1]
            for i in range(1, n):
                p = [(rs[:, i][0], rs[:, i][1], rs[:, i][2])][0]
                qi = q[:, i]
                T_combined, x, y, z = self.MR.plot_satellite_namipulator(angs[:, i], qi, pos=p, size=size)
                xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
                xa, ya, za = T_combined[1:, 0, 3], T_combined[1:, 1, 3], T_combined[1:, 2, 3]
                robot_base.mlab_source.set(x=xx, y=yy, z=zz)
                satellite.mlab_source.set(x=x, y=y, z=z)
                manipulator.mlab_source.set(x=xa, y=ya, z=za)
                if fig_save:
                    mlab.savefig(save_dir+"animation/%02d.png" % i)
                # cam, foc = mlab.move()
                # mlab.move(-3, i/45, 1.2)
                # mlab.view(azimuth=-i/45, elevation=-i/85, distance=None, focalpoint=None, roll=None,
                # reset_roll=True, figure=None)
                yield

    def matplotlib_anim(self, opt_ang, ref_ang):
        fig = plt.figure()

        # ax = plt.axes()
        ax = plt.axes(xlim=(-5, 5), ylim=(-4, 14))
        line1, = ax.plot([], [], 'r--', lw=3)
        line2, = ax.plot([], [], 'b--', lw=6)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2

        def animate(i):
            q1 = opt_ang[0][0:i]
            q2 = opt_ang[1][0:i]
            t = np.arange(len(q1))
            line1.set_data(t, q1)
            line2.set_data(t, q2)
            return line1, line2

        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=40, interval=20, blit=True)
        anim.save('sine_wave.gif', writer='imagemagick')


if __name__ == '__main__':
    end_eff_xyz, joint_angles, ref_path, spacecraft_angles, spacecraft_coms, target_loc = \
        np.load(load_dir+'end_eff_xyz.npy', allow_pickle=True), \
        np.load(load_dir+'joint_angs_inv_kin.npy', allow_pickle=True), \
        np.load(load_dir+'ref_path_xyz.npy', allow_pickle=True), \
        np.load(load_dir+'spacecraft_angs_inv_kin.npy', allow_pickle=True), \
        np.load(load_dir+'spacecraft_com_inv_kin.npy', allow_pickle=True), \
        np.load(load_dir+'target_loc_inv_kin.npy', allow_pickle=True)
    # q = joint_angles, ang_s = spacecraft_angles, r_s = spacecraft_coms
    mpc_optimized_angles = np.load(save_dir_mpc+'optimized_angles.npy')
    ref_angles = joint_angles

    # fig = plt.figure()
    # plt.cla()
    # ax = plt.axes()
    # # ax.set_aspect('equal')
    # fig.set_facecolor('black')
    # ax.set_facecolor('black')
    # ax.grid(False)
    # ax.w_xaxis.pane.fill = False
    # ax.w_yaxis.pane.fill = False
    # ax.w_zaxis.pane.fill = False
    # plt.axis('off')

    render = Rendering()
    render.matplotlib_anim(mpc_optimized_angles, ref_angles)
    # render.anim(rs=spacecraft_coms, angs=spacecraft_angles, q=joint_angles, ref_path=ref_path, target=target_loc,
    #             end_eff_pos=end_eff_xyz, fig_save=False, reverse=False)
    # mlab.show()
    print('hi')