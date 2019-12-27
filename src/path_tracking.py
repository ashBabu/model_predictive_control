import numpy as np
from fwd_kin import MayaviRendering, rot_mat_3d
from mayavi import mlab
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
plt.matplotlib.rc('font', **font)

save_dir = '/home/ash/Ash/repo/model_predictive_control/src/save_data_inv_kin/'
load_dir = '/home/ash/Ash/repo/model_predictive_control/src/save_data_inv_kin/data/'
save_dir_mpc = '/home/ash/Ash/repo/model_predictive_control/src/save_ffmpc/'


class Rendering:
    def __init__(self, nDoF=3, robot='3DoF', image_file='Nasa_blue_marble.jpg'):
        self.MR = MayaviRendering(nDoF=nDoF, robot=robot, image_file=image_file)

    @mlab.animate(delay=200)
    def anim(self, rs=None, angs=None, q=None, ref_path=None, fig_save=False, reverse=False):
        size = self.MR.kin.size[0]
        if reverse:
            rs, angs, q = np.hstack((rs, np.fliplr(rs))), np.hstack((angs, np.fliplr(angs))), np.hstack((q, np.fliplr(q)))
        mlab.figure(size=(1000, 800), bgcolor=(0., 0., 0.), )
        self.MR.manual_sphere(self.MR.image_file)
        p = [(rs[:, 0][0], rs[:, 0][1], rs[:, 0][2])][0]
        qi = q[:, 0]
        T_combined, x, y, z = self.MR.plot_satellite_manipulator(angs[:, 0], qi, pos=p, size=size)
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
                T_combined, x, y, z = self.MR.plot_satellite_manipulator(angs[:, i], qi, pos=p, size=size)
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

    @mlab.animate(delay=200)
    def anim1(self, rs=None, angs=None, q=None, b0=None, ref_path=None, rs1=None, angs1=None, q1=None, b01=None,
              ref_path1=None, fig_save=False, reverse=False):
        size = self.MR.kin.size[0]
        if reverse:
            rs, angs, q = np.hstack((rs, np.fliplr(rs))), np.hstack((angs, np.fliplr(angs))), \
                          np.hstack((q, np.fliplr(q)))
            rs1, angs1, q1 = np.hstack((rs1, np.fliplr(rs1))), np.hstack((angs1, np.fliplr(angs1))), \
                             np.hstack((q1, np.fliplr(q1)))
        mlab.figure(size=(1000, 800), bgcolor=(0., 0., 0.), )
        self.MR.manual_sphere(self.MR.image_file)
        newcg = np.zeros_like(rs)

        def find_newcg(ang, rs, rs1):
            for i in range(ang.shape[1]):
                Rot = rot_mat_3d(np.array(ang[:, i], dtype=float))
                newcg[:, i] = (rs[:, i] + Rot @ rs1) * 0.5
            return newcg

        newcg = find_newcg(angs, rs, rs1[:, 0])
        p = [(newcg[:, 0][0], newcg[:, 0][1], newcg[:, 0][2])][0]
        qi = q[:, 0]

        T_combined, x, y, z = self.MR.plot_satellite_manipulator(angs[:, 0], qi, pos=p, size=size, b0=b0)
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        robot_base = mlab.points3d(xx, yy, zz, mode='sphere', scale_factor=0.25, color=(0.5, 0.5, 0.5))
        satellite = mlab.plot3d(x, y, z, tube_radius=.1, color=(0.8, 0.9, 0.5))
        mlab.points3d(0, 0, 0, mode='sphere', scale_factor=0.4)
        xa, ya, za = T_combined[1:, 0, 3], T_combined[1:, 1, 3], T_combined[1:, 2, 3]
        manipulator = mlab.plot3d(xa, ya, za, color=(0.53, 0.74, 0.15), tube_radius=.1)
        mlab.plot3d(ref_path[:, 0], ref_path[:, 1], ref_path[:, 2], tube_radius=0.05, color=(0.4, 0.7, 0.25))

        qi1 = q1[:, 0]
        T_combined1, x1, y1, z1 = self.MR.plot_satellite_manipulator(angs[:, 0], qi, pos=p, size=size, b0=b01)
        xx1, yy1, zz1 = T_combined1[1, 0, 3], T_combined1[1, 1, 3], T_combined1[1, 2, 3]
        robot_base1 = mlab.points3d(xx1, yy1, zz1, mode='sphere', scale_factor=0.25, color=(0.5, 0.25, 0.15))
        xa1, ya1, za1 = T_combined1[1:, 0, 3], T_combined1[1:, 1, 3], T_combined1[1:, 2, 3]
        manipulator1 = mlab.plot3d(xa1, ya1, za1, color=(0.3, 0.4, 0.5), tube_radius=.1)

        if fig_save:
            mlab.savefig(save_dir+"animation/%02d.png" % 0)
        if angs.shape[1] > 3:
            n = angs.shape[1]
            for i in range(1, n):
                # p = [(rs[:, i][0], rs[:, i][1], rs[:, i][2])][0]
                p = [(newcg[:, i][0], newcg[:, i][1], newcg[:, i][2])][0]
                qi = q[:, i]
                T_combined, x, y, z = self.MR.plot_satellite_manipulator(angs[:, i], qi, pos=p, size=size, b0=b0)
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
                T_combined1, x1, y1, z1 = self.MR.plot_satellite_manipulator(angs[:, i], qi1, pos=p, size=size,
                                                                             b0=b01)
                xx1, yy1, zz1 = T_combined1[1, 0, 3], T_combined1[1, 1, 3], T_combined1[1, 2, 3]
                robot_base1.mlab_source.set(x=xx1, y=yy1, z=zz1)
                xa1, ya1, za1 = T_combined1[1:, 0, 3], T_combined1[1:, 1, 3], T_combined1[1:, 2, 3]
                manipulator1.mlab_source.set(x=xa1, y=ya1, z=za1)
                yield
        mlab.show()
        mlab.plot3d(ref_path1[:, 0], ref_path1[:, 1], ref_path1[:, 2], tube_radius=0.05, color=(0.14, 0.7, 0.95))
        newcg1 = find_newcg(angs1, rs1, rs[:, 0])
        # p1 = [(newcg1[:, 0][0], newcg1[:, 0][1], newcg1[:, 0][2])][0]
        if angs1.shape[1] > 3:
            n = angs1.shape[1]
            for i in range(1, n):
                # p1 = [(rs1[:, i][0], rs1[:, i][1], rs1[:, i][2])][0]
                p1 = [(newcg1[:, i][0], newcg1[:, i][1], newcg1[:, i][2])][0]
                qi1 = q1[:, i]
                T_combined1, x1, y1, z1 = self.MR.plot_satellite_manipulator(angs1[:, i], qi1, pos=p1, size=size, b0=b01)
                xx1, yy1, zz1 = T_combined1[1, 0, 3], T_combined1[1, 1, 3], T_combined1[1, 2, 3]
                xa1, ya1, za1 = T_combined1[1:, 0, 3], T_combined1[1:, 1, 3], T_combined1[1:, 2, 3]
                robot_base1.mlab_source.set(x=xx1, y=yy1, z=zz1)
                satellite.mlab_source.set(x=x1, y=y1, z=z1)
                manipulator1.mlab_source.set(x=xa1, y=ya1, z=za1)
                if fig_save:
                    mlab.savefig(save_dir+"animation/%02d.png" % i)
                # cam, foc = mlab.move()
                # mlab.move(-3, i/45, 1.2)
                # mlab.view(azimuth=-i/45, elevation=-i/85, distance=None, focalpoint=None, roll=None,
                # reset_roll=True, figure=None)
                T_combined, x, y, z = self.MR.plot_satellite_manipulator(angs1[:, i], q[:, 0], pos=p1, size=size, b0=b0)
                xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
                xa, ya, za = T_combined[1:, 0, 3], T_combined[1:, 1, 3], T_combined[1:, 2, 3]
                robot_base.mlab_source.set(x=xx, y=yy, z=zz)
                satellite.mlab_source.set(x=x, y=y, z=z)
                manipulator.mlab_source.set(x=xa, y=ya, z=za)
                yield

    def matplotlib_anim(self, opt_ang, ref_ang):
        fig = plt.figure(figsize=(8, 6))
        fig.set_facecolor((0.1, 0.2, 0.2, 0.3))
        # fig.set_facecolor('black')
        ax = plt.axes()
        ax.set_facecolor('black')
        ax.plot(ref_ang[0], 'r', label=r'$\displaystyle\theta_1$')
        ax.plot(ref_ang[1], 'b', label=r'$\displaystyle\theta_2$')
        ax.plot(ref_ang[2], 'green', label=r'$\displaystyle\theta_3$')
        ax.legend()

        for i in range(opt_ang.shape[1]):
            q1 = opt_ang[0][0:i]
            q2 = opt_ang[1][0:i]
            q3 = opt_ang[2][0:i]
            # t = np.arange(len(q1))
            ax.plot(q1, 'r*')
            ax.plot(q2, 'b*')
            ax.plot(q3, 'g*')
            plt.pause(0.2)
            # plt.savefig(save_dir_mpc+"animation/%02d.png" % i)


if __name__ == '__main__':
    end_eff_xyz, joint_angles, ref_path, spacecraft_angles, spacecraft_coms, target_loc = \
        np.load(load_dir+'end_eff_xyz.npy', allow_pickle=True), \
        np.load(load_dir+'joint_angs_inv_kin.npy', allow_pickle=True), \
        np.load(load_dir+'ref_path_xyz.npy', allow_pickle=True), \
        np.load(load_dir+'spacecraft_angs_inv_kin.npy', allow_pickle=True), \
        np.load(load_dir+'spacecraft_com_inv_kin.npy', allow_pickle=True), \
        np.load(load_dir+'target_loc_inv_kin.npy', allow_pickle=True)

    end_eff_xyz1, joint_angles1, ref_path1, spacecraft_angles1, spacecraft_coms1, target_loc1 = \
        np.load(load_dir+'end_eff_xyz1.npy', allow_pickle=True), \
        np.load(load_dir+'joint_angs_inv_kin1.npy', allow_pickle=True), \
        np.load(load_dir+'ref_path_xyz1.npy', allow_pickle=True), \
        np.load(load_dir+'spacecraft_angs_inv_kin1.npy', allow_pickle=True), \
        np.load(load_dir+'spacecraft_com_inv_kin1.npy', allow_pickle=True), \
        np.load(load_dir+'target_loc_inv_kin1.npy', allow_pickle=True)
    # q = joint_angles, ang_s = spacecraft_angles, r_s = spacecraft_coms
    mpc_optimized_angles = np.load(save_dir_mpc+'data/optimized_angles.npy')
    ref_angles = joint_angles
    ref_angles1 = joint_angles1

    # plt.plot(mpc_optimized_angles[0])
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
    b0, b01 = np.array([1.05, 1.05, 0]), np.array([-1.05, -1.05, 0])
    render = Rendering()
    # render.matplotlib_anim(mpc_optimized_angles, ref_angles)
    # render.anim(rs=spacecraft_coms, angs=spacecraft_angles, q=joint_angles, ref_path=ref_path,
    #             fig_save=True, reverse=False)
    render.anim1(rs=spacecraft_coms, angs=spacecraft_angles, q=joint_angles, ref_path=ref_path, b0=b0, b01=b01,
                 rs1=spacecraft_coms1, angs1=spacecraft_angles1, q1=joint_angles1, ref_path1=ref_path1, fig_save=False, reverse=True)
    mlab.show()
    # plt.show()
    print('hi')