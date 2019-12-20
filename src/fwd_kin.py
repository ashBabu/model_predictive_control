import numpy as np
from eom_symbolic import kinematics, dynamics
from draw_satellite import SatellitePlotter
from dh_plotter import DH_plotter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
np.set_printoptions(precision=3)
from mayavi import mlab
from tvtk.api import tvtk

save_dir = '/home/ash/Ash/repo/model_predictive_control/src/save_dir_fwd_kin/'


class Simulation(object):

    def __init__(self, nDoF=3, robot='3DoF'):
        self.nDoF = nDoF
        self.kin = kinematics(self.nDoF, robot=robot)
        self.dyn = dynamics(self.nDoF, robot=robot)
        self.satPlot = SatellitePlotter()
        self.DHPlot = DH_plotter(self.nDoF, robot=robot)

    def rotation_matrix(self, ang):
        rx = [[1, 0, 0], [0, np.cos(ang[0]), -np.sin(ang[0])], [0, np.sin(ang[0]), np.cos(ang[0])]]
        ry = [[np.cos(ang[1]), 0, np.sin(ang[1])], [0, 1, 0], [-np.sin(ang[1]), 0, np.cos(ang[1])]]
        rz = [[np.cos(ang[2]), -np.sin(ang[2]), 0], [np.sin(ang[2]), np.cos(ang[2]), 0], [0, 0, 1]]
        return np.array(rx), np.array(ry), np.array(rz)

    def satellite_namipulator(self, rot_ang, q, pos, size, ax=None, **kwargs):
        # rot_ang = [ang_sx, ang_sy, ang_sz], q = [q1, q2, ...]
        Rx = R.from_rotvec(rot_ang[0] * np.array([1, 0, 0]))
        Ry = R.from_rotvec(rot_ang[1] * np.array([0, 1, 0]))
        Rz = R.from_rotvec(rot_ang[2] * np.array([0, 0, 1]))
        Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
        # Rx, Ry, Rz = self.rotation_matrix(rot_ang)
        Rot = Rx @ Ry @ Rz
        ang_zb = self.kin.ang_zb
        Rz_b = R.from_rotvec(ang_zb * np.array([0, 0, 1]))  # Refer pic.
        Rz_b = Rz_b.as_dcm()
        j_Ts, s_Tb = np.eye(4), np.eye(4)  # j_Ts = transf. matrix from inertial {j} to satellite; s_Tb = transf. matrix
        # from satellite to robot base
        j_Ts[0:3, 0:3], s_Tb[0:3, 0:3] = Rot, Rz_b
        j_Ts[0:3, 3] = np.array([pos[0], pos[1], pos[2]])
        s_Tb[0:3, 3] = self.kin.b0  # the vector b's x comp and size[0]/2 has to be same
        # s_Tb[0:3, 3] = np.array([0.7, 0, 0])  # the vector b's x comp and size[0]/2 has to be same
        j_Tb = j_Ts @ s_Tb
        T_joint_manip, Ti = self.DHPlot.robot_DH_matrix(q)
        T_combined = np.zeros((T_joint_manip.shape[0]+2, 4, 4))
        T_combined[0, :, :], T_combined[1, :, :] = j_Ts, j_Tb
        for i in range(2, T_joint_manip.shape[0]+2):
            T_combined[i, :, :] = j_Tb @ T_joint_manip[i-2, :, :]
        a = 4.2
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        ax.scatter(xx, yy, zz, lw=5)
        if ax is not None:
            xxx, yyy, zzz = self.satPlot.cube_data2((0, 0, 0), size)
            m = np.vstack((xxx, yyy, zzz))
            mr = Rot @ m + np.array([[pos[0]], [pos[1]], [pos[2]]])
            x, y, z = mr[0, :], mr[1, :], mr[2, :]
            ax.plot(x, y, z, lw=5)
            ax.scatter(0, 0, 0, lw=5)
            # ax.scatter(pos[0], pos[1], pos[2], lw=5)

            for i in range(2, T_combined.shape[0]):
                ax.plot([xx, T_combined[i, 0, 3]], [yy, T_combined[i, 1, 3]], [zz, T_combined[i, 2, 3]], lw=5)
                ax.scatter(T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3], lw=5)
                xx, yy, zz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
            plt.xlabel('X')
            plt.ylabel('Y')
            # ax.axis('equal')
            ax.view_init(elev=83., azim=-83.)
            ax.set_zlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_xlim(-a, a)

    def call_plot(self, pos, size, color, rot_ang, q, pv_com=None, ax=None):
        # rot_ang is a 3 x t vector of the rotation angles of the spacecraft. q is manipulator angles
        if not ax:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # ax.set_aspect('equal')
            if rot_ang.shape[1] > 3:
                n = rot_ang.shape[1]
                for i in range(n):
                    temp = [(pos[:, i][0], pos[:, i][1], pos[:, i][2])]
                    qi = q[:, i]
                    plt.cla()
                    if isinstance(pv_com, (list, tuple, np.ndarray)):
                        ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
                    for p, s, c in zip(temp, size, color):
                        self.satellite_namipulator(rot_ang[:, i], qi, pos=p, size=s, ax=ax, color=c)
                    plt.pause(0.05)
            else:
                # n = len(rot_ang)
                # for i in range(n):
                temp = [(pos[0][0], pos[1][0], pos[2][0])]
                plt.cla()
                # if isinstance(pv_com, (list, tuple, np.ndarray)):
                #     ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
                for p, s, c in zip(temp, size, color):
                    self.satellite_namipulator(rot_ang, q, pos=p, size=s, ax=ax, color=c)
                # plt.pause(0.05)

            # plt.savefig("/home/ar0058/Ash/repo/model_predictive_control/src/animation/%02d.png" % i)
            # print('hi')

    def simulation(self):
        sizes = self.kin.size
        colors = ["salmon", "limegreen", "crimson", ]
        # r_s, ang_s, q, q_dot, t, pv_com = self.dyn.get_positions()
        # np.save('t.npy', t, allow_pickle=True)

        r_s, ang_s, q, q_dot, pv_com = np.load('rs.npy', allow_pickle=True), np.load('ang_s.npy',
                                                                                     allow_pickle=True), np.load(
            'q.npy', allow_pickle=True), \
                                       np.load('q_dot.npy', allow_pickle=True), np.load('pv_com.npy', allow_pickle=True)
        t = np.load('t.npy', allow_pickle=True)
        self.call_plot(r_s, sizes, colors, ang_s, q, )  # Animation
        r_sx, r_sy, r_sz = r_s[0, :], r_s[1, :], r_s[2, :]
        ang_sx, ang_sy, ang_sz = ang_s[0, :], ang_s[1, :], ang_s[2, :]
        q1, q2, q3 = q[0, :], q[1, :], q[2, :]
        q1_dot, q2_dot, q3_dot = q_dot[0, :], q_dot[1, :], q_dot[2, :]

        fig1 = plt.figure()
        plt.plot(t, r_sx, label='satellite_x_position')
        plt.plot(t, r_sy, label='satellite_y_position')
        plt.plot(t, r_sz, label='satellite_z_position')
        plt.legend()

        fig2 = plt.figure()
        plt.plot(t, ang_sx, label='satellite_x_rotation')
        plt.plot(t, ang_sy, label='satellite_y_rotation')
        plt.plot(t, ang_sz, label='satellite_z_rotation')
        plt.legend()

        fig3 = plt.figure()
        plt.plot(t, q1_dot, label='q1_dot')
        plt.plot(t, q2_dot, label='q2_dot')
        plt.plot(t, q3_dot, label='q3_dot')
        plt.legend()
        plt.show()


class MayaviRendering:
    def __init__(self, nDoF=3, robot='3DoF', image_file='Nasa_blue_marble.jpg'):
        self.satPlot = SatellitePlotter(nDoF=nDoF, robot=robot)
        self.kin = kinematics(nDoF=nDoF, robot=robot)
        self.dyn = dynamics(nDoF=nDoF, robot=robot)
        self.DHPlot = DH_plotter(nDoF=nDoF, robot=robot)
        self.image_file = image_file

    def manual_sphere(self, image_file):
        # caveat 1: flip the input image along its first axis
        img = plt.imread(image_file)  # shape (N,M,3), flip along first dim
        outfile = image_file.replace('.jpg', '_flipped.jpg')
        # flip output along first dim to get right chirality of the mapping
        img = img[::-1, ...]
        plt.imsave(outfile, img)
        image_file = outfile  # work with the flipped file from now on

        # parameters for the sphere
        R = 3  # radius of the sphere
        Nrad = 180  # points along theta and phi
        phi = np.linspace(0, 2 * np.pi, Nrad)  # shape (Nrad,)
        theta = np.linspace(0, np.pi, Nrad)    # shape (Nrad,)
        phigrid, thetagrid = np.meshgrid(phi, theta)  # shapes (Nrad, Nrad)

        xc, yc, zc = 8, 0, 0
        # compute actual points on the sphere
        x = xc + R * np.sin(thetagrid) * np.cos(phigrid)
        y = yc + R * np.sin(thetagrid) * np.sin(phigrid)
        z = zc + R * np.cos(thetagrid)

        # create figure

        # mlab.figure(size=(600, 600),)

        # create meshed sphere
        mesh = mlab.mesh(x, y, z)
        mesh.actor.actor.mapper.scalar_visibility = False
        mesh.actor.enable_texture = True  # probably redundant assigning the texture later

        # load the (flipped) image for texturing
        img = tvtk.JPEGReader(file_name=image_file)
        texture = tvtk.Texture(input_connection=img.output_port, interpolate=0, repeat=0)
        mesh.actor.actor.texture = texture

        # tell mayavi that the mapping from points to pixels happens via a sphere
        mesh.actor.tcoord_generator_mode = 'sphere' # map is already given for a spherical mapping
        cylinder_mapper = mesh.actor.tcoord_generator
        # caveat 2: if prevent_seam is 1 (default), half the image is used to map half the sphere
        cylinder_mapper.prevent_seam = 0 # use 360 degrees, might cause seam but no fake data
        #cylinder_mapper.center = np.array([0,0,0])  # set non-trivial center for the mapping sphere if necessary

    def manual_sphere2(self, image_file):
        # caveat 1: flip the input image along its first axis
        img = plt.imread(image_file)  # shape (N,M,3), flip along first dim
        outfile = image_file.replace('.jpg', '_flipped.jpg')
        # flip output along first dim to get right chirality of the mapping
        img = img[::-1,...]
        plt.imsave(outfile, img)
        image_file = outfile  # work with the flipped file from now on

        # parameters for the sphere
        R = 5 # radius of the sphere
        Nrad = 180 # points along theta and phi
        phi = np.linspace(0, 2 * np.pi, Nrad)  # shape (Nrad,)
        theta = np.linspace(0, np.pi, Nrad)    # shape (Nrad,)
        phigrid,thetagrid = np.meshgrid(phi, theta) # shapes (Nrad, Nrad)

        xc, yc, zc = 20, 0, 0
        # compute actual points on the sphere
        x = xc + R * np.sin(thetagrid) * np.cos(phigrid)
        y = yc + R * np.sin(thetagrid) * np.sin(phigrid)
        z = zc + R * np.cos(thetagrid)

        # create figure
        mlab.figure(size=(1600, 1600), bgcolor=(0., 0., 0.),)

        # create meshed sphere
        mesh = mlab.mesh(x,y,z)
        mesh.actor.actor.mapper.scalar_visibility = False
        mesh.actor.enable_texture = True  # probably redundant assigning the texture later

        # load the (flipped) image for texturing
        img = tvtk.JPEGReader(file_name=image_file)
        texture = tvtk.Texture(input_connection=img.output_port, interpolate=0, repeat=0)
        mesh.actor.actor.texture = texture

        # tell mayavi that the mapping from points to pixels happens via a sphere
        mesh.actor.tcoord_generator_mode = 'sphere' # map is already given for a spherical mapping
        cylinder_mapper = mesh.actor.tcoord_generator
        # caveat 2: if prevent_seam is 1 (default), half the image is used to map half the sphere
        cylinder_mapper.prevent_seam = 0 # use 360 degrees, might cause seam but no fake data
        #cylinder_mapper.center = np.array([0,0,0])

    def plot_satellite_namipulator(self, rot_ang, q, pos, size):
        # rot_ang = [ang_sx, ang_sy, ang_sz], q = [q1, q2, ...]
        Rx = R.from_rotvec(rot_ang[0] * np.array([1, 0, 0]))
        Ry = R.from_rotvec(rot_ang[1] * np.array([0, 1, 0]))
        Rz = R.from_rotvec(rot_ang[2] * np.array([0, 0, 1]))
        Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
        Rot = Rx @ Ry @ Rz
        ang_zb = self.kin.ang_zb
        Rz_b = R.from_rotvec(ang_zb * np.array([0, 0, 1]))  # Refer pic.
        Rz_b = Rz_b.as_dcm()
        j_Ts, s_Tb = np.eye(4), np.eye(4)  # j_Ts = transf. matrix from inertial {j} to satellite; s_Tb = transf. matrix
        # from satellite to robot base
        j_Ts[0:3, 0:3], s_Tb[0:3, 0:3] = Rot, Rz_b
        j_Ts[0:3, 3] = np.array([pos[0], pos[1], pos[2]])
        s_Tb[0:3, 3] = self.kin.b0  # the vector b's x comp and size[0]/2 has to be same
        j_Tb = j_Ts @ s_Tb
        T_joint_manip, Ti = self.DHPlot.robot_DH_matrix(q)
        T_combined = np.zeros((T_joint_manip.shape[0]+2, 4, 4))
        T_combined[0, :, :], T_combined[1, :, :] = j_Ts, j_Tb
        for i in range(2, T_joint_manip.shape[0]+2):
            T_combined[i, :, :] = j_Tb @ T_joint_manip[i-2, :, :]
        xxx, yyy, zzz = self.satPlot.cube_data2((0, 0, 0), size)
        m = np.vstack((xxx, yyy, zzz))
        mr = Rot @ m + np.array([[pos[0]], [pos[1]], [pos[2]]])
        x, y, z = mr[0, :], mr[1, :], mr[2, :]
        return T_combined, x, y, z

    @mlab.animate(delay=100)
    def anim(self, rs=None, angs=None, q=None, fig_save=False, reverse=False):
        size = self.kin.size[0]
        if reverse:
            rs, angs, q = np.hstack((rs, np.fliplr(rs))), np.hstack((angs, np.fliplr(angs))), np.hstack((q, np.fliplr(q)))
        mlab.figure(size=(1000, 800), bgcolor=(0., 0., 0.), )
        self.manual_sphere(self.image_file)
        p = [(rs[:, 0][0], rs[:, 0][1], rs[:, 0][2])][0]
        qi = q[:, 0]
        T_combined, x, y, z = self.plot_satellite_namipulator(angs[:, 0], qi, pos=p, size=size)
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        robot_base = mlab.points3d(xx, yy, zz, mode='sphere', scale_factor=0.25, color=(0.5, 0.5, 0.5))
        satellite = mlab.plot3d(x, y, z, tube_radius=.1, color=(0.8, 0.9, 0.5))
        mlab.points3d(0, 0, 0, mode='point', scale_factor=4)
        xa, ya, za = T_combined[1:, 0, 3], T_combined[1:, 1, 3], T_combined[1:, 2, 3]
        manipulator = mlab.plot3d(xa, ya, za, color=(0.3, 0.4, 0.5), tube_radius=.1)
        if fig_save:
            mlab.savefig(save_dir+"animation/%02d.png" % 0)
        if angs.shape[1] > 3:
            n = angs.shape[1]
            for i in range(1, n):
                p = [(rs[:, i][0], rs[:, i][1], rs[:, i][2])][0]
                qi = q[:, i]
                T_combined, x, y, z = self.plot_satellite_namipulator(angs[:, i], qi, pos=p, size=size)
                xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
                xa, ya, za = T_combined[1:, 0, 3], T_combined[1:, 1, 3], T_combined[1:, 2, 3]
                robot_base.mlab_source.set(x=xx, y=yy, z=zz)
                satellite.mlab_source.set(x=x, y=y, z=z)
                manipulator.mlab_source.set(x=xa, y=ya, z=za)
                if fig_save:
                    mlab.savefig(save_dir+"animation/%02d.png" % i)
                yield

    def get_plots(self, r_s, ang_s, q, q_dot, t):
        r_sx, r_sy, r_sz = r_s[0, :], r_s[1, :], r_s[2, :]
        ang_sx, ang_sy, ang_sz = ang_s[0, :], ang_s[1, :], ang_s[2, :]
        q1, q2, q3 = q[0, :], q[1, :], q[2, :]
        q1_dot, q2_dot, q3_dot = q_dot[0, :], q_dot[1, :], q_dot[2, :]
    
        fig1 = plt.figure()
        plt.plot(t, r_sx, label='satellite_x_position')
        plt.plot(t, r_sy, label='satellite_y_position')
        plt.plot(t, r_sz, label='satellite_z_position')
        plt.legend()
    
        fig2 = plt.figure()
        plt.plot(t, ang_sx, label='satellite_x_rotation')
        plt.plot(t, ang_sy, label='satellite_y_rotation')
        plt.plot(t, ang_sz, label='satellite_z_rotation')
        plt.legend()
    
        fig3 = plt.figure()
        plt.plot(t, q1_dot, label='q1_dot')
        plt.plot(t, q2_dot, label='q2_dot')
        plt.plot(t, q3_dot, label='q3_dot')
        plt.legend()


if __name__ == '__main__':
    i = int(input('1:Matplotlib, 2: Mayavi'))
    ######  To find various parameters #######
    # r_s = COM of satellite wrt inertial (3 x time), ang_s = x, y, z rotations of satellite wrt inertial (3 x time)
    # q = manipulator joint angles (3 x time), pv_com = COM of all links including satellite (time x 3 x nLinks)
    # r_s, ang_s, q, q_dot, t, pv_com = dyn.get_positions()
    # np.save('rs.npy', r_s, allow_pickle=True), np.save('ang_s.npy', ang_s, allow_pickle=True), np.save('q.npy', q, allow_pickle=True), \
    # np.save('q_dot.npy', q_dot, allow_pickle=True), np.save('pv_com.npy', pv_com, allow_pickle=True)

    t = np.load(save_dir+'data/t.npy', allow_pickle=True)
    r_s, ang_s, q, q_dot, pv_com = np.load(save_dir+'data/rs.npy', allow_pickle=True), \
                                   np.load(save_dir+'data/ang_s.npy', allow_pickle=True),\
                                       np.load(save_dir+'data/q.npy', allow_pickle=True),\
                                   np.load(save_dir+'data/q_dot.npy', allow_pickle=True),\
                                    np.load(save_dir+'data/pv_com.npy', allow_pickle=True)
    if i == 1:
        # For Matplotlib simulation:
        sim = Simulation()
        sim.simulation()
    # For Mayavi Rendering:
    elif i == 2:
        MR = MayaviRendering()
        MR.get_plots(r_s, ang_s, q, q_dot, t)
        MR.anim(rs=r_s, angs=ang_s, q=q, fig_save=False, reverse=True)
        mlab.show()
