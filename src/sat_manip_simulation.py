import numpy as np
from eom_symbolic import kinematics, dynamics
from draw_satellite import SatellitePlotter
from dh_plotter import DH_plotter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
np.set_printoptions(precision=3)


class Simulation(object):

    def __init__(self, nDoF=3):
        self.nDoF = nDoF
        self.kin = kinematics(self.nDoF, robot='3DoF')
        self.dyn = dynamics(self.nDoF, robot='3DoF')
        self.satPlot = SatellitePlotter()
        self.DHPlot = DH_plotter(self.nDoF, robot='3DoF')

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
        r_s, ang_s, q, q_dot, t, pv_com = self.dyn.get_positions()
        self.call_plot(r_s, sizes, colors, ang_s, q,)  # Animation

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


class Background(Simulation):

    def __init__(self):
        super().__init__()
        print(self.nDoF)

    def earth(self):
        from mayavi import mlab

        # Display continents outline, using the VTK Builtin surface 'Earth'
        from mayavi.sources.builtin_surface import BuiltinSurface
        mlab.figure(1, bgcolor=(0., 0., 0.), fgcolor=(0, 0, 0),
                    size=(400, 400))
        mlab.clf()
        continents_src = BuiltinSurface(source='earth', name='Continents')
        # The on_ratio of the Earth source controls the level of detail of the
        # continents outline.
        continents_src.data_source.on_ratio = 2
        continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))

        ###############################################################################
        # Display a semi-transparent sphere, for the surface of the Earth

        # We use a sphere Glyph, throught the points3d mlab function, rather than
        # building the mesh ourselves, because it gives a better transparent
        # rendering.
        xc, yc, zc = 1, 0, 0
        center = np.array([xc, yc, zc])
        sphere = mlab.points3d(xc, yc, zc, scale_mode='none',
                               scale_factor=2,
                               color=(0., 0.77, 0.93),
                               resolution=50,
                               opacity=0.7,
                               name='Earth')

        # These parameters, as well as the color, where tweaked through the GUI,
        # with the record mode to produce lines of code usable in a script.
        sphere.actor.property.specular = 0.45
        sphere.actor.property.specular_power = 5
        # Backface culling is necessary for more a beautiful transparent rendering.
        sphere.actor.property.backface_culling = True

        ###############################################################################
        # Plot the equator and the tropiques
        theta = np.linspace(0, 2 * np.pi, 100)
        for angle in (- np.pi / 6, 0, np.pi / 6):
            x = xc + np.cos(theta) * np.cos(angle)
            y = yc + np.sin(theta) * np.cos(angle)
            z = zc + np.ones_like(theta) * np.sin(angle)

            mlab.plot3d(x, y, z, color=(1, 1, 1),
                        opacity=0.2, tube_radius=None)
        mlab.points3d(xc, yc, zc, color=(1,1,1), scale_factor=.125)
        mlab.view(63.4, 73.8, 4, [-0.05, 0, 0])
        mlab.show()

    def sim(self, ax=None):
        r_s, ang_s, q, q_dot, t, pv_com = self.dyn.get_positions()
        size = self.kin.size
        color = ["salmon", "limegreen", "crimson", ]

        # pos, size, color, rot_ang, q, pv_com = None,
        #
        # r_s, ang_s, q, q_dot, t, pv_com = self.dyn.get_positions()
        # self.call_plot(r_s, sizes, colors, ang_s, q, )  # Animation

        rot_ang, pos = ang_s, r_s,
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

if __name__ == '__main__':
    # sim = Simulation()
    # sim.simulation()
    bgd = Background()
    bgd.earth()
