import numpy as np
from eom_symbolic import kinematics, dynamics
from draw_satellite import SatellitePlotter
from dh_plotter import DH_plotter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3)


class Simulation(object):

    def __init__(self, nDoF=3):
        self.nDoF = nDoF
        self.kin = kinematics(self.nDoF, robot='3DoF')
        self.dyn = dynamics(self.nDoF, robot='3DoF')
        self.satPlot = SatellitePlotter()
        self.DHPlot = DH_plotter(self.nDoF, robot='3DoF')

    def satellite_namipulator(self, rot_ang, q, pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
        # rot_ang = [ang_sx, ang_sy, ang_sz], q = [q1, q2, ...]
        Rx = R.from_rotvec(rot_ang[0] * np.array([1, 0, 0]))
        Ry = R.from_rotvec(rot_ang[1] * np.array([0, 1, 0]))
        Rz = R.from_rotvec(rot_ang[2] * np.array([0, 0, 1]))
        Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
        Rot = Rx @ Ry @ Rz
        ang_zb = self.kin.ang_zb
        Rz_b = R.from_rotvec(ang_zb * np.array([0, 0, 1]))  # Refer pict.
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
        a = 3.5
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        print((xx, yy, zz))
        ax.scatter(xx, yy, zz, lw=5)
        if ax is not None:
            xxx, yyy, zzz = self.satPlot.cube_plot(pos, size)
           ###############
            # X, Y, Z = self.satPlot.cuboid_data(pos, size)
            # sh = X.shape
            # # The below steps are included to rotate the cube
            # x, y, z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            # m = np.stack((x, y, z))
           ################
            m = np.vstack((xxx, yyy, zzz))
            mr = Rot @ m

            # x, y, z = mr[0, :].reshape(sh), mr[1, :].reshape(sh), mr[2, :].reshape(sh)
            # ax.plot_surface(x, y, z, rstride=1, cstride=1, **kwargs)
            x, y, z = mr[0, :], mr[1, :], mr[2, :]
            ax.plot(x, y, z, lw=5)
            ax.scatter(0, 0, 0, lw=5)
            ax.scatter(pos[0], pos[1], pos[2], lw=5)

            for i in range(2, T_combined.shape[0]):
                ax.plot([xx, T_combined[i, 0, 3]], [yy, T_combined[i, 1, 3]], [zz, T_combined[i, 2, 3]], lw=5)
                ax.scatter(T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3], lw=5)
                xx, yy, zz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
            plt.xlabel('X')
            plt.ylabel('Y')
            # ax.axis('equal')
            # ax.view_init(elev=64., azim=67.)
            ax.set_zlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_xlim(-a, a)
            print(j_Tb,)
            print('hi')

    def call_plot(self, pos, size, color, rot_ang, q):
        # rot_ang is a 3 x t vector of the rotation angles of the spacecraft. q is manipulator angles
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        for i in range(rot_ang.shape[1]):
            temp = [(pos[:, i][0], pos[:, i][1], pos[:, i][2])]
            qi = q[:, i]
            plt.cla()
            for p, s, c in zip(temp, size, color):
                self.satellite_namipulator(rot_ang[:, i], qi,  pos=p, size=s, ax=ax, color=c)
            plt.pause(0.1)
            # plt.savefig("%i" % i)

    def simulation(self):
        sizes = self.kin.size
        colors = ["salmon", "limegreen", "crimson", ]
        r_s, ang_s, q, q_dot, t = self.dyn.get_positions()
        self.call_plot(r_s, sizes, colors, ang_s, q)  # Animation

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
        plt.plot(t, ang_sx, label='satellite_ang_x_position')
        plt.plot(t, ang_sy, label='satellite_ang_y_position')
        plt.plot(t, ang_sz, label='satellite_ang_z_position')
        plt.legend()

        fig3 = plt.figure()
        plt.plot(t, q1_dot, label='q1_dot')
        plt.plot(t, q2_dot, label='q2_dot')
        plt.plot(t, q3_dot, label='q3_dot')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    sim = Simulation()
    sim.simulation()
