import numpy as np
from eom_symbolic import kinematics, dynamics
from draw_satellite import SatellitePlotter
from dh_plotter import DH_plotter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=4)


class Simulation(object):

    def __init__(self, nDoF=3):
        self.nDoF = nDoF
        self.kin = kinematics(self.nDoF, robot='3DoF')
        self.dyn = dynamics(self.nDoF, robot='3DoF')
        self.satPlot = SatellitePlotter()
        self.DHPlot = DH_plotter(self.nDoF, robot='3DoF')

    def satellite_namipulator(self, rot_ang, q, pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
        # rot_ang = [ang_sx, ang_sy, ang_sz], q = [q1, q2]
        Rx = R.from_rotvec(rot_ang[0] * np.array([1, 0, 0]))
        Ry = R.from_rotvec(rot_ang[1] * np.array([0, 1, 0]))
        Rz = R.from_rotvec(rot_ang[2] * np.array([0, 0, 1]))
        Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
        Rot = Rx @ Ry @ Rz
        Rz90 = R.from_rotvec(-np.pi/2 * np.array([0, 0, 1]))
        Rz90 = Rz90.as_dcm()
        Ts, Tc = np.eye(4), np.eye(4)
        Ts[0:3, 0:3], Tc[0:3, 0:3] = Rot, Rz90
        Ts[0:3, 3] = np.array([pos[0], pos[1], pos[2]])
        Tc[0:3, 3] = np.array([size[0]/2+0.05, 0, 0])  # the vector b's x comp and size[0]/2 has to be same
        # Tc[0:3, 3] = np.array([0.7, 0, 0])  # the vector b's x comp and size[0]/2 has to be same
        T1 = Ts @ Tc
        T_joint_manip, Ti = self.DHPlot.robot_DH_matrix(q)
        T_combined = np.zeros((T_joint_manip.shape[0]+2, 4, 4))
        T_combined[0, :, :], T_combined[1, :, :] = Ts, T1
        for i in range(2, T_joint_manip.shape[0]+2):
            T_combined[i, :, :] = T1 @ T_joint_manip[i-2, :, :]

        a = 2.5
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        ax.scatter(xx, yy, zz, lw=5)
        if ax is not None:
            X, Y, Z = self.satPlot.cuboid_data(pos, size)
            sh = X.shape
            x, y, z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            m = np.stack((x, y, z))
            mr = Rot @ m
            x, y, z = mr[0, :].reshape(sh), mr[1, :].reshape(sh), mr[2, :].reshape(sh)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, **kwargs)

            for i in range(2, T_combined.shape[0]):
                ax.plot([xx, T_combined[i, 0, 3]], [yy, T_combined[i, 1, 3]], [zz, T_combined[i, 2, 3]], lw=5)
                ax.scatter(T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3], 'gray', lw=5)
                xx, yy, zz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
            plt.xlabel('X')
            plt.ylabel('Y')
            # ax.axis('equal')
            ax.set_zlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_xlim(-a, a)

    def call_plot(self, pos, size, color, rot_ang, q):
        # rot angle is a 3 x t vector of the rotation angles of the spacecraft. q is manipulator angles
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        for i in range(rot_ang.shape[1]):
            temp = [(pos[:, i][0], pos[:, i][1], pos[:, i][2])]
            qi = q[:, i]
            for p, s, c in zip(temp, size, color):
                self.satellite_namipulator(rot_ang[:, i], qi,  pos=p, size=s, ax=ax, color=c)
            plt.pause(0.1)
            # plt.savefig("%i" % i)
            plt.cla()

    def simulation(self):
        sizes = self.kin.sizes
        colors = ["limegreen", "crimson", ]
        r_s, ang_s, q, q_dot, t = self.dyn.get_positions()
        self.call_plot(r_s, sizes, colors, ang_s, q)

        r_sx, r_sy, r_sz = r_s[0, :], r_s[1, :], r_s[2, :]
        ang_sx, ang_sy, ang_sz = ang_s[0, :], ang_s[1, :], ang_s[2, :]
        q1, q2, q3 = q[0, :], q[1, :], q[2, :]
        q1_dot, q2_dot, q3_dot = q_dot[0, :], q_dot[1, :], q_dot[2, :]

        fig1 = plt.figure(1)
        plt.plot(t, r_sx)
        plt.plot(t, r_sy)
        plt.plot(t, r_sz)

        fig2 = plt.figure(2)
        plt.plot(t, ang_sx)
        plt.plot(t, ang_sy)
        plt.plot(t, ang_sz)

        fig3 = plt.figure(3)
        plt.plot(t[:-1], q1_dot)
        plt.plot(t[:-1], q2_dot)
        plt.plot(t[:-1], q3_dot)
        plt.show()


if __name__ == '__main__':
    sim = Simulation()
    sim.simulation()
