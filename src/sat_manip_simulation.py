import numpy as np
from eom_symbolic import kinematics, dynamics
from draw_satellite import SatellitePlotter
from dh_plotter import DH_plotter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


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
        Tc[0:3, 3] = np.array([size[0]/2, 0, 0])
        T1 = Ts @ Tc
        T_joint_manip = self.DHPlot.robot_DH_matrix(q)
        T_combined = np.zeros((T_joint_manip.shape[0]+2, 4, 4))
        T_combined[0, :, :], T_combined[1, :, :] = Ts, T1
        for i in range(2, T_joint_manip.shape[0]+2):
            T_combined[i, :, :] = T1 @ T_joint_manip[i-2, :, :]

        a = 2
        if ax is not None:
            X, Y, Z = self.satPlot.cuboid_data(pos, size)
            sh = X.shape
            x, y, z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            m = np.stack((x, y, z))
            mr = Rot @ m
            x, y, z = mr[0, :].reshape(sh), mr[1, :].reshape(sh), mr[2, :].reshape(sh)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, **kwargs)

            x, y, z = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
            ax.scatter(x, y, z, lw=5)
            for i in range(2, T_combined.shape[0]):
                ax.plot([x, T_combined[i, 0, 3]], [y, T_combined[i, 1, 3]], [z, T_combined[i, 2, 3]], lw=5)
                ax.scatter(T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3], 'gray', lw=5)
                x, y, z = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
            plt.xlabel('X')
            plt.ylabel('Y')
            # ax.axis('equal')
            ax.set_zlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_xlim(-a, a)

    def call_plot(self, pos, size, color, rot_ang, q):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        tt = [(pos[:, 0][0], pos[:, 0][1], pos[:, 0][2])]
        tr = rot_ang[:, 0]
        for i in range(rot_ang.shape[1]):
            temp = [(pos[:, i][0], pos[:, i][1], pos[:, i][2])]
            # temp = tt
            qi = q[:, i]
            for p, s, c in zip(temp, size, color):
                # self.satellite_namipulator(tr, qi,  pos=p, size=s, ax=ax, color=c)
                self.satellite_namipulator(rot_ang[:, i], qi,  pos=p, size=s, ax=ax, color=c)
            plt.pause(0.1)
            plt.cla()

    def simulation(self):
        sizes = [(0.5, 0.5, 0.5), (3, 3, 7)]
        colors = ["crimson", "limegreen"]
        r_s, ang_s, q = self.dyn.get_positions()
        self.call_plot(r_s, sizes, colors, ang_s, q)


if __name__ == '__main__':
    sim = Simulation()
    sim.simulation()
    plt.show()
