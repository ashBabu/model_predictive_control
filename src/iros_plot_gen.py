import numpy as np
import matplotlib.pyplot as plt
from dh_plotter import DH_plotter
from draw_satellite import SatellitePlotter
from fwd_kin import MayaviRendering
np.set_printoptions(precision=3)
from scipy.spatial.transform import Rotation as R
from mayavi import mlab
from arrow3d import Arrow3D


def rot_mat_3d(*args):
    if isinstance(args, tuple):
        ang_x, ang_y, ang_z = args[0]
    else:
        ang_x, ang_y, ang_z = args
    Rx = R.from_rotvec(ang_x * np.array([1, 0, 0]))
    Ry = R.from_rotvec(ang_y * np.array([0, 1, 0]))
    Rz = R.from_rotvec(ang_z * np.array([0, 0, 1]))
    Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
    return Rx @ Ry @ Rz


class ForwardKin:
    def __init__(self, nDoF=7, robot='7DoF',  size=2.1):
        self.nDoF = nDoF
        self.size = [(size, size, size)]
        self.DHPlot = DH_plotter(nDoF=self.nDoF, robot=robot)
        self.satPlot = SatellitePlotter()

    def robot_base_ang(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = b0
        an = (np.arctan2(b0[1], b0[0]) * 180 / np.pi) % 360  # gives an angles in 0 - 360 degrees
        an = (an - 90.) * np.pi / 180  # This means y axis is along the robot's first link as per DH
        ang_xb, ang_yb, ang_zb = 0., 0., an
        ang_b = np.array([ang_xb, ang_yb, ang_zb], dtype=float)
        return ang_b

    def rotation_matrix(self, ang):
        rx = [[1, 0, 0], [0, np.cos(ang[0]), -np.sin(ang[0])], [0, np.sin(ang[0]), np.cos(ang[0])]]
        ry = [[np.cos(ang[1]), 0, np.sin(ang[1])], [0, 1, 0], [-np.sin(ang[1]), 0, np.cos(ang[1])]]
        rz = [[np.cos(ang[2]), -np.sin(ang[2]), 0], [np.sin(ang[2]), np.cos(ang[2]), 0], [0, 0, 1]]
        return np.array(rx), np.array(ry), np.array(rz)

    def satellite_namipulator(self, rot_ang, q, pos, size, ax=None, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = b0
        # rot_ang = [ang_sx, ang_sy, ang_sz], q = [q1, q2, ...]
        Rot = rot_mat_3d(rot_ang)
        ang_b = self.robot_base_ang(b0)
        ang_zb = ang_b[2]
        Rz_b = R.from_rotvec(ang_zb * np.array([0, 0, 1]))  # Refer pic.
        Rz_b = Rz_b.as_dcm()
        j_Ts, s_Tb = np.eye(4), np.eye(
            4)  # j_Ts = transf. matrix from inertial {j} to satellite; s_Tb = transf. matrix
        # from satellite to robot base
        j_Ts[0:3, 0:3], s_Tb[0:3, 0:3] = Rot, Rz_b
        j_Ts[0:3, 3] = np.array([pos[0], pos[1], pos[2]])
        s_Tb[0:3, 3] = b0  # the vector b's x comp and size[0]/2 has to be same
        # s_Tb[0:3, 3] = np.array([0.7, 0, 0])  # the vector b's x comp and size[0]/2 has to be same
        j_Tb = j_Ts @ s_Tb
        T_joint_manip, Ti = self.DHPlot.robot_DH_matrix(q)
        T_combined = np.zeros((T_joint_manip.shape[0] + 2, 4, 4))
        T_combined[0, :, :], T_combined[1, :, :] = j_Ts, j_Tb
        for i in range(2, T_combined.shape[0]):
            T_combined[i, :, :] = j_Tb @ T_joint_manip[i - 2, :, :]
        a = 1.5
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]
        ax.scatter(xx, yy, zz, lw=5)
        if ax is not None:
            xxx, yyy, zzz = self.satPlot.cube_data2((0, 0, 0), size)
            m = np.vstack((xxx, yyy, zzz))
            mr = Rot @ m + np.array([[pos[0]], [pos[1]], [pos[2]]])
            x, y, z = mr[0, :], mr[1, :], mr[2, :]
            ax.plot(x, y, z, 'g', lw=5)
            a = Arrow3D([0, 0], [0, 0], [0, 0.7], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color="b")
            ax.add_artist(a)
            ax.scatter(0, 0, 0, lw=5)
            ll = T_combined.shape[0]
            scl = 0.8
            for i in range(2, T_combined.shape[0]):
                jx, jy, jz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
                ax.plot([xx, jx], [yy, jy], [zz, jz], lw=10)
                ax.add_artist(Arrow3D([jx, jx + scl*T_combined[i, 0, 2]], [jy, jy+ scl*T_combined[i, 1, 2]], [jz, jz+ + scl*T_combined[i, 2, 2]], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color="b"))  # z (joint) axis
                ax.add_artist(Arrow3D([jx, jx + scl*T_combined[i, 0, 0]], [jy, jy+ scl*T_combined[i, 1, 0]], [jz, jz+ + scl*T_combined[i, 2, 0]], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color="r"))  # x  axis
                if i < ll-1:
                    ax.scatter(T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3], 'gray', lw=10)
                xx, yy, zz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
                plt.pause(0.05)
            plt.xlabel('X')
            plt.ylabel('Y')
            # ax.axis('equal')
            ax.view_init(elev=54., azim=-140.)
            # ax.set_zlim(-a, a)
            # ax.set_ylim(-a, a)
            # ax.set_xlim(-a, a)

    def call_plot(self, pos, size, color, rot_ang, q, pv_com=None, ax=None, b0=None):
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
                    self.satellite_namipulator(rot_ang[:, i], qi, pos=p, size=s, ax=ax, b0=b0)
                plt.pause(0.05)
        else:
            # n = len(rot_ang)
            # for i in range(n):
            temp = [(pos[0][0], pos[1][0], pos[2][0])]
            plt.cla()
            # if isinstance(pv_com, (list, tuple, np.ndarray)):
            #     ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
            for p, s, c in zip(temp, size, color):
                self.satellite_namipulator(rot_ang, q, pos=p, size=s, ax=ax, b0=b0)
            # plt.pause(0.05)

            # plt.savefig("/home/ar0058/Ash/repo/model_predictive_control/src/animation/%02d.png" % i)
            # print('hi')

    def simulation(self, r_s=None, ang_s=None, q=None, b0=None):
        sizes = self.size
        colors = ["salmon", "limegreen", "crimson", ]
        self.call_plot(r_s, sizes, colors, ang_s, q, b0=b0)  # Animation


if __name__ == '__main__':
    pi = np.pi
    q1 = np.array([0., 5 * pi / 4, 0., 0., 0., 0., 0.])
    b0 = np.array([-1.05, 1.05, 0])

    fwd_kin = ForwardKin(nDoF=7, robot='7DoF')
    i = 1  # int(input('1:Matplotlib, 2: Mayavi'))
    rs1, ang_s1 = np.zeros(3).reshape(-1, 1), np.zeros(3).reshape(-1, 1)
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
