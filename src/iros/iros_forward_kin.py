import numpy as np
import matplotlib.pyplot as plt
from iros_dh_plotter import DH_plotter
from iros_draw_satellite import SatellitePlotter
# from fwd_kin import MayaviRendering
from iros_7DoF_EOM import Dynamics, Kinematics
np.set_printoptions(precision=3)
from scipy.spatial.transform import Rotation as R
# from mayavi import mlab
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
        self.dyn = Dynamics(nDoF=self.nDoF, robot=robot)

    def robot_base_ang(self, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = b0
        an = (np.arctan2(b0[1], b0[0]) * 180 / np.pi) % 360  # gives an angles in 0 - 360 degrees
        an = (an - 90.) * np.pi / 180  # This means y axis is along the robot's first link as per DH
        ang_xb, ang_yb, ang_zb = 0., 0., an
        ang_b = np.array([ang_xb, ang_yb, ang_zb], dtype=float)
        return ang_b

    def satellite_namipulator(self, rot_ang=None, q=None, rs=None, size=None, ax=None, b0=None):
        if not isinstance(b0, (list, tuple, np.ndarray)):
            b0 = self.dyn.kin.b0
        # rot_ang = [ang_sx, ang_sy, ang_sz], q = [q1, q2, ...]
        Rot = rot_mat_3d(rot_ang)
        T_combined = self.dyn.transf_from_inertial(q=q, ang_s=rot_ang, b0=b0)
        ll = T_combined.shape[0]
        scl, a = 0.8, 1.5

        ax.scatter(rs[0], rs[1], rs[2], marker="4")  # satellite CG
        ax.text(rs[0], rs[1], rs[2], "spacecraft_CG")  # satellite CG
        xz, yz, zz = T_combined[0, 0, 2], T_combined[0, 1, 2], T_combined[0, 2, 2]
        ax.add_artist(Arrow3D([rs[0], rs[0] + scl * xz], [rs[1], rs[1] + scl * yz],
                              [rs[2], rs[2] + + scl * zz], mutation_scale=20,
                              lw=3, arrowstyle="-|>", color="b"))  # spacecraft z (joint) axis
        # plt.pause(0.05)
        xx, yy, zz = T_combined[1, 0, 3], T_combined[1, 1, 3], T_combined[1, 2, 3]  # robot base (x, y, z)
        ax.scatter(xx, yy, zz, lw=5)
        if ax is not None:
            xxx, yyy, zzz = self.satPlot.cube_data2((0, 0, 0), size)
            m = np.vstack((xxx, yyy, zzz))
            mr = Rot @ m + np.array([[rs[0]], [rs[1]], [rs[2]]])
            x, y, z = mr[0, :], mr[1, :], mr[2, :]
            ax.plot(x, y, z, 'g', lw=5)
            ax.scatter(0, 0, 0, marker="D")  # CG of the whole spacecraaft-robot arm system is at (0, 0, 0)
            ax.text(0, 0, 0, "CG_system")  # CG of the whole spacecraaft-robot arm system is at (0, 0, 0)
            for i in range(2, T_combined.shape[0]):
                jx, jy, jz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
                ax.plot([xx, jx], [yy, jy], [zz, jz], lw=10)  # line segment between joints or links
                ax.add_artist(Arrow3D([jx, jx + scl*T_combined[i, 0, 2]], [jy, jy+ scl*T_combined[i, 1, 2]], [jz, jz+ + scl*T_combined[i, 2, 2]], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color="b"))  # z (joint) axis
                ax.add_artist(Arrow3D([jx, jx + scl*T_combined[i, 0, 0]], [jy, jy+ scl*T_combined[i, 1, 0]], [jz, jz+ + scl*T_combined[i, 2, 0]], mutation_scale=20,
                        lw=3, arrowstyle="-|>", color="r"))  # x  axis
                if i < ll-1:
                    ax.scatter(T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3], 'gray', lw=10)
                xx, yy, zz = T_combined[i, 0, 3], T_combined[i, 1, 3], T_combined[i, 2, 3]
                # plt.pause(0.05)
            plt.xlabel('X')
            plt.ylabel('Y')
            # ax.axis('equal')
            ax.view_init(elev=54., azim=-140.)
            ax.set_zlim(-a, a)
            # ax.set_ylim(-a, a)
            # ax.set_xlim(-a, a)

    def call_plot(self, rs, size, color, rot_ang, q, pv_com=None, ax=None, b0=None):
        # rot_ang is a 3 x t vector of the rotation angles of the spacecraft. q is manipulator angles
        if not ax:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # ax.set_aspect('equal')
        if rot_ang.shape[1] > 3:
            n = rot_ang.shape[1]
            for i in range(n):
                temp = [(rs[:, i][0], rs[:, i][1], rs[:, i][2])]
                qi = q[:, i]
                plt.cla()
                if isinstance(pv_com, (list, tuple, np.ndarray)):
                    ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
                for p, s, c in zip(temp, size, color):
                    self.satellite_namipulator(rot_ang=rot_ang[:, i], q=qi, rs=p, size=s, ax=ax, b0=b0)
                plt.pause(0.05)
        else:
            temp = [(rs[0], rs[1], rs[2])]
            plt.cla()
            # if isinstance(pv_com, (list, tuple, np.ndarray)):
            #     ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
            for p, s, c in zip(temp, size, color):
                self.satellite_namipulator(rot_ang=rot_ang, q=q, rs=p, size=s, ax=ax, b0=b0)
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
        # MR = MayaviRendering(nDoF=7, robot='7DoF')
        # MR.anim(rs=rs1, angs=ang_s1, q=q1, b0=b0, fig_save=False, reverse=False)
        # mlab.show()
        pass