import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


class SatellitePlotter(object):

    def __init__(self):
        pass

    def cuboid_data(self, o, size=(1, 1, 1)):  # o is the centre/position of the cuboid
        l, w, h = size
        x = [[o[0] - l/2, o[0] + l/2, o[0] + l/2, o[0] - l/2, o[0] - l/2],
             [o[0] - l/2, o[0] + l/2, o[0] + l/2, o[0] - l/2, o[0] - l/2],
             [o[0] - l/2, o[0] + l/2, o[0] + l/2, o[0] - l/2, o[0] - l/2],
             [o[0] - l/2, o[0] + l/2, o[0] + l/2, o[0] - l/2, o[0] - l/2]]
        y = [[o[1] - w/2, o[1] - w/2, o[1] + w/2, o[1] + w/2, o[1] - w/2],
             [o[1] - w/2, o[1] - w/2, o[1] + w/2, o[1] + w/2, o[1] - w/2],
             [o[1] - w/2, o[1] - w/2, o[1] - w/2, o[1] - w/2, o[1] - w/2],
             [o[1] + w/2, o[1] + w/2, o[1] + w/2, o[1] + w/2, o[1] + w/2]]
        z = [[o[2] - h/2, o[2] - h/2, o[2] - h/2, o[2] - h/2, o[2] - h/2],
             [o[2] + h/2, o[2] + h/2, o[2] + h/2, o[2] + h/2, o[2] + h/2],
             [o[2] - h/2, o[2] - h/2, o[2] + h/2, o[2] + h/2, o[2] - h/2],
             [o[2] - h/2, o[2] - h/2, o[2] + h/2, o[2] + h/2, o[2] - h/2]]
        return np.array(x), np.array(y), np.array(z)

    def plotCubeAt(self, rot_ang, pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
        # Plotting a cube element at position pos
        Rx = R.from_rotvec(rot_ang[0] * np.array([1, 0, 0]))
        Ry = R.from_rotvec(rot_ang[1] * np.array([0, 1, 0]))
        Rz = R.from_rotvec(rot_ang[2] * np.array([0, 0, 1]))
        Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
        Rot = Rx @ Ry @ Rz
        a = 5
        if ax is not None:
            X, Y, Z = self.cuboid_data(pos, size)
            sh = X.shape
            x, y, z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
            m = np.stack((x, y, z))
            mr = Rot @ m
            x, y, z = mr[0, :].reshape(sh), mr[1, :].reshape(sh), mr[2, :].reshape(sh)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, **kwargs)
            plt.xlabel('X')
            plt.ylabel('Y')
            ax.axis('equal')
            ax.set_zlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_xlim(-a, a)

    def call_plot(self, pos, size, color, rot_ang):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        for i in range(rot_ang.shape[1]):
            temp = [(pos[:, i][0], pos[:, i][1], pos[:, i][2])]
            for p, s, c in zip(temp, size, color):
                self.plotCubeAt(rot_ang[:, i], pos=p, size=s, ax=ax, color=c)
            plt.pause(0.2)
            plt.cla()


if __name__ == '__main__':
    rot_angle = np.linspace(0, 2*np.pi/4, 50)
    positions = [(0, 0, 0)]
    sizes = [(2, 2, 2), (3, 3, 7)]
    sat_plot = SatellitePlotter()
    colors = ["crimson", "limegreen"]
    sat_plot.call_plot(positions, sizes, colors, rot_angle)

