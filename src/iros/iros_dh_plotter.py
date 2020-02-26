import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
np.set_printoptions(precision=2)
from arrow3d import Arrow3D


class DH_plotter():

    def __init__(self, nDoF=3, robot='3DoF'):
        self.nDoF = nDoF
        if robot == 'Franka':
            # franka robot DH values
            self.a = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
            self.d = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
            self.alpha = np.array([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
            self.eef_dist = 0.0
        elif robot == '3DoF':  # 3DOF robot given in yoshida and umeneti: resolved motion rate control of space manipulators
            self.eef_dist = 2.50
            self.alpha = np.array([-np.pi / 2, np.pi / 2, 0., 0.])
            self.a = np.array([0., 0., 2.5, 0.])
            self.d = np.array([0.25, 0., 0., self.eef_dist])
        elif robot == '7DoF':
            self.eef_dist = 0.3
            self.a = np.array([0., 0., 1., 0.7, 0.5, 0.5, 0., 0.])
            self.d = np.array([0.5, 0.0, 0., 0., 0., 0., 1.5, self.eef_dist])
            self.alpha = np.array([-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.])

    def robot_DH_matrix(self, q):
        t, T_joint, Ti = np.eye(4), np.zeros((self.nDoF+1, 4, 4)), np.zeros((self.nDoF+1, 4, 4))
        q = np.array(simplify(q)).astype(np.float64)
        q = np.insert(q, len(q), 0.0, axis=0)  # for end-effector (just a translation for the fixed joint)
        for i in range(q.shape[0]):
            T = np.array([[np.cos(q[i]), -np.sin(q[i]), 0, self.a[i]],
                          [np.sin(q[i]) * np.cos(self.alpha[i]), np.cos(q[i]) * np.cos(self.alpha[i]), -np.sin(self.alpha[i]), -np.sin(self.alpha[i]) * self.d[i]],
                          [np.sin(q[i]) * np.sin(self.alpha[i]), np.cos(q[i]) * np.sin(self.alpha[i]), np.cos(self.alpha[i]), np.cos(self.alpha[i]) * self.d[i]],
                          [0, 0, 0, 1]], dtype='float')
            t = t @ T
            Ti[i, :, :] = T
            T_joint[i, :, :] = t
        # tmp = np.eye(4)
        # tmp[0, 3] = self.eef_dist
        # T_ee = t @ tmp
        # T_joint[i+1, :, :] = T_ee
        # Ti[i+1, :, :] = tmp
        return T_joint, Ti

    def plotter(self, ax, Tt, lgnd, color='r'):
        for j in range(len(Tt)):
            x, y, z = 0, 0, 0
            plt.ylabel('Y')
            plt.cla()
            T_joint = Tt[j]
            ll = T_joint.shape[0]
            ax.scatter(0, 0, 0, lw=10)
            ax.add_artist(Arrow3D([x, 0], [y, 0], [z, 0.5], mutation_scale=20,
                                  lw=3, arrowstyle="-|>", color="b"))  # z (joint) axis
            scl = 0.6
            for i in range(ll):
                jx, jy, jz = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
                ax.plot([x, jx], [y, jy], [z, jz], lw=10, label=lgnd)
                ax.add_artist(Arrow3D([jx, jx + scl * T_joint[i, 0, 2]], [jy, jy + scl * T_joint[i, 1, 2]],
                                      [jz, jz + + scl * T_joint[i, 2, 2]], mutation_scale=20,
                                      lw=3, arrowstyle="-|>", color="b"))  # z (joint) axis
                ax.set_zlim(0., 2.)
                ax.set_ylim(-2, 2.)
                ax.set_xlim(-2., 2.)
                plt.pause(0.01)
                if i < ll-1:
                    ax.scatter(T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3], 'gray', lw=10)
                x, y, z = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
                # ax.axis('equal')
                plt.xlabel('X')
                plt.ylabel('Y')


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    i = 3  # int(input('1==Franka ; 2 == 3DoF ; 3 == 7DoF', ))
    q3 = np.linspace(0, np.pi/2, 50)
    q1 = np.linspace(0, np.pi, 50)
    Tt = list()
    if i == 1:
        robot = 'Franka'
        nDoF = 8
        q = np.zeros(8)
    elif i == 2:
        robot = '3DoF'
        nDoF = 3
        q = np.array([q1[i], np.pi / 2, np.pi / 2])
    elif i == 3:
        robot = '7DoF'
        nDoF = 7
        q = np.array([0., 5 * pi / 4, 0., 0., 0., 0., 0.])

    dh_plotter = DH_plotter(nDoF, robot)
    # for i in range(len(q)):
    T_joint, Ti = dh_plotter.robot_DH_matrix(q)
    Tt.append(T_joint)
    dh_plotter.plotter(ax, Tt, 'desired', color='blue')
    plt.show()
