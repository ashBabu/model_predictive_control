import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from eom_symbolic import Kinematics
from mpl_toolkits import mplot3d
np.set_printoptions(precision=2)


class DH_plotter():

    def __init__(self, nDoF=3, robot='3DoF'):
        self.nDoF = nDoF
        self.kin = Kinematics()
        if robot == 'Franka':
            # franka robot DH values
            self.a = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
            self.d = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
            self.alpha = np.array([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
            self.eef_dist = 0.0
        elif robot == '3DoF':  # 3DOF robot given in yoshida and umeneti: resolved motion rate control of space manipulators
            self.alpha = np.array([-np.pi / 2, np.pi / 2, 0.])
            self.a = self.kin.a
            self.d = self.kin.d
            self.eef_dist = self.kin.eef_dist
        elif robot == '7DoF':
            # self.a = np.array([0., 0., 1.2, 0., 0., 1., 0.])
            # self.d = np.array([0.5, 0., 0., 1.5, 0., 0., 1.])
            # self.alpha = np.array([-np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])

            self.a = np.array([0., 0., 1., 0.7, 0.5, 0.5, 1.51])
            self.d = np.array([0.5, 0., 0., 0., 0., 0., 0.])
            self.alpha = np.array([-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2])
            self.eef_dist = 0.0

    def robot_DH_matrix(self, q):
        t, T_joint, Ti = np.eye(4), np.zeros((self.nDoF+1, 4, 4)), np.zeros((self.nDoF+1, 4, 4))
        i = 0
        q = np.array(simplify(q)).astype(np.float64)
        for i in range(self.nDoF):
            T = np.array([[np.cos(q[i]), -np.sin(q[i]), 0, self.a[i]],
                          [np.sin(q[i]) * np.cos(self.alpha[i]), np.cos(q[i]) * np.cos(self.alpha[i]), -np.sin(self.alpha[i]), -np.sin(self.alpha[i]) * self.d[i]],
                          [np.sin(q[i]) * np.sin(self.alpha[i]), np.cos(q[i]) * np.sin(self.alpha[i]), np.cos(self.alpha[i]), np.cos(self.alpha[i]) * self.d[i]],
                          [0, 0, 0, 1]], dtype='float')
            t = t @ T
            Ti[i, :, :] = T
            T_joint[i, :, :] = t
        tmp = np.eye(4)
        tmp[0, 3] = self.eef_dist
        T_ee = t @ tmp
        T_joint[i+1, :, :] = T_ee
        Ti[i+1, :, :] = tmp
        return T_joint, Ti

    def plotter(self, ax, Tt, lgnd, color='r'):
        for j in range(len(Tt)):
            x, y, z = 0, 0, 0
            plt.ylabel('Y')
            plt.cla()
            T_joint = Tt[j]
            for i in range(T_joint.shape[0]):
                ax.plot([x, T_joint[i, 0, 3]], [y, T_joint[i, 1, 3]], [z, T_joint[i, 2, 3]], color, lw=10, label=lgnd)
                ax.scatter(T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3], 'gray', lw=10)
                x, y, z = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
                # ax.axis('equal')
                # ax.set_zlim(0., 2.)
                # ax.set_ylim(0., 3.)
                # ax.set_xlim(-2., 2.)
                plt.xlabel('X')
            # plt.pause(0.01)


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    i = int(input('1==Franka ; 2 == 3DoF ; 3 == 7DoF', ))
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
        q = np.zeros(7)

    dh_plotter = DH_plotter(nDoF, robot)
    for i in range(len(q)):
        T_joint, Ti = dh_plotter.robot_DH_matrix(q)
        Tt.append(T_joint)
    dh_plotter.plotter(ax, Tt, 'desired', color='blue')
    plt.show()
