import numpy as np
import matplotlib.pyplot as plt
# import tf.transformations as tf_tran
from mpl_toolkits import mplot3d
import scipy.optimize as opt


class DH_plotter():

    def __init__(self, nDoF, robot='Franka'):
        self.nDoF = nDoF
        if robot == 'Franka':
            # franka robot DH values
            self.a = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
            self.d = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
            self.alpha = np.array([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
            self.eef_dist = 0.0
        else:
            self.alpha = np.array([-np.pi/2, np.pi/2, 0.])
            self.a = np.array([0, 0, 1.])
            self.d = np.array([0.5, 0.,  0.])
            self.eef_dist = 0.5

    def robot_DH_matrix(self, q):
        t, T_joint = np.eye(4), np.zeros((self.nDoF+1, 4, 4))
        i = 0
        for i in range(self.nDoF):
            T = np.array([[np.cos(q[i]), -np.sin(q[i]), 0, self.a[i]],
                          [np.sin(q[i]) * np.cos(self.alpha[i]), np.cos(q[i]) * np.cos(self.alpha[i]), -np.sin(self.alpha[i]), -np.sin(self.alpha[i]) * self.d[i]],
                          [np.sin(q[i]) * np.sin(self.alpha[i]), np.cos(q[i]) * np.sin(self.alpha[i]), np.cos(self.alpha[i]), np.cos(self.alpha[i]) * self.d[i]],
                          [0, 0, 0, 1]])
            t = t.dot(T)
            T_joint[i, :, :] = t
        tmp = np.eye(4)
        tmp[0, 3] = self.eef_dist
        T_ee = t @ tmp
        T_joint[i+1, :, :] = T_ee
        return T_joint

    def plotter(self, ax, T_joint, lgnd, color='r'):
        x, y, z = 0, 0, 0
        for i in range(T_joint.shape[0]):
            ax.plot([x, T_joint[i, 0, 3]], [y, T_joint[i, 1, 3]], [z, T_joint[i, 2, 3]], color, label=lgnd)
            ax.scatter(T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3], 'gray')
            x, y, z = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
        # ax.arrow(0.0, 0.0, 0.0, 0., 0.15, 0., head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial y axis
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.axis('equal')
        ax.set_zlim(0, 1)


if __name__ == '__main__':
    current_joint_values = np.zeros(8) #+ 0.01 * np.random.random(7)
    ang_deg = 60
    # desired_joint_values = [np.pi*ang_deg/180, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desired_joint_values = [np.pi * ang_deg / 180, np.pi / 3, 0.0, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, ]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # robot = 'Franka'
    # nDoF = 8
    # q = np.zeros(8)
    robot = '3DOF'
    nDoF = 3
    q = np.array([np.pi/4, np.pi/2, np.pi/2])
    franka_kin = DH_plotter(nDoF, robot)
    T_joint = franka_kin.robot_DH_matrix(q)
    franka_kin.plotter(ax, T_joint, 'desired', color='blue')

    plt.show()
