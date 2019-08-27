# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
#
# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,
#
# fig1 = plt.figure()
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                    interval=50, blit=True)
# plt.show()
#
# # To save the animation, use the command: line_ani.save('lines.mp4')
#
# from numpy import sin, cos
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import matplotlib.animation as animation
#
# G = 9.8  # acceleration due to gravity, in m/s^2
# L1 = 1.0  # length of pendulum 1 in m
# L2 = 1.0  # length of pendulum 2 in m
# M1 = 1.0  # mass of pendulum 1 in kg
# M2 = 1.0  # mass of pendulum 2 in kg
#
#
# def derivs(state, t):
#
#     dydx = np.zeros_like(state)
#     dydx[0] = state[1]
#
#     delta = state[2] - state[0]
#     den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
#     dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
#                 + M2 * G * sin(state[2]) * cos(delta)
#                 + M2 * L2 * state[3] * state[3] * sin(delta)
#                 - (M1+M2) * G * sin(state[0]))
#                / den1)
#
#     dydx[2] = state[3]
#
#     den2 = (L2/L1) * den1
#     dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
#                 + (M1+M2) * G * sin(state[0]) * cos(delta)
#                 - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
#                 - (M1+M2) * G * sin(state[2]))
#                / den2)
#
#     return dydx
#
# # create a time array from 0..100 sampled at 0.05 second steps
# dt = 0.05
# t = np.arange(0, 20, dt)
#
# # th1 and th2 are the initial angles (degrees)
# # w10 and w20 are the initial angular velocities (degrees per second)
# th1 = 120.0
# w1 = 0.0
# th2 = -10.0
# w2 = 0.0
#
# # initial state
# state = np.radians([th1, w1, th2, w2])
#
# # integrate your ODE using scipy.integrate.
# y = integrate.odeint(derivs, state, t)
#
# x1 = L1*sin(y[:, 0])
# y1 = -L1*cos(y[:, 0])
#
# x2 = L2*sin(y[:, 2]) + x1
# y2 = -L2*cos(y[:, 2]) + y1
#
# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
# ax.set_aspect('equal')
# ax.grid()
#
# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
#
#
# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text
#
#
# def animate(i):
#     thisx = [0, x1[i], x2[i]]
#     thisy = [0, y1[i], y2[i]]
#
#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*dt))
#     return line, time_text
#
#
# ani = animation.FuncAnimation(fig, animate,
#                               interval=dt*1000, blit=True, init_func=init)
# plt.show()


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

    def plotter(self, ax, Tt, lgnd, color='r'):
        for j in range(len(Tt)):
            x, y, z = 0, 0, 0
            T_joint = Tt[j]
            for i in range(T_joint.shape[0]):
                ax.plot([x, T_joint[i, 0, 3]], [y, T_joint[i, 1, 3]], [z, T_joint[i, 2, 3]], color, lw=10, label=lgnd)
                ax.scatter(T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3], 'gray', lw=10)
                x, y, z = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
                # ax.axis('equal')
                ax.set_zlim(0., 1.)
                ax.set_ylim(0., 2.)
                ax.set_xlim(-1., 1.)
                plt.xlabel('X')
                plt.ylabel('Y')
            plt.pause(0.01)
            plt.cla()


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # robot = 'Franka'
    # nDoF = 8
    # q = np.zeros(8)
    robot = '3DOF'
    nDoF = 3
    # the manipulator is a straight line when the joint angles are [0, 90, , 0]
    q = np.array([np.pi/4*0, np.pi/2, np.pi/2])
    dh_plotter = DH_plotter(nDoF, robot)
    q3 = np.linspace(0, np.pi/2, 50)
    q1 = np.linspace(0, np.pi, 50)
    Tt = list()
    for i in range(len(q3)):
        # q = np.array([0, np.pi/2, q3[i]])
        q = np.array([q1[i], np.pi/2, np.pi/2])
        # q = np.array([0, np.pi/2, q3[i]])
        T_joint = dh_plotter.robot_DH_matrix(q)
        Tt.append(T_joint)
    dh_plotter.plotter(ax, Tt, 'desired', color='blue')

    plt.show()
