# considering 2 link planar manipulator fixed on a free-floating spacecraft at the moment
import numpy as np
from scipy.spatial.transform import Rotation as rm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from graphics import *
# win = GraphWin()


class eom():
    def __init__(self):
        self.numJoints = 2
        self.m_0, self.m1, self.m2 = 5, 3, 2   # m_s = spacecraft mass, m_i = mass of i^th link
        self.m = [self.m_0, self.m1, self.m2]
        # DH parameters
        self.l1, self.l2 = 1, 0.5   # l_i = length of i^th link
        self.a = np.array([0, self.l1, self.l2])
        self.d = np.array([0., 0.])
        self.alpha = np.array([0., 0.])

        # Dynamic Parameters
        # Inertia tensor wrt centre of mass of each link
        I1_zz, I2_zz = 3, 4
        self.I1 = np.zeros((3, 3))
        self.I1[2, 2] = I1_zz
        self.I2 = np.zeros((3, 3))
        self.I2[2, 2] = I2_zz
        self.I = [self.I1, self.I2]

        self.b0 = np.array([0.2, 0.3, 0.])  # vector from the spacecraft COM to the base of the robot in spacecraft CS

    def robot_transformation_matrix(self, *args):  # general transformation matrix relating i to i+1^th CS using DH convention
        q, a, d, alpha = args
        c_q, s_q = np.cos(q), np.sin(q)
        c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
        T = np.array([[c_q, -s_q, 0, a],
                      [s_q * c_alpha, c_q * c_alpha, -s_alpha, -s_alpha * d],
                      [s_q * s_alpha, c_q * s_alpha, c_alpha, c_alpha * d],
                      [0, 0, 0, 1]])
        return T

    def euler_transformation_mat(self, *args):
        # spacecraft COM location with respect to inertial CS is defined as three consecutive  Euler rotations and a
        # translation
        # Also,  from spacecraft  COM location to manipulator base with respect to spacecraft CS is defined as three
        # consecutive  Euler rotations and a translation (a constant matrix)
        ang_x, ang_y, ang_z, r0 = args
        rx, ry, rz = rm.from_euler('x', ang_x), rm.from_euler('y', ang_y), rm.from_euler('z', ang_z)
        temp = rx * ry * rz
        r = temp.as_dcm()
        T = np.zeros((4, 4))
        T[0:3, 0:3] = r
        tmp = np.insert(r0, len(r0), 1)
        T[:, -1] = tmp
        return T

    def fwd_kin(self, q, l=None):  # ln is the length of the last link on which the EE is attached.
        # q = np.insert(q, len(q), 0.0, axis=0)  # added 0 at the end for the fixed joint
        # As per Craigs convention, T = Rotx * Trans x * Rotz * Trans z; Franka follows this
        if isinstance(l, (list, tuple, np.ndarray)):
            a = l
        else:
            a = self.a
        T = np.eye(4)
        T_joint = np.zeros([self.numJoints+1, 4, 4])
        for i in range(self.numJoints):
            ac = self.robot_transformation_matrix(q[i], a[i], self.d[i], self.alpha[i])
            T = np.dot(T, ac)
            T_joint[i, :, :] = T  # gives the transformation of each joint wrt base CS
        temp = np.eye(4)
        temp[0, 3] = self.a[-1]
        T_ee = np.dot(T, temp)
        T_joint[-1, :, :] = T_ee
        return T_ee, T_joint

    def position_vectors(self, *args): # position vectors of COM of each link wrt inertial CS, {j}
        # {s}, {ji} are respectively the CS of spacecraft at its COM and joint CS of the manipulator
        q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = args
        j_T_s = self.euler_transformation_mat(ang_xs, ang_ys, ang_zs, r0)
        s_T_j1 = self.euler_transformation_mat(ang_xb, ang_yb, ang_zb, b0)  # a constant 4 x 4 matrix
        T_ee, T_joint = self.fwd_kin(q, self.a)  #
        j_T_j1 = j_T_s @ s_T_j1
        j_T_full = np.zeros([self.numJoints+3, 4, 4])
        j_T_full[0, :, :] = j_T_s
        j_T_full[1, :, :] = j_T_j1
        k = 0
        for i in range(2, 3+self.numJoints):
            j_T_full[i, :, :] = j_T_j1 @ T_joint[k, :, :]
            k += 1
        return j_T_full

    def draw_satellite(self, T, r):
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros((1, len(x)))
        v = np.vstack((x, y, z))
        centre = T[0:3, 3]
        rot = T[0:3, 0:3]
        points = centre.reshape(-1, 1) + rot @ v
        return points

    def draw_sat_rect(self, T, l, b):
        p1, p2, p3, p4 = np.array([[l], [b], [0]]) * 0.5, np.array([[-l], [b], [0]]) * 0.5, np.array([[-l], [-b], [0]]) * 0.5, np.array([[l], [-b], [0]]) * 0.5
        centre = T[0:3, 3]
        rot = T[0:3, 0:3]
        points = centre.reshape(-1, 1) + rot @ np.hstack((p1, p2, p3, p4))
        return points

    def geometric_jacobian(self, q):
        T_ee, T_joint = self.fwd_kin(q)
        # Jac = np.zeros([3, len(q)])  # initilizing jacobian for 2D
        Jac = np.zeros([6, len(q)])  # initilizing jacobian for 3D
        for i in range(len(q)):
            pos_vec = T_ee[0:3, 3] - T_joint[i, 0:3, 3]
            rot_axis = T_joint[i, 0:3, 2]
            Jac[0:3, i] = np.cross(rot_axis, pos_vec)
            Jac[3:6, i] = rot_axis
        return Jac

    def mass_matrix(self, q, rs, rq):
        T_ee, T_joint = self.fwd_kin(q)
        Jac = self.geometric_jacobian(q)
        # I_infr = np.zeros([len(q), 3, 3])
        J_T = Jac[0:3, :]
        J_R = Jac[3:6, :]
        for i in range(len(q)):
            R = T_joint[i, 0:3, 0:3]
            temp = np.dot(R, self.I[i])
            I_infr = temp.dot(np.transpose(R))   # I in inertial frame. here with respect to base (for testing)

    def plotter(self, ax, T_joint, lgnd, color='r'):
        x, y, z = 0, 0, 0
        for i in range(len(self.a)):
            ax.plot([x, T_joint[i, 0, 3]], [y, T_joint[i, 1, 3]], [z, T_joint[i, 2, 3]], color, label=lgnd)
            ax.scatter(T_joint[i, 0, 3], T_joint[i,1, 3], T_joint[i, 2, 3], 'gray')
            x, y, z = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
            plt.xlabel('X')
            plt.ylabel('Y')
            scale = 0.4
            ax.set_xlim(-1*scale, 1*scale)
            ax.set_ylim(-1*scale, 1*scale)
            ax.set_zlim(0, 1)


if __name__ == '__main__':
    eom = eom()
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    ax = fig.add_subplot(111,)
    q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0 = [np.pi/6, np.pi/6], 0., 0., np.pi/6, 0., 0., np.pi/4, \
                                                                np.array([1., 1., 0.]), np.array([0.5, 0.25, 0.])
    j_T_full = eom.position_vectors(q, ang_xs, ang_ys, ang_zs, ang_xb, ang_yb, ang_zb, r0, b0)
    sat_centre = j_T_full[0, 0:3, 3]
    robot_base = j_T_full[1, 0:3, 3]
    # points = eom.draw_satellite(j_T_full[0], rad)
    points = eom.draw_sat_rect(j_T_full[0], 1., 0.5)
    aa = points[:, 0]
    points = np.hstack((points, aa.reshape(-1, 1)))
    ax.plot(points[0, :], points[1, :])  # draw rectangular satellite
    # ax.plot([sat_centre[0], points[0, 0]], [sat_centre[1], points[1, 0]],)
    ax.arrow(0, 0.0, 0.5, 0., head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial x axis
    ax.arrow(0, 0., 0, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial y axis
    origin_vect = j_T_full[:, 0:3, 3]
    tmp = np.diff(origin_vect, axis=0)
    sc = 0.25
    for i in range(j_T_full.shape[0]):
        trans_temp = j_T_full[i, 0:3, 3]
        rot_temp = j_T_full[i, 0:3, 0:3]
        ax.plot([0, trans_temp[0]], [0, trans_temp[1]])  # vector from {j} to each of the origins of {j_i} CS
        # ax.plot([trans_temp[0], trans_temp[0] + rot_temp[0, 0]], [trans_temp[1], trans_temp[1] + rot_temp[1, 0]])  # x axis of {j_i} th CS
        ax.arrow(trans_temp[0], trans_temp[1], sc * rot_temp[0, 0], sc * rot_temp[1, 0], head_width=0.05, head_length=0.1, fc='k', ec='k')  # x axis of {j_i} th CS
        # ax.plot([trans_temp[0], trans_temp[0] + rot_temp[0, 1]], [trans_temp[1], trans_temp[1] + rot_temp[1, 1]])  # y axis of {j_i} th CS
        ax.arrow(trans_temp[0], trans_temp[1], sc * rot_temp[0, 1], sc * rot_temp[1, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')  # y axis of {j_i} th CS

    for i in range(origin_vect.shape[0] - 1):
        ax.plot([origin_vect[i, 0], origin_vect[i+1, 0]], [origin_vect[i, 1], origin_vect[i+1, 1]])
        # plt.arrow(temp[0] + temp1[0, 0], temp[1] + temp1[1, 0], temp[0] + temp1[0, 1], temp[1] + temp1[1, 1] )
    # ax.plot([sat_centre[0], 0], [sat_centre[1], 0],)
    # ax.plot([robot_base[0], 0], [robot_base[1], 0],)
    # ax.plot(points[0, 0], points[1, 0], 'b*')
    # ax.plot(sat_centre[0], sat_centre[1], 'r*')
    ax.axis('equal')
    ax.set_ylim(0, 4)
    plt.show()
    print('hi')

