import numpy as np
from sympy import *
from sympy.physics.mechanics import *
from sympy.tensor.array import Array
import matplotlib.pyplot as plt
from ff_eom_symbolic import kinematics, dynamics


class testing():

    def __init__(self, nDoF=2):
        self.nDoF = nDoF
        self.kin = kinematics(self.nDoF)
        self.dyn = dynamics(self.nDoF)

    def draw_sat_rect(self, T, l, b):
        p1, p2, p3, p4 = np.array([[l], [b], [0]]) * 0.5, np.array([[-l], [b], [0]]) * 0.5, np.array(
            [[-l], [-b], [0]]) * 0.5, np.array([[l], [-b], [0]]) * 0.5
        centre = np.array([0, 0, 0])
        rot = np.array(T).astype(np.float64)  # T[0:3, 0:3]
        rect_points = centre.reshape(-1, 1) + rot @ np.hstack((p1, p2, p3, p4))
        return rect_points

    def rotations_from_inertial(self):
        j_T_full, _, _ = self.kin.position_vectors()
        rot_full = list()
        for i in range(len(j_T_full)):
            rot_full.append(j_T_full[i][0:3, 0:3])  # rotation matrix of spacecraft COM + each joint CS wrt inertial
            # including end-eff (which is same as link n). rot_full = [0_R_s, 0_R_rb, 0_R_j1, 0_R_j2, ...].
            # rb = robot base or joint 0 {j0}
        return rot_full

    def com_pos_vect(self):
        # rs, r1, r2 etc are pv from inertial to COM of spacecraft, link1, lin2, ...
        k11 = self.dyn.mass_frac()
        rot_full = self.rotations_from_inertial()
        rot_full.remove(rot_full[1])  # rot_full = [0_R_s, 0_R_j1, 0_R_j2, ...].
        a, b = self.kin.ab_vectors()
        r0 = zeros(3, 1)
        pv_com = zeros(3, self.nDoF+1)
        for i in range(self.nDoF):
            r0 += k11[i] * (rot_full[i] @ b[:, i] + rot_full[i+1] @ a[:, i])
            # r0 += k11[i] * (b[:, i] + a[:, i])
        pv_com[:, 0] = r0
        for i in range(1, self.nDoF+1):
            # temp = pv_com[:, i-1] + b[:, i-1] + a[:, i-1]
            pv_com[:, i] = pv_com[:, i-1] + rot_full[i-1] @ b[:, i-1] + rot_full[i] @ a[:, i-1]
            # temp.applyfunc(simplify)
        return pv_com


    def velocities(self):
        ws = Matrix([[self.kin.w_sxd], [self.kin.w_syd], [self.kin.w_szd]])
        j_omega = zeros(3, self.nDoF+1)  # matrix containing ang vel of spacecraft + each of the links wrt inertial
        j_omega[:, 0] = ws
        rot_full = self.rotations_from_inertial()
        for i in range(1, self.nDoF+1):
            temp = rot_full[i+1][:, 2] * self.kin.qdm[i-1]
            j_omega[:, i] = j_omega[:, i-1] + temp  # j_w = [0_w_s, 0_w_j1, 0_w_j2, ...]
            # omega_skew_sym = self.kin.skew_matrix()
        j_vel_com = zeros(3, self.nDoF+1)  # matrix of linear vel of spacecraft + each of the links wrt inertial
        k11 = self.dyn.mass_frac()
        r0_d = zeros(3, 1)
        a, b = self.kin.ab_vectors()

        for i in range(self.nDoF):
            r0_d += k11[i] * (rot_full[i] @ b[:, i] + rot_full[i+1] @ a[:, i])
            # j_vel_com[:, 0] =
        return j_omega

    def plotter(self, rect, rot_full):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        ax = fig.add_subplot(111, )
        ax.plot(rect[0, :], rect[1, :])  # draw rectangular satellite
        ax.arrow(0., 0., 0.5, 0., head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial x axis
        ax.arrow(0., 0., 0., 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')  # inertial y axis
        col = ['g', 'r', 'b', 'c', 'm', 'yellow']
        sc, kk = 0.25, 1
        for i in range(len(rot_full)):
            # trans_temp = pv_origins[:, i]
            # rot_temp = j_T_full[i, 0:3, 0:3]
            xx, xy = rot_full[i][0, 0], rot_full[i][1, 0]
            yx, yy = rot_full[i][0, 1], rot_full[i][1, 1]
            xx, xy = np.array(xx).astype(np.float64) , np.array(xy).astype(np.float64)
            yx, yy = np.array(yx).astype(np.float64) , np.array(yy).astype(np.float64)
            ax.arrow(0., 0., sc * xx, sc * xy, head_width=0.05, head_length=0.1, fc=col[i], ec=col[i])  # x axis
            ax.arrow(0., 0., sc *yx, sc *yy, head_width=0.05, head_length=0.1, fc=col[i], ec=col[i])  # x axis
        ax.axis('equal')
        ax.set_ylim(-4., 4.5)
        ax.set_xlim(-1., 1.)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


if __name__ == '__main__':
    test = testing()
    rot_full = test.rotations_from_inertial()
    Rs = list()
    j_omega = test.velocities()
    com_pos_vec = test.com_pos_vect()

    for i in range(len(rot_full)):
        Rs.append(msubs(rot_full[i], {test.kin.ang_xs: 0, test.kin.ang_ys: 0, test.kin.ang_zs: np.pi/4 *0,
                                      test.kin.ang_xb: 0, test.kin.ang_yb: 0, test.kin.ang_zb: np.pi/4 *0,
                                      test.kin.q[-2]: np.pi/6, test.kin.q[-1]:np.pi/6}))

    rect_point = test.draw_sat_rect(Rs[0], 0.5, 0.25)
    aa = rect_point[:, 0]
    rect_point = np.hstack((rect_point, aa.reshape(-1, 1)))
    test.plotter(rect_point, Rs)

    print('hi')