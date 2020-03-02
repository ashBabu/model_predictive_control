import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from iros_7DoF_EOM import Dynamics, Kinematics
from iros_forward_kin import ForwardKin
from scipy.spatial.transform import Rotation as R
from utils import Utilities
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3)

save_dir = os.path.dirname(os.path.abspath(__file__))


class InvKin:

    def __init__(self, nDoF=3, robot='3DoF', b0=np.array([1.05, 1.05, 0]), q=None):
        self.nDoF = nDoF
        if not isinstance(q, (list, tuple, np.ndarray)):
            self.q = np.array([0., 5*np.pi/4, 0., 0., 0., np.pi/2, 0.])
        else:
            self.q = q
        if not isinstance(b0, (list, tuple, np.ndarray)):
            self.b0 = self.kin.b0
        else:
            self.b0 = b0
        self.kin = Kinematics(nDoF=self.nDoF, robot=robot)
        self.dyn = Dynamics(nDoF=self.nDoF, robot=robot)
        self.fwd_kin = ForwardKin(nDoF=self.nDoF, robot=robot)
        self.m, self.I, self.l = self.dyn.mass, self.dyn.I_num, self.kin.l_num[1:]  # cutting out satellite length l0
        self.ang_s0 = self.kin.ang_s0

        self.ang_b0 = self.kin.robot_base_ang(b0=self.b0)
        # self.pv_com_num, self.pv_eef_num, _ = self.dyn.com_pos_vect(b0=self.b0)
        # self.pv_com_num = self.dyn.substitute(pv_com, m=self.m, l=self.l, I=self.I)
        # self.pv_eef_num = self.dyn.substitute(pv_eef, m=self.m, l=self.l, I=self.I)
        # L = self.dyn.ang_momentum_conservation(b0=self.b0)
        # self.L_num = self.dyn.substitute(L, m=self.m, l=self.l, I=self.I)
        # j_omega, _, j_vel_eef = self.dyn.velocities_frm_momentum_conservation(b0=self.b0)
        # self.omega_eef = self.dyn.substitute(j_omega[:, -1], m=self.m, l=self.l, I=self.I)
        # self.vel_eef = self.dyn.substitute(j_vel_eef, m=self.m, l=self.l, I=self.I)
        # self.qd = self.kin.qd[3:]
        # self.qd_s, self.qd_m = self.qd[0:3], self.qd[3:]
        self.lmda1, self.lmda2 = 2, 0.5  # optimization weights

    def pos_vec(self, q, *args):
        ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz = args
        pv_com, pv_eef, _ = self.dyn.com_pos_vect(q, ang_xs, ang_ys, ang_zs, r_sx, r_sy, r_sz, self.b0)
        return pv_com, pv_eef

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def manip_eef_pos(self, ang_s, q):
        pv_com, pv_eef, pv_origin = self.dyn.pos_vect_inertial(ang_s=ang_s, q=q)
        pv_eef1 = pv_origin[:, -1]
        return pv_eef1

    def path(self, eef_des_pos, q0):  # q0 is current joint angles which is used to calculate current end_eff position
        init_pos = self.manip_eef_pos(self.ang_s0, q0)
        points = self.discretize(init_pos, eef_des_pos, step_size=0.15)  # step_size is inversely proportional to vel
        points = np.insert(points, 0, init_pos, axis=0)
        points = np.insert(points, len(points), eef_des_pos, axis=0)
        return points

    # Method 1: works
    # def jacobians(self, ang_s, q):
    #     omega_eef_num = self.dyn.substitute(self.omega_eef, ang_s0=ang_s, q0=q)
    #     Jw_s, Jw_m = omega_eef_num.jacobian(self.qd_s), omega_eef_num.jacobian(self.qd_m)  # Jv_s and Jv_m are both functions of ang_s, q_i
    #     Jw_s, Jw_m = np.array(Jw_s).astype(np.float64), np.array(Jw_m).astype(np.float64),
    #
    #     vel_eef_num = self.dyn.substitute(self.vel_eef, ang_s0=ang_s, q0=q)
    #     Jv_s, Jv_m = vel_eef_num.jacobian(self.qd_s), vel_eef_num.jacobian(self.qd_m)  # Jv_s and Jv_m are both functions of ang_s, q_i
    #     Jv_s, Jv_m = np.array(Jv_s).astype(np.float64), np.array(Jv_m).astype(np.float64),
    #     return Jv_s, Jv_m, Jw_s, Jw_m
    #
    # def generalized_jacobian(self, ang_s, q):
    #     Jv_s, Jv_m, Jw_s, Jw_m = self.jacobians(ang_s, q)
    #     L_num = self.dyn.substitute(self.L_num, ang_s0=ang_s, q0=q)
    #     Ls, Lm = L_num.jacobian(self.qd_s), L_num.jacobian(self.qd_m)
    #     Ls, Lm = np.array(Ls).astype(np.float64), np.array(Lm).astype(np.float64)
    #     a1 = np.linalg.solve(Ls, Lm)
    #     Jv = Jv_m - Jv_s @ a1
    #     Jw = Jw_m - Jw_s @ a1
    #     return np.vstack((Jv, Jw)), Ls, Lm

    # Method 1: Directly calculating as given in Umetani and Yoshida equation 22
    # Cant handle joint limits

    def dir(self, X, point):
        ang_s, q = X[0:3], X[3:]
        r_eef_current = self.manip_eef_pos(ang_s, q)
        dx = point - r_eef_current
        dx = np.hstack((dx, 0., 0., 0.))
        J = self.dyn.generalized_jacobian(ang_s=ang_s, q=q)
        Ls, Lm = self.dyn.momentOfInertia_transform(ang_s=ang_s, q=q)
        dq = np.linalg.pinv(J) @ dx
        dphi = -np.linalg.solve(Ls, Lm) @ dq
        return dq, dphi

    def call_dir(self, target, q0, ang_s0):
        # q0 = np.squeeze(np.array(q0).astype(np.float64))
        # ang_s0 = np.squeeze(np.array(ang_s0).astype(np.float64))
        points = self.path(target, q0)
        sh = points.shape[0]
        X0 = np.hstack((ang_s0, q0))
        q, ang_s = np.zeros((self.nDoF, sh+1)), np.zeros((3, sh+1))
        q[:, 0], ang_s[:, 0] = q0, ang_s0
        for i in range(1, sh+1):
            dq, dphi = self.dir(X0, points[i-1, :])
            q[:, i] = q[:, i-1] + dq
            ang_s[:, i] = ang_s[:, i-1] + dphi
            X0 = np.hstack((ang_s[:, i], q[:, i]))
        return ang_s, q

    # Method 2: Using optimization and hence can handle bounds

    def cost(self, dq, r_eef_current, eef_des_pos, J):
        dx = eef_des_pos - r_eef_current
        dx = np.hstack((dx, 0., 0., 0.))
        t1 = dx - J @ dq
        g = np.array([0.5, 0.1, 0.5])
        W = np.eye(dq.shape[0]) #@ np.diag(g)
        # W = self.lmda1 * np.eye(dq.shape[0])
        cost = 0.5 * (dq.T @ W @ dq + self.lmda2 * t1.T @ t1)
        return cost

    def jac_cost(self, dq, r_eef_current, eef_des_pos, J):
        # r_eef_current = self.manip_eef_pos_num(ang_s, q)
        dx = eef_des_pos - r_eef_current
        dx = np.hstack((dx, 0., 0., 0.))
        t1 = dx - J @ dq
        g = np.array([0.5, 2, 0.8])
        W = np.eye(dq.shape[0]) #@ np.diag(g)
        # W = self.lmda1 * np.eye(dq.shape[0])
        jac = W @ dq - self.lmda2 * J.T @ t1
        return jac

    def inv_kin(self, dq0, r_eef_current, eef_des_pos, J):
        # results = opt.minimize(self.inv_kin_optim_func, X0, args=eef_des_pos, method='BFGS',
        #                        options={'maxiter': 150, 'disp': True})
        bnds = ((-np.pi, np.pi), (-np.pi/6, np.pi), (-np.pi/4, np.pi/2))
        bnds1 = ((None, None), (None, None), (None, None), (None, None), (None, None), (None, None))
        results = opt.minimize(self.cost, dq0, args=(r_eef_current, eef_des_pos, J), method='SLSQP', jac=self.jac_cost,
                               options={'maxiter': 150, 'disp': True},)# bounds=bnds)
        # results = opt.fmin_slsqp(func=self.inv_kin_optim_func,
        #                           x0=X0, eqcons=[self.constraints[0],self.constraints[1], self.constraints[2]],
        #                           args=eef_des_pos, iprint=0)
        return results.x

    def call_optimize(self, ang_s0=np.zeros(3), q0=None, target=np.array([-3, 2.5, 0.0]), ref_path=None):
        if isinstance(ref_path, (list, tuple, np.ndarray)):
            points = ref_path
        else:
            points = self.path(target, q0)  # q = initial joint values to compute the position of end_eff
        dq0 = np.random.randn(self.nDoF) * 0.001
        pr, pc = points.shape
        q, ang_s = np.zeros((self.nDoF, pc + 1)), np.zeros((3, pc + 1))
        q[:, 0], ang_s[:, 0] = q0, ang_s0
        J = self.dyn.generalized_jacobian(ang_s=ang_s0, q=q0)
        Is, Im = self.dyn.momentOfInertia_transform(ang_s=ang_s0, q=q0)
        r_eef_current = self.manip_eef_pos(ang_s0, q0)
        for i in range(1, pc+1):
            dq = self.inv_kin(dq0, r_eef_current, points[:, i-1], J)
            q[:, i] = q[:, i - 1] + dq
            ang_s[:, i] = ang_s[:, i - 1] - np.linalg.solve(Is, Im) @ dq
            ang_s0, q0, dq0 = ang_s[:, i], q[:, i], dq
            J = self.dyn.generalized_jacobian(ang_s=ang_s0, q=q0)
            Is, Im = self.dyn.momentOfInertia_transform(ang_s=ang_s0, q=q0)
            r_eef_current = self.manip_eef_pos(ang_s0, q0)
        return ang_s, q

    def animation(self, r_s, size, rot_ang, q, path, color='green', pv_com=None, ax=None,):
        a = 3.1
        # rot_ang is a 3 x t vector of the rotation angles of the spacecraft. q is manipulator angles
        if not ax:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # ax = fig.gca(projection='3d')
            ax.set_aspect('equal')

            fig.set_facecolor('black')
            ax.set_facecolor('black')
            ax.grid(False)
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False

        for i in range(rot_ang.shape[1]):
            plt.cla()
            temp = [(r_s[:, i][0], r_s[:, i][1], r_s[:, i][2])]
            qi = q[:, i]
            # ax = plt.axes(projection='3d')
            # ax.set_aspect('equal')
            # fig.set_facecolor('black')
            # ax.set_facecolor('black')
            # ax.grid(False)
            # ax.w_xaxis.pane.fill = False
            # ax.w_yaxis.pane.fill = False
            # ax.w_zaxis.pane.fill = False
            # X1, Y1, arr = self.image_draw('space.png')
            # ax.plot_surface(X1, Y1, np.ones(X1.shape), rstride=1, cstride=1, facecolors=arr)
            # plt.axis('off')
            if isinstance(pv_com, (list, tuple, np.ndarray)):
                ax.scatter(pv_com[i, 0, :], pv_com[i, 1, :], pv_com[i, 2, :], 'r^', lw=8)  # plot of COMs
            for p, s, c in zip(temp, size, color):
                self.fwd_kin.satellite_namipulator(rot_ang=rot_ang[:, i], q=qi, rs=p, size=s, ax=ax, b0=self.b0)
                ax.scatter(path[0, :], path[1, :], path[2, :], 'r-', lw=4)
                # ax.view_init(elev=85., azim=-58)
            ax.set_zlim(-a, a)
            ax.set_ylim(-a, a)
            ax.set_xlim(-a, a)
            plt.pause(0.05)
            # plt.savefig("/home/ar0058/Ash/repo/model_predictive_control/src/animation/inv_kinematics_direct/%02d.png" % i)
            # print('hi')

    def get_circle(self, scale=0.1, start=None, goal=None):
        angles = np.linspace(0, 2 * np.pi, 30)
        circ = np.zeros((3, angles.shape[0]))
        center = start + 0.5 * (goal - start)
        dir_vec = goal - start
        """
            A vector perpendicular to dir_vec has np.dot(dir_vec, perp_vec) = 0
            Let perp_vec = (a, b, c) implies a x + b y + c z = 0. Put arbitrary values for a, b 
            which means c = -(1/z) * (a x + b y) 
            """
        a, b = 1, 1
        c = -(1 / dir_vec[2]) * (dir_vec[0] + dir_vec[1])
        perp_vec = np.array([a, b, c])

        for i, ang in enumerate(angles):
            rot = R.from_rotvec(ang * dir_vec)
            circ[:, i] = scale * (rot.as_dcm() @ perp_vec) + center
        return circ

    def get_plts(self, A, Q):
        plt.figure()
        plt.plot(A[0, :], label='satellite_x_rotation')
        plt.plot(A[1, :], label='satellite_y_rotation')
        plt.plot(A[2, :], label='satellite_z_rotation')
        plt.legend()

        plt.figure()
        for i in range(Q.shape[0]):
            plt.plot(Q[i, :], label='q%d' % i)
        plt.legend()


if __name__ == '__main__':
    nDoF, b0 = 7, np.array([1.05, 1.05, 0])
    robot = '7DoF'
    IK = InvKin(nDoF=nDoF, robot=robot, b0=b0)
    util = Utilities()
    target_loc = np.array([[-3, 2.5, 0.0], [-3, 1.2, 0.0], [-2, 1.5, 1]])  # end-effector target location
    q0 = np.array([0., 5 * np.pi / 4, 0., 0., 0., 0., 0.])  # initial joint angles
    ang_s0 = IK.kin.ang_s0  # initial spacecraft angles
    r_s0 = IK.dyn.spacecraft_com_pos(ang_s=ang_s0, q=q0, b0=b0)  # initial position vector of spacecraft CG wrt inertial
    # points = IK.path(target_loc, q0)  # straight line path (x, y, z) from current to target location
    start = IK.manip_eef_pos(ang_s0, q0)  # manipulator end-effector position for the ang_s0 and q0

    def call_plots(spacecraftAngles=None, jointAngles=None, spacecraftPosVec=None, ref_path=None):
        # IK.get_plts(spacecraftAngles, jointAngles)  # to get plots
        plt.figure()
        ax = plt.axes(projection='3d')
        IK.animation(spacecraftPosVec, IK.kin.size, spacecraftAngles, jointAngles, ref_path, ax=ax)  # animation
    """
    To generate reference trajectories from the current to target position, quadratic bezier curves are used. They
    require start, goal and another point in between them. This is calculated as the points on the circumference of 
    circle whose center is at the midpoint between start and goal 
    """
    plott = True
    for target in target_loc:
        circ1 = IK.get_circle(start=start, goal=target)
        spacecraft_angles, joint_angles = [], []
        endEff_posVec, spacecraft_postionVec = [], []
        for jk, point in enumerate(circ1.T):
            ref_path = util.quadratic_bezier(start, point, target)
            A, Q = IK.call_optimize(ang_s0=ang_s0, q0=q0, ref_path=ref_path)
            Q = np.c_[q0, Q]
            A = np.c_[ang_s0, A]
            spacecraft_angles.append(A)
            joint_angles.append(Q)
            r_s = np.zeros((3, A.shape[1]))
            end_eff_pos = np.zeros((3, Q.shape[1]))
            for i in range(A.shape[1]):
                r_s[:, i] = IK.dyn.spacecraft_com_pos(ang_s=A[:, i], q=Q[:, i], b0=b0)
                end_eff_pos[:, i] = IK.manip_eef_pos(A[:, i], Q[:, i])
            endEff_posVec.append(end_eff_pos)
            spacecraft_postionVec.append(r_s)
            if plott and jk % 26 == 0:
                call_plots(spacecraftAngles=A, jointAngles=Q, spacecraftPosVec=r_s, ref_path=ref_path)

    """
    A, Q, points, r_s = np.load(save_dir+'/save_data_inv_kin/data/spacecraft_angs_inv_kin1.npy', allow_pickle=True),\
                        np.load(save_dir+'/save_data_inv_kin/data/joint_angs_inv_kin1.npy', allow_pickle=True), \
                        np.load(save_dir+'/save_data_inv_kin/data/ref_path_xyz1.npy', allow_pickle=True),\
                        np.load(save_dir+'/save_data_inv_kin/data/spacecraft_com_inv_kin1.npy', allow_pickle=True)
    """
    """
    np.save(save_dir+'/save_data_inv_kin/data/joint_angs_inv_kin1.npy', q, allow_pickle=True),
    np.save(save_dir+'/save_data_inv_kin/data/spacecraft_com_inv_kin1.npy', r_s, allow_pickle=True),
    np.save(save_dir+'/save_data_inv_kin/data/spacecraft_angs_inv_kin1.npy', ang_s, allow_pickle=True),
    np.save(save_dir+'/save_data_inv_kin/data/target_loc_inv_kin1.npy', target_loc, allow_pickle=True)
    np.save(save_dir+'/save_data_inv_kin/data/ref_path_xyz1.npy', points, allow_pickle=True)
    np.save(save_dir+'/save_data_inv_kin/data/end_eff_xyz1.npy', end_eff_pos, allow_pickle=True)
    """
    print('hi')
    plt.show()
