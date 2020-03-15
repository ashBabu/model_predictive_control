import numpy as np
import os
import phase as phase
import basis as basis
import promps_for_control as prompsControl
from iros_inv_7dof import InvKin
from iros_forward_kin import ForwardKin
from iros_7DoF_EOM import Dynamics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


save_dir = os.path.dirname(os.path.abspath(__file__))
QList = np.load(save_dir+'/save_data_inv_kin/data/JointAnglesList.npy', allow_pickle=True)
QList = list(np.transpose(QList, (0, 2, 1)))  # Trajectories should of the shape nT x nDoF having length nDemo
nT, nDoF = QList[0].shape[0], QList[0].shape[1]
VList = [np.diff(Q, axis=0) for Q in QList]  # velocity (nT - 1) x nDoF
VList = [np.vstack((np.zeros(nDoF), V)) for V in VList]
StateVector = [np.vstack((q, v)) for q, v in zip(QList, VList)]
timeList = [np.linspace(0, 1, nT) for i in range(len(QList))]


class TrajectoryLearning:
    def __init__(self, time, target, ang_s0, q0, nDoF=3, nBf=5):
        self.target_loc = target
        self.nBf = nBf
        self.q0 = q0   # initial joint angles
        self.ang_s0 = ang_s0   # initial spacecraft angles wrt COM/inertial
        self.nDoF = nDoF
        self.spacecraft_inv_kin = InvKin(nDoF=self.nDoF, robot='7DoF')
        self.spacecraft_fwd_kin = ForwardKin(nDoF=self.nDoF, robot='7DoF')
        self.spacecraft_dyn = Dynamics(nDoF=self.nDoF, robot='7DoF')
        self.time_normalised = time
        self.phaseGenerator = phase.LinearPhaseGenerator()
        self.basisGenerator = basis.NormalizedRBFBasisGeneratorAsh(self.phaseGenerator, numBasis=self.nBf, duration=1,
                                                                basisBandWidthFactor=3, numBasisOutside=1)
        self.basisMultiDoF = self.basisGenerator.totalBasis(self.time_normalised, self.nDoF)
        self.learnedProMP = prompsControl.ProMP(self.basisGenerator, self.phaseGenerator, self.nDoF)
        self.learner = prompsControl.MAPWeightLearner(self.learnedProMP)
        self.learnedData = self.learner.learnFromData(StateVector, timeList)
        # mu_theta, sig_theta = self.learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))
        mu_w, sig_w = self.learnedProMP.mu, self.learnedProMP.covMat
        mu_theta, sig_theta = self.learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))
        self.mu_w, self.sig_w = np.squeeze(mu_theta), np.squeeze(sig_theta)

        A, Q = self.spacecraft_inv_kin.call_optimize(target=self.target_loc, ang_s0=ang_s0, q0=q0)
        self.post_mean_q = np.hstack((Q[:, -1], np.zeros(self.nDoF)))
        q0_new = np.hstack((self.q0, np.zeros(self.nDoF)))
        self.taskProMP0 = self.learnedProMP.jointSpaceConditioning(0, desiredTheta=q0_new,
                                                                   desiredVar=np.eye(len(q0_new)) * 0.0000001)
        self.taskProMP = self.learnedProMP.jointSpaceConditioning(1.0, desiredTheta=self.post_mean_q, desiredVar=np.eye(
            len(self.post_mean_q)) * 0.00001)  # desiredVar=self.post_cov_q)

        self.trajectories_learned = self.learnedProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
        self.trajectories_task_conditioned = self.taskProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
        print('hi')


class PrompController(TrajectoryLearning):
    def __init__(self, time, target, ang_s0, q0, nDoF=3, nBf=5, robot='7DoF'):
        super().__init__(time, target, ang_s0, q0, nDoF=nDoF, nBf=nBf)
        self.robot = robot
        self.nState = 2 * self.nDoF
        self.dt = self.time_normalised[1] - self.time_normalised[0]
        self.basis_dot = self.basisGenerator.basisDerivative(self.time_normalised)
        if self.robot == 'Double Integreator':
            ##### for a double integrator  #####
            self.A = np.array([[0, 1], [0, 0]])  # system dynamics matrix
            self.B = np.array([[0], [1]])  # input matrix
            self.c = 0

            # self.mu_w = np.random.rand((self.nBf * self.nDoF))  # assuming a value for the mean of the learned distribution
            # self.sig_w = np.random.rand(self.nBf * self.nDoF, self.nBf * self.nDoF)  # assuming a covariance matrix
            ############################################

        elif self.robot =='7DoF':
            ########### For 7-DoF manipulator ############
            # states are [q1, q2, q3..... q7, q1_dot, q2_dot, ......q7_dot]
            n_states, n_inputs = 2*self.nDoF, self.nDoF
            identity = np.eye(int(n_states / 2))
            zeros = np.zeros((int(n_states / 2), int(n_states / 2)))
            # Continuous time dynamics:
            self.A = np.vstack((np.hstack((zeros, identity)), np.hstack((zeros, zeros))))
            self.B = np.vstack((zeros, identity))
            self.c = 0
            # self.mu_w = np.random.rand(
            #     (self.nBf * self.nDoF * 2))  # assuming a value for the mean of the learned distribution
            # self.sig_w = np.random.rand(self.nBf * 2 * self.nDoF, self.nBf * 2 * self.nDoF)  # assuming a covariance mat
            ####################################

    def calculate_basis_t(self, t_index):  # psi_t as in Using probabilistic MPs in robotics
        basis_t = np.zeros((self.nState, self.nState * self.nBf))
        basis_t_index = self.basisMultiDoF[t_index, 0:self.nBf]
        i, j, r = self.nBf, 0, 0
        while r < self.nState:
            basis_t[r, j:i] = basis_t_index
            r += 1
            j = i
            i += self.nBf
        return basis_t

    def calculate_basis_dot_t(self, t_index):  # psi_dot_t
        basis_dot_t = np.zeros((self.nState, self.nState * self.nBf))
        basis_dot_t_index = self.basis_dot[t_index, 0:self.nBf]
        i, j, r = self.nBf, 0, 0
        while r < self.nState:
            basis_dot_t[r, j:i] = basis_dot_t_index
            r += 1
            j = i
            i += self.nBf
        return basis_dot_t

    def calculate_sig_t(self, t_index):
        basis_t = self.calculate_basis_t(t_index)
        temp1 = np.dot(basis_t, self.sig_w)
        return np.dot(temp1, basis_t.transpose())

    def calculate_mu_t(self, t_index):
        basis_t = self.calculate_basis_t(t_index)
        return np.dot(basis_t, self.mu_w)

    def calculate_cross_correlation(self, t_index):  # C_t
        basis_t = self.calculate_basis_t(t_index)
        basis_t_dt = self.calculate_basis_t(t_index + 1)
        temp1 = np.dot(basis_t, self.sig_w)
        return np.dot(temp1, basis_t_dt.transpose())

    def calculate_sig_dot_t(self, t_index):
        basis_t = self.calculate_basis_t(t_index)
        basis_dot_t = self.calculate_basis_dot_t(t_index)
        temp1 = np.dot(basis_dot_t, self.sig_w)
        temp2 = np.dot(basis_t, self.sig_w)
        return np.dot(temp1, basis_t.transpose()) + np.dot(temp2, basis_dot_t.transpose())

    def calculate_sig_t_dt(self, t_index):
        sig_dot_t = self.calculate_sig_dot_t(t_index)
        return self.dt * sig_dot_t + self.calculate_sig_t(t_index)  # sig_t_dt = dt*(sig_dot_t) + sig_t

    def calculate_sig_s(self, t_index):
        sig_t_dt = self.calculate_sig_t_dt(t_index)
        sig_t = self.calculate_sig_t(t_index)
        if np.multiply(*sig_t.shape) != 1:
            sig_t_inv = np.linalg.inv(sig_t)
        else:
            sig_t_inv = 1. / sig_t
        C_t = self.calculate_cross_correlation(t_index)
        return (1. / self.dt) * (sig_t_dt - np.dot(C_t.transpose(), np.dot(sig_t_inv, C_t)))

    def calculate_sig_u(self, t_index):
        sig_s = self.calculate_sig_s(t_index)
        B_inv = np.linalg.pinv(self.B)
        temp1 = np.dot(B_inv, np.squeeze(sig_s))
        return np.dot(temp1, B_inv.transpose())

    def calculate_feedback_gain(self, t_index):  # K
        basis_dot_t = self.calculate_basis_dot_t(t_index)
        basis_t = self.calculate_basis_t(t_index)
        sig_t = self.calculate_sig_t(t_index)
        if len(sig_t) == 1:
            sig_t = sig_t[0][0]
            sig_t_inv = 1. / sig_t
        else:
            sig_t_inv = np.linalg.inv(sig_t)
        sig_s = self.calculate_sig_s(t_index)
        B_inv = np.linalg.pinv(self.B)
        temp1 = np.dot(np.dot(basis_dot_t, self.sig_w), basis_t.transpose())
        temp2 = np.dot(self.A, sig_t)
        temp3 = np.dot(B_inv, (temp1 - temp2 - 0.5 * np.squeeze(sig_s)))
        return np.dot(temp3, sig_t_inv)

    def calculate_feedforward_gain(self, t_index):  # k
        basis_dot_t = self.calculate_basis_dot_t(t_index)
        basis_t = self.calculate_basis_t(t_index)
        B_inv = np.linalg.pinv(self.B)
        K = self.calculate_feedback_gain(t_index)
        temp1 = np.dot(basis_dot_t, self.mu_w)
        temp2 = self.A + np.dot(self.B, K)
        temp3 = np.dot(basis_t, self.mu_w)
        if len(temp3) == 1:
            temp3 = temp3[0]
        temp4 = np.dot(temp2, temp3)
        return np.dot(B_inv, (temp1 - temp4 - self.c))

    def calculate_control(self, time, n_samples=20):
        conditioned_trajs = self.taskProMP.getTrajectorySamples(time, n_samples=n_samples)  # nT x 2*nDoF x n_samples
        plt.figure()
        plt.plot(time, conditioned_trajs[:, 3, :])
        plt.xlabel('time')
        plt.title('Joint-Space conditioning')
        plt.show()
        dt = time[1] - time[0]
        u = np.zeros((len(time), self.nDoF))
        for i in range(len(time)):
            K = self.calculate_feedback_gain(i)
            k = self.calculate_feedforward_gain(i)
            sig_u = self.calculate_sig_u(i)
            mu_noise = np.zeros(sig_u.shape[0])
            eps_u = np.random.multivariate_normal(mu_noise, sig_u / dt)
            y_t = conditioned_trajs[i, :, 0]
            u[i, :] = K @ y_t + k + eps_u
        return u

    def plott(self, time, n_samples=10):
        u = self.calculate_control(time, n_samples=n_samples)
        plt.plot(time, u)
        plt.show()
        print('hi')


if __name__ == '__main__':
    time = np.linspace(0, 1, 50)
    nDoF, nBf = 7, 10
    aa, robot = 4, '7DoF'
    target = np.array([-2.8, 0.95, 0.25])
    ang_s0, q0 = np.array([0., 0., 0.]), np.array([0., 5 * np.pi / 4, 0., 0., 0., 0., 0.])
    b0 = np.array([1.05, 1.05, 0])

    ProMP_controller = PrompController(time, target, ang_s0, q0, nDoF=nDoF, nBf=nBf, robot=robot)
    n_samples = 20

    ProMP_controller.plott(time, n_samples=n_samples)
    print('hi')