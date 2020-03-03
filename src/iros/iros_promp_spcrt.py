import numpy as np
import os
import pickle
import phase as phase
import basis as basis
import matplotlib.pyplot as plt
import promps as promps
from iros_inv_7dof import InvKin
from iros_forward_kin import ForwardKin
from iros_7DoF_EOM import Dynamics
from mpl_toolkits.mplot3d import Axes3D
import time as tme
from scipy.spatial import distance
from numpy.lib.stride_tricks import as_strided

save_dir = os.path.dirname(os.path.abspath(__file__))
QList = np.load(save_dir+'/save_data_inv_kin/data/JointAnglesList.npy', allow_pickle=True)
QList = list(np.transpose(QList, (0, 2, 1)))  # Trajectories should of the shape nT x nDoF having length nDemo
timeList = [np.linspace(0, 1, QList[0].shape[0]) for i in range(len(QList))]


class TrajectoryLearning:
    def __init__(self, time, target, ang_s0, q0, nDoF=3, nBf=5, ):
        self.target_loc = target
        self.q0 = q0   # initial joint angles
        self.ang_s0 = ang_s0   # initial spacecraft angles wrt COM/inertial
        self.nDoF = nDoF
        self.spacecraft_inv_kin = InvKin(nDoF=self.nDoF, robot='7DoF')
        self.spacecraft_fwd_kin = ForwardKin(nDoF=self.nDoF, robot='7DoF')
        self.spacecraft_dyn = Dynamics(nDoF=self.nDoF, robot='7DoF')
        self.time_normalised = time
        self.phaseGenerator = phase.LinearPhaseGenerator()
        self.basisGenerator = basis.NormalizedRBFBasisGenerator(self.phaseGenerator, numBasis=nBf, duration=1,
                                                                basisBandWidthFactor=3, numBasisOutside=1)
        self.basisMultiDoF = self.basisGenerator.basisMultiDoF(self.time_normalised, self.nDoF)
        self.learnedProMP = promps.ProMP(self.basisGenerator, self.phaseGenerator, self.nDoF)
        self.learner = promps.MAPWeightLearner(self.learnedProMP)
        self.learnedData = self.learner.learnFromData(QList, timeList)
        mu_theta, sig_theta = self.learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))
        self.mu_theta, self.sig_theta = np.squeeze(mu_theta), np.squeeze(sig_theta)

        A, Q = self.spacecraft_inv_kin.call_optimize(target=self.target_loc, ang_s0=ang_s0, q0=q0)
        self.post_mean_q = Q[:, -1]

        # self.taskProMP0 = self.learnedProMP.jointSpaceConditioning(0, desiredTheta=self.post_mean_q,
        #                                                            desiredVar=np.eye(len(self.q0)) * 0.00001)
        self.taskProMP = self.learnedProMP.jointSpaceConditioning(1.0, desiredTheta=self.post_mean_q, desiredVar=np.eye(
            len(self.q0)) * 0.00001)  # desiredVar=self.post_cov_q)

        self.trajectories_learned = self.learnedProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
        self.trajectories_task_conditioned = self.taskProMP.getTrajectorySamples(self.time_normalised, n_samples=20)


if __name__ == '__main__':
    time = np.linspace(0, 1, 15)
    nDoF, nBf = 7, 5
    target = np.array([-2.8, 0.95, 0.25])
    ang_s0, q0 = np.array([0., 0., 0.]), np.array([0., 5 * np.pi / 4, 0., 0., 0., 0., 0.])
    traj_learn = TrajectoryLearning(time, target, ang_s0, q0, nDoF=nDoF, nBf=nBf)
    eef_curr_position = traj_learn.spacecraft_inv_kin.manip_eef_pos(ang_s0, q0)
    rs0 = traj_learn.spacecraft_dyn.spacecraft_com_pos(q=q0, ang_s=ang_s0, b0=np.array([1.05, 1.05, 0]))
    size = traj_learn.spacecraft_inv_kin.kin.size

    conditioned_trajs = traj_learn.trajectories_task_conditioned  # shape is nsamples x ntime x nDoF



    eef_pos = np.zeros((3, len(time)))
    eef_pos_list = []
    for j in range(conditioned_trajs.shape[2]):
        for i in range(conditioned_trajs.shape[0]):
            eef_pos[:, i] = traj_learn.spacecraft_inv_kin.manip_eef_pos(ang_s0, conditioned_trajs[i, :, j])
        eef_pos_list.append(eef_pos)

    traj_learn.spacecraft_fwd_kin.call_plot(rs0, size, 'red', ang_s0, q0)

    for i in range(len(eef_pos_list)):
        plt.plot(eef_pos_list[i][0], eef_pos_list[i][1], eef_pos_list[i][2])
    plt.show()
    print('hi')