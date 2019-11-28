import numpy as np
import pickle
import phase as phase
import basis as basis
import promps as promps
from inv_kin import InverseKinematics
from sat_manip_simulation import Simulation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import scipy.ndimage
import time as tme
from scipy.spatial import distance
from numpy.lib.stride_tricks import as_strided

file_path = '/home/ar0058/Ash/repo/model_predictive_control/src/trajectory/'
with open('%strajectories.pickle' %file_path, 'rb') as f2:
    QList = pickle.loads(f2.read())
timeList = []
for i in range(len(QList)):
    timeList.append(np.linspace(0, 1, QList[i].shape[1]))


class TrajectoryLearning():
    def __init__(self, time, target, ang_s0, q0, nDoF=3, nBf=5, ):
        self.target_loc = target
        self.q0 = q0   # initial joint angles
        self.ang_s0 = ang_s0   # initial spacecraft angles wrt COM/inertial
        self.nDoF = nDoF
        self.spacecraft_inv_kin = InverseKinematics(self.nDoF)
        self.spacecraft_sim = Simulation(self.nDoF)
        self.time_normalised = time
        self.phaseGenerator = phase.LinearPhaseGenerator()
        self.basisGenerator = basis.NormalizedRBFBasisGenerator(self.phaseGenerator, numBasis=nBf, duration=1,
                                                                basisBandWidthFactor=3,
                                                                numBasisOutside=1)
        self.basisMultiDoF = self.basisGenerator.basisMultiDoF(self.time_normalised, self.nDoF)
        self.learnedProMP = promps.ProMP(self.basisGenerator, self.phaseGenerator, self.nDoF)
        self.learner = promps.MAPWeightLearner(self.learnedProMP)
        self.learnedData = self.learner.learnFromData(QList, timeList)
        self.mu_theta, self.sig_theta = self.learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))
        self.sig_theta = np.squeeze(self.sig_theta)

        A, Q = self.spacecraft_inv_kin.call_optimize(self.target_loc, ang_s0, q0)
        self.post_mean_q = Q[:, -1]

        self.taskProMP0 = self.learnedProMP.jointSpaceConditioning(0, desiredTheta=np.squeeze(self.q0),
                                                                   desiredVar=np.eye(len(self.q0)) * 0.00001)
        self.taskProMP = self.taskProMP0.jointSpaceConditioning(1.0, desiredTheta=self.post_mean_q, desiredVar=np.eye(
            len(self.q0)) * 0.00001)  # desiredVar=self.post_cov_q)

        self.trajectories_learned = self.learnedProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
        self.trajectories_task_conditioned = self.taskProMP.getTrajectorySamples(self.time_normalised, n_samples=20)


if __name__ == '__main__':
    time = np.linspace(0, 1, 15)
    nDoF, nBf = 3, 5
    target = np.array([-3., 0.75, 0.25])
    ang_s0, q0 = np.array([0., 0., 0.]), np.array([0., 5*np.pi / 4, np.pi/2]),  # np.array([[0.], [5*np.pi / 4], [np.pi/2]])
    traj_learn = TrajectoryLearning(time, target, ang_s0, q0, nDoF=nDoF, nBf=nBf)
    eef_curr_position = traj_learn.spacecraft_inv_kin.manip_eef_pos_num(ang_s0, q0)
    rs0 = traj_learn.spacecraft_inv_kin.spacecraft_com_num(ang_s0, q0)
    size = traj_learn.spacecraft_inv_kin.kin.size

    conditioned_trajs = traj_learn.trajectories_task_conditioned
    eef_pos = np.zeros((3, len(time)))
    eef_pos_list = []
    for j in range(conditioned_trajs.shape[2]):
        for i in range(conditioned_trajs.shape[0]):
            eef_pos[:, i] = traj_learn.spacecraft_inv_kin.manip_eef_pos_num(ang_s0, conditioned_trajs[i, :, j])
        eef_pos_list.append(eef_pos)

    traj_learn.spacecraft_sim.call_plot(rs0, size, 'red', ang_s0, q0)

    for i in range(len(eef_pos_list)):
        plt.plot(eef_pos_list[i][0], eef_pos_list[i][1], eef_pos_list[i][2])
    # plt.show()
    print('hi')