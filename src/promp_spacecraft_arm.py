import numpy as np
import pickle
import phase as phase
import basis as basis
import promps as promps
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
    def __init__(self, time, nDoF=3, nBf=5, ):
        self.nDoF = nDoF
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



        self.post_mean_q, self.post_cov_q = self.franka_kin.inv_kin_ash_pose_quaternion(np.squeeze(self.mu_theta),
                                                                                        self.sig_theta, self.mu_x,
                                                                                        self.sig_x, self.mu_quat_des,
                                                                                        self.sig_quat)
        self.taskProMP0 = self.learnedProMP.jointSpaceConditioning(0, desiredTheta=self.curr_jvs,
                                                                   desiredVar=np.eye(len(self.curr_jvs)) * 0.00001)
        self.taskProMP = self.taskProMP0.jointSpaceConditioning(1.0, desiredTheta=self.post_mean_q, desiredVar=np.eye(
            len(self.curr_jvs)) * 0.00001)  # desiredVar=self.post_cov_q)

        self.trajectories_learned = self.learnedProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
        self.trajectories_task_conditioned = self.taskProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
