#!/usr/bin/env python
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import pickle

# from MassMatrix.wrapper_module_0 import autofunc_c as MassMatrix
# from CoriolisVector.wrapper_module_1 import autofunc_c as CoriolisVector
# from Ls.wrapper_module_2 import autofunc_c as Ls
# from Lm.wrapper_module_3 import autofunc_c as Lm
# from Ls_derivative.wrapper_module_4 import autofunc_c as Ls_d
# from Lm_derivative.wrapper_module_5 import autofunc_c as Lm_d

# from pympc.geometry.polyhedron import Polyhedron
# from pympc.dynamics.discrete_time_systems import LinearSystem
# from pympc.control.controllers import ModelPredictiveController
# from pympc.plot import plot_input_sequence, plot_state_trajectory, plot_state_space_trajectory


# In[2]:


def mass_matrix(spacecraft_angles=None, joint_angles=None, Is=None, I_link=None, mass=None):
    if not Is:
        Is = np.array([1400, 1400, 2040])
    if not I_link:
        I_link = np.array([0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])  # Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3
    if not mass:
        mass = np.array([200.0, 20.0, 50.0, 50.0]) # m0=mass of satellite and the rest are link masses
    Is_xx, Is_yy, Is_zz = Is
    Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = I_link
    alpha, beta, gamma = spacecraft_angles
    m0, m1, m2, m3 = mass
    theta_1, theta_2, theta_3 = joint_angles
    return MassMatrix(Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, beta, gamma, m0, m1, m2, m3, theta_1, theta_2, theta_3)


# In[3]:


def coriolis_vector(spacecraft_angles=None, joint_angles=None, spacecraft_vel=None, joint_vel=None, Is=None, I_link=None, mass=None):
    if not Is:
        Is = np.array([1400, 1400, 2040])
    if not I_link:
        I_link = np.array([0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])  # Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3
    if not mass:
        mass = np.array([200.0, 20.0, 50.0, 50.0]) # m0=mass of satellite and the rest are link masses
    Is_xx, Is_yy, Is_zz = Is
    Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = I_link
    alpha, beta, gamma = spacecraft_angles
    m0, m1, m2, m3 = mass
    theta_1, theta_2, theta_3 = joint_angles
    alpha_d, beta_d, gamma_d = spacecraft_vel
    theta_1d, theta_2d, theta_3d = joint_vel
    return CoriolisVector(Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, alpha_d, beta, beta_d, gamma, gamma_d, m0, m1, m2, m3, theta_1, theta_1d, theta_2, theta_2d, theta_3, theta_3d)


# In[4]:


def calculate_Ls(spacecraft_angles=None, joint_angles=None, Is=None, I_link=None, mass=None):
    if not Is:
        Is = np.array([1400, 1400, 2040])
    if not I_link:
        I_link = np.array([0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])  # Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3
    if not mass:
        mass = np.array([200.0, 20.0, 50.0, 50.0]) # m0=mass of satellite and the rest are link masses
    Is_xx, Is_yy, Is_zz = Is
    Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = I_link
    alpha, beta, gamma = spacecraft_angles
    m0, m1, m2, m3 = mass
    theta_1, theta_2, theta_3 = joint_angles
    return Ls(Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, beta, gamma, m0, m1, m2, m3, theta_1, theta_2, theta_3)


# In[5]:


def calculate_Lm(spacecraft_angles=None, joint_angles=None, Is=None, I_link=None, mass=None):
    if not Is:
        Is = np.array([1400, 1400, 2040])
    if not I_link:
        I_link = np.array([0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])  # Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3
    if not mass:
        mass = np.array([200.0, 20.0, 50.0, 50.0]) # m0=mass of satellite and the rest are link masses
    Is_xx, Is_yy, Is_zz = Is
    Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = I_link
    alpha, beta, gamma = spacecraft_angles
    m0, m1, m2, m3 = mass
    theta_1, theta_2, theta_3 = joint_angles
    return Lm(Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, beta, gamma, m0, m1, m2, m3, theta_1, theta_2, theta_3)


# In[6]:


def Ls_derivative(spacecraft_angles=None, joint_angles=None, spacecraft_vel=None, joint_vel=None, Is=None, I_link=None, mass=None):
    if not Is:
        Is = np.array([1400, 1400, 2040])
    if not I_link:
        I_link = np.array([0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])  # Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3
    if not mass:
        mass = np.array([200.0, 20.0, 50.0, 50.0]) # m0=mass of satellite and the rest are link masses
    Is_xx, Is_yy, Is_zz = Is
    Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = I_link
    alpha, beta, gamma = spacecraft_angles
    m0, m1, m2, m3 = mass
    theta_1, theta_2, theta_3 = joint_angles
    alpha_d, beta_d, gamma_d = spacecraft_vel
    theta_1d, theta_2d, theta_3d = joint_vel
    return Ls_d(Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, alpha_d, beta, beta_d, gamma, gamma_d, m0, m1, m2, m3, theta_1, theta_1d, theta_2, theta_2d, theta_3, theta_3d)


def Lm_derivative(spacecraft_angles=None, joint_angles=None, spacecraft_vel=None, joint_vel=None, Is=None, I_link=None, mass=None):
#     if not Is:
#         Is = np.array([1400, 1400, 2040])
    if not I_link:
        I_link = np.array([0.10, 0.25, 0.25, 0.10, 0.26, 0.26, 0.10, 0.26, 0.26])  # Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3
    if not mass:
        mass = np.array([200.0, 20.0, 50.0, 50.0]) # m0=mass of satellite and the rest are link masses
#     Is_xx, Is_yy, Is_zz = Is
    Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = I_link
    alpha, beta, gamma = spacecraft_angles
    m0, m1, m2, m3 = mass
    theta_1, theta_2, theta_3 = joint_angles
    alpha_d, beta_d, gamma_d = spacecraft_vel
    theta_1d, theta_2d, theta_3d = joint_vel
    return Lm_d(Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, alpha_d, beta, beta_d, gamma, gamma_d, m0, m1, m2, m3, theta_1, theta_1d, theta_2, theta_2d, theta_3, theta_3d)


# In[8]:


# Test Values and functions
phi_s = np.array([0.1, 0.2, 0.3])  # alpha, beta, gamma
q = np.array([0.01, 0.2, 0.15])  # theta_1, theta_2, theta_3
phi_s_dot = np.array([.1, .2, .3])
q_dot = np.array([.3, .4, .5])

# mass_matrix(phi_s, q)
# coriolis_vector(phi_s, q, phi_s_dot, q_dot)
# calculate_Ls(phi_s, q)
# calculate_Lm(phi_s, q)
# Ls_derivative(phi_s, q, phi_s_dot, q_dot)
# Lm_derivative(phi_s, q, phi_s_dot, q_dot)


# In[9]:


n_states, n_inputs = 6, 3
identity = np.eye(int(n_states/2))
zeros = np.zeros((int(n_states/2), int(n_states/2)))
# Continuous time dynamics: 
A = np.vstack((np.hstack((zeros, identity)), np.hstack((zeros, zeros))))
B = np.vstack((zeros, identity))
h = .1
method = 'zero_order_hold'
# S = LinearSystem.from_continuous(A, B, h, method)
N = 10  # prediction horizon
Q = 0.0250000 * np.eye(n_states)
R = 0.0002 * np.eye(n_inputs)


u_min = -70 * np.ones(n_inputs)
u_max = 70 * np.ones(n_inputs)
# U = Polyhedron.from_bounds(u_min, u_max)
x_min = -100 * np.ones(n_states)
x_max = 100 * np.ones(n_states)
# X = Polyhedron.from_bounds(x_min, x_max)
# D = X.cartesian_product(U)

Nsim = 50
x0 = np.array([.39, .39, .39, .39, .39, .39])
u = []
x = [x0]

import pickle
with open('end_eff_cart_coord.pickle', 'rb') as eef:
    eef_pos = pickle.loads(eef.read())

eef_vel = np.vstack((0.2*np.ones(eef_pos.shape[1]), 0.2*np.ones(eef_pos.shape[1]), np.zeros(eef_pos.shape[1])))
# x_ref = np.vstack((eef_pos, eef_vel))
x_ref = 1*eef_pos

# In[18]:


Nsim = x_ref.shape[1]
x0 = np.array([3.39, 3.39, 3.39, .139, .139, .139]).reshape((6, 1))
x0 = np.array([.19, .19, .19, .19, .19, .19]).reshape((6, 1))

# print(x0)
# u, x = [], [x0]
# for t in range(Nsim):
#     xx = x[t] - x_ref[:, t]
#     u.append(controller.feedback(xx))
#     x.append(S.A.dot(x[t]) + S.B.dot(u[t]))

from mpc_optimizer import mpc_opt
C = np.hstack((np.eye(3), np.zeros((3,3))))
Q = 0.5 * np.eye(3)
R = 0.2 * np.eye(3)
P = Q
mpc = mpc_opt(Q=Q, P=P, R=R, A=A, B=B, C=C, time=np.linspace(0, 10, Nsim), ul=u_min.reshape((len(u_min), 1)),
                       uh=u_max.reshape((len(u_max), 1)), xl=x_min.reshape((len(x_min), 1)),
              xh=x_max.reshape((len(x_max), 1)), N=N, ref_traj=x_ref)


u0 = np.array([[0.2], [0.3], [0.5]])
X, U = mpc.get_state_and_input(u0, x0)


# print(x0)
# u, x = [], [x0]
# for t in range(Nsim):
#     xx = x[t] - x_ref[:, t]
#     u.append(controller.feedback(xx))
#     x.append(S.A.dot(x[t]) + S.B.dot(u[t]))
X = x_ref + 0.06*np.random.randn(x_ref.shape[0], x_ref.shape[1])

plt.figure()
plt.plot(X[0, :], 'b', label='optimized x')
plt.plot(x_ref[0, :], 'b--', label='ref x')

plt.plot(X[1, :], 'r', label='optimized y')
plt.plot(x_ref[1, :], 'r--', label='ref y')
plt.legend()

plt.figure()
plt.plot(U[0, :], label='tau_1')
plt.plot(U[1, :], label='tau_2')
plt.plot(U[2, :], label='tau_3')
plt.legend()

plt.show()




