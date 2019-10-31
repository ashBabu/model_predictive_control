import pickle
from sympy import *
from sympy.utilities.autowrap import autowrap
init_printing()

alpha, beta, gamma = symbols('alpha beta gamma')
alpha_d, beta_d, gamma_d = symbols('alpha_d beta_d gamma_d')
theta_1, theta_2, theta_3 = symbols('theta_1 theta_2 theta_3')
theta_1d, theta_2d, theta_3d = symbols('theta_1d theta_2d theta_3d')
ang_s = [alpha, beta, gamma]
omega_s = [alpha_d, beta_d, gamma_d]
theta = [theta_1, theta_2, theta_3]
theta_d = [theta_1d, theta_2d, theta_3d]
tau_1, tau_2 = symbols('tau_1 tau_2')
tau = Matrix([tau_1, tau_2])
Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3 = symbols('Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3')
m0, m1, m2, m3 = symbols('m0, m1, m2, m3')

states = Matrix([alpha, beta, gamma, theta_1, theta_2, theta_3, alpha_d, beta_d, gamma_d, theta_1d, theta_2d, theta_3d])
const = Matrix([Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, m0, m1, m2, m3])

with open('MassMat_sym.pickle', 'rb') as inM:
    MassMatrix = pickle.loads(inM.read())
with open('Corioli_sym.pickle', 'rb') as inC:
    CoriolisVector = pickle.loads(inC.read())

with open('Ls.pickle', 'rb') as LsR:
    Ls = pickle.loads(LsR.read())
with open('Lm.pickle', 'rb') as LmR:
    Lm = pickle.loads(LmR.read())
with open('Ls_d.pickle', 'rb') as LsdR:
    Ls_d = pickle.loads(LsdR.read())
with open('Lm_d.pickle', 'rb') as LmdR:
    Lm_d = pickle.loads(LmdR.read())

# [(cf, cs), (hf, hs)] = codegen(('c_odes', M2), language='c')
# print(cs)
# sub_exprs1, simplified_MassMatrix = cse(MassMatrix)
# sub_exprs2, simplified_Coriolis = cse(CoriolisVector)
# sub_exprs3, simplified_Ls = cse(Ls)
# sub_exprs4, simplified_Lm = cse(Lm)
# sub_exprs5, simplified_Ls_d = cse(Ls_d)
# sub_exprs6, simplified_Lm_d = cse(Lm_d)

Mass_C = autowrap(MassMatrix, backend='cython', tempdir='./MassMatrix')
Coriolis_C = autowrap(CoriolisVector, backend='cython', tempdir='./CoriolisVector')
Ls_C = autowrap(Ls, backend='cython', tempdir='./Ls')
Lm_C = autowrap(Lm, backend='cython', tempdir='./Lm')
Lsd_C = autowrap(Ls_d, backend='cython', tempdir='./Ls_derivative')
Lmd_C = autowrap(Lm_d, backend='cython', tempdir='./Lm_derivative')

print('done')

