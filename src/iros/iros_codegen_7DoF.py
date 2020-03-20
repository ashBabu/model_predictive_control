import pickle
import numpy as np
import sympy as sym
from sympy.utilities.codegen import codegen
from sympy.utilities.autowrap import autowrap
# init_printing()


def get_symbols():
    alpha, beta, gamma = sym.symbols('alpha beta gamma')  # spacecraft angles
    alphaDot, betaDot, gammaDot = sym.symbols('alphaDot betaDot gammaDot')  # spacecraft velocities
    theta = sym.symbols('theta1:8')  # joint angles assuming 7DoF
    thetaDot = sym.symbols('thetaDot1:8')  # joint velocities
    ang_s = (alpha, beta, gamma)
    omega_s = (alphaDot, betaDot, gammaDot)
    return theta, thetaDot, ang_s, omega_s


theta, thetaDot, ang_s, omega_s = get_symbols()
states = sym.Matrix([*ang_s, *theta, *omega_s, *thetaDot])
# print(states)
y = sym.MatrixSymbol('y', *states.shape)
# dMass = sym.MatrixSymbol('dMass', int(0.5*np.multiply(*states.shape))**2, 1)
state_array_map = dict(zip(states, y))
print(state_array_map)

""" For Mass Matrix
with open('MassMatrix.pickle', 'rb') as inM:
    MassMatrix = pickle.loads(inM.read())

print('Loaded MassMatrix')
MassMatrix = MassMatrix.xreplace(state_array_map)
# massMatrix_eq = sym.Eq(dMass, MassMatrix)
print('Did xreplace')
codegen(('MassMatrix', MassMatrix), language='c', to_files=True)
# [(cf, cs), (hf, hs)] = codegen(('MassMatrix', MassMatrix), language='c')
print('MassMatrix done')
"""

""" For Coriolis & Centripetal vector 

with open('Coriolis.pickle', 'rb') as inC:
    Coriolis = pickle.loads(inC.read())

print('Loaded CoriolisVector')
Coriolis = Coriolis.xreplace(state_array_map)
# Coriolis_eq = sym.Eq(dMass, Coriolis)
print('Did xreplace')
codegen(('Coriolis', Coriolis), language='c', to_files=True)
# [(cf, cs), (hf, hs)] = codegen(('MassMatrix', MassMatrix), language='c')
print('Coriolis done')

"""


#            module name specified by `%%cython_pyximport` magic
#            |        just `modname + ".pyx"`
#            |        |
def make_ext(modname, pyxfilename):
    from setuptools.extension import Extension
    return Extension(modname,
                     sources=[pyxfilename, 'MassMatrix.c'],
                     include_dirs=['.', np.get_include()])

# with open('MassMatrix..pyxbld', 'wb') as inC: