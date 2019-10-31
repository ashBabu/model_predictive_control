import numpy as np
cimport numpy as np

cdef extern from 'wrapped_code_5.h':
    void autofunc(double Ixx1, double Ixx2, double Ixx3, double Iyy1, double Iyy2, double Iyy3, double Izz1, double Izz2, double Izz3, double alpha, double alpha_d, double beta, double beta_d, double gamma, double gamma_d, double m0, double m1, double m2, double m3, double theta_1, double theta_1d, double theta_2, double theta_2d, double theta_3, double theta_3d, double *out_3623850070012617309)

def autofunc_c(double Ixx1, double Ixx2, double Ixx3, double Iyy1, double Iyy2, double Iyy3, double Izz1, double Izz2, double Izz3, double alpha, double alpha_d, double beta, double beta_d, double gamma, double gamma_d, double m0, double m1, double m2, double m3, double theta_1, double theta_1d, double theta_2, double theta_2d, double theta_3, double theta_3d):

    cdef np.ndarray[np.double_t, ndim=2] out_3623850070012617309 = np.empty((3,3))
    autofunc(Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, alpha_d, beta, beta_d, gamma, gamma_d, m0, m1, m2, m3, theta_1, theta_1d, theta_2, theta_2d, theta_3, theta_3d, <double*> out_3623850070012617309.data)
    return out_3623850070012617309