import numpy as np
cimport numpy as np

cdef extern from 'wrapped_code_0.h':
    void autofunc(double Is_xx, double Is_yy, double Is_zz, double Ixx1, double Ixx2, double Ixx3, double Iyy1, double Iyy2, double Iyy3, double Izz1, double Izz2, double Izz3, double alpha, double beta, double gamma, double m0, double m1, double m2, double m3, double theta_1, double theta_2, double theta_3, double *out_5832402673795264142)

def autofunc_c(double Is_xx, double Is_yy, double Is_zz, double Ixx1, double Ixx2, double Ixx3, double Iyy1, double Iyy2, double Iyy3, double Izz1, double Izz2, double Izz3, double alpha, double beta, double gamma, double m0, double m1, double m2, double m3, double theta_1, double theta_2, double theta_3):

    cdef np.ndarray[np.double_t, ndim=2] out_5832402673795264142 = np.empty((6,6))
    autofunc(Is_xx, Is_yy, Is_zz, Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, beta, gamma, m0, m1, m2, m3, theta_1, theta_2, theta_3, <double*> out_5832402673795264142.data)
    return out_5832402673795264142