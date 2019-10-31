import numpy as np
cimport numpy as np

cdef extern from 'wrapped_code_3.h':
    void autofunc(double Ixx1, double Ixx2, double Ixx3, double Iyy1, double Iyy2, double Iyy3, double Izz1, double Izz2, double Izz3, double alpha, double beta, double gamma, double m0, double m1, double m2, double m3, double theta_1, double theta_2, double theta_3, double *out_5519968318545254017)

def autofunc_c(double Ixx1, double Ixx2, double Ixx3, double Iyy1, double Iyy2, double Iyy3, double Izz1, double Izz2, double Izz3, double alpha, double beta, double gamma, double m0, double m1, double m2, double m3, double theta_1, double theta_2, double theta_3):

    cdef np.ndarray[np.double_t, ndim=2] out_5519968318545254017 = np.empty((3,3))
    autofunc(Ixx1, Ixx2, Ixx3, Iyy1, Iyy2, Iyy3, Izz1, Izz2, Izz3, alpha, beta, gamma, m0, m1, m2, m3, theta_1, theta_2, theta_3, <double*> out_5519968318545254017.data)
    return out_5519968318545254017