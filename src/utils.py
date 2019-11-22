# external imports
import numpy as np
from scipy.linalg import expm


def nullspace_basis(A):
    """
    Uses SVD to find a basis of the nullsapce of A.

    Arguments
    ----------
    A : numpy.ndarray
        Matrix for the nullspace.

    Returns
    ----------
    Z : numpy.ndarray
        Nullspace basis.
    """

    # get singular values
    V = np.linalg.svd(A)[2].T

    # cut to the dimension of the rank
    rank = np.linalg.matrix_rank(A)
    Z = V[:,rank:]

    return Z


def linearly_independent_rows(A, tol=1.e-6):
    """
    uses the QR decomposition to find the indices of a set of linear independent rows of the matrix A.

    Arguments
    ----------
    A : numpy.ndarray
        Matrix for the linear independent rows.
    tol : float
        Threshold value for the diagonal elements of R.

    Returns
    ----------
    independent_rows : list of int
        List of indices of a set of independent rows of A.
    """

    # QR decomposition
    R = np.linalg.qr(A.T)[1]

    # check diagonal elements
    R_diag = np.abs(np.diag(R))
    independent_rows = list(np.where(R_diag > tol)[0])

    return sorted(independent_rows)


def plane_through_points(points):
    """
    Returns the plane a' x = b passing through the points.
    It first adds a random offset to be sure that the matrix of the points is invertible (it wouldn't be the case if the plane we are looking for passes through the origin).
    The vector a has norm equal to one and b is non-negative.

    Arguments
    ----------
    points : list of numpy.ndarray
        List of points that the plane has to fit.

    Returns
    ----------
    a : numpy.ndarray
        Left-hand side of the equality describing the plane.
    d : numpy.ndarray
        Right-hand side of the equality describing the plane.
    """

    # generate random offset
    offset = np.random.rand(points[0].size)
    points = [p + offset for p in points]

    # solve linear system
    P = np.vstack(points)
    a = np.linalg.solve(P, np.ones(points[0].size))

    # go back to the original coordinates
    d = 1. - a.dot(offset)

    # scale and select sign of the result
    a_norm = np.linalg.norm(a)
    d_sign = np.sign(d)
    if d_sign == 0.:
        d_sign = 1.
    a /= a_norm * d_sign
    d /= a_norm * d_sign

    return a, d


def same_rows(A, B, normalize=True):
    """
    Checks if two matrices contain the same rows.
    The order of the rows can be different.
    The option normalize, normalizes the rows of A and B; i.e., if True, checks that set of rows of A is the same of the one of B despite a scaling factor.

    Arguments
    ----------
    A : numpy.ndarray
        First matrix to check.
    B : numpy.ndarray
        Second matrix to check.
    normalize : bool
        If True scales the rows of A and B to have norm equal to one.

    Returns:
    equal : bool
        True if the set of rows of A and B are the same.
    """

    # first check the sizes
    if A.shape[0] != B.shape[0]:
        return False

    # if required, normalize
    if normalize:
        for i in range(A.shape[0]):
            A[i] = A[i] / np.linalg.norm(A[i])
            B[i] = B[i] / np.linalg.norm(B[i])

    # check one row per time
    for a in A:
        i = np.where([np.allclose(a, b) for b in B])[0]
        if len(i) != 1:
            return False
        B = np.delete(B, i, 0)

    return True


def same_vectors(v_list, u_list):
    """
    Tests that two lists of array contain the same elements.
    The order of the elements in the lists can be different.

    Arguments
    ----------
    v_list : list of numpy.ndarray
        First ist of arrays to be checked.
    u_list : list of numpy.ndarray
        Second ist of arrays to be checked.

    Returns:
    equal : bool
        True if the set of arrays oin v_list and u_list are the same.
    """

    # check inputs
    for z_list in [v_list, u_list]:
        if any(len(z.shape) >= 2 for z in z_list):
            raise ValueError('input vectors must be 1-dimensional arrays.')

    # construct matrices
    V = np.vstack(v_list)
    U = np.vstack(u_list)

    return same_rows(V, U, False)


def check_affine_system(A, B, c=None, h=None):
    """
    Check that the matrices A, B, and c of an affine system have compatible sizes.

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    B : numpy.ndarray
        Input to state map.
    c : numpy.ndarray
        Offset term.
    h : float
        Discretization time step.
    """

    # A square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')

    # equal number of rows for A and B
    if A.shape[0] != B.shape[0]:
        raise ValueError('A and B must have the same number of rows.')

    # check c
    if c is not None:
        if c.ndim > 1:
            raise ValueError('c must be a 1-dimensional array.')
        if A.shape[0] != c.size:
            raise ValueError('A and c must have the same number of rows.')

    # check h
    if h is not None:
        if h < 0:
            raise ValueError('the time step h must be positive.')


def zero_order_hold(A, B, c, h):
    """
    Assuming piecewise constant inputs, it returns the exact discretization of the affine system dx/dt = A x + B u + c.

    Math
    ----------
    Solving the differential equation, we have
    x(h) = exp(A h) x(0) + int_0^h exp(A (h - t)) (B u(t) + c) dt.
    Being u(t) = u(0) constant between 0 and h we have
    x(h) = A_d x(0) + B_d u(0) + c_d,
    where
    A_d := exp(A h),
    B_d := int_0^h exp(A (h - t)) dt B,
    c_d = B_d := int_0^h exp(A (h - t)) dt c.
    I holds
         |A B c|      |A_d B_d c_d|
    exp (|0 0 0| h) = |0   I   0  |
         |0 0 0|      |0   0   1  |
    where both the matrices are square.
    Proof: apply the definition of exponential and note that int_0^h exp(A (h - t)) dt = sum_{k=1}^inf A^(k-1) h^k/k!.

    Arguments
    ----------
    A : numpy.ndarray
        State transition matrix.
    B : numpy.ndarray
        Input to state map.
    c : numpy.ndarray
        Offset term.
    h : float
        Discretization time step.

    Returns
    ----------
    A_d : numpy.ndarray
        Discrete-time state transition matrix.
    B_d : numpy.ndarray
        Discrete-time input to state map.
    c_d : numpy.ndarray
        Discrete-time offset term.
    """

    # check inputs
    check_affine_system(A, B, c, h)

    # system dimensions
    n_x = np.shape(A)[0]
    n_u = np.shape(B)[1]

    # zero order hold
    M_c = np.vstack((
        np.column_stack((A, B, c)),
        np.zeros((n_u + 1, n_x + n_u + 1))
    ))
    M_d = expm(M_c * h)

    # discrete time dynamics
    A_d = M_d[:n_x, :n_x]
    B_d = M_d[:n_x, n_x:n_x + n_u]
    c_d = M_d[:n_x, n_x + n_u]

    return A_d, B_d, c_d