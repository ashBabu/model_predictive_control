import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=3)


def rot_mat_3d(*args):
    if isinstance(args, tuple):
        ang_x, ang_y, ang_z = args[0]
    else:
        ang_x, ang_y, ang_z = args
    Rx = R.from_rotvec(ang_x * np.array([1, 0, 0]))
    Ry = R.from_rotvec(ang_y * np.array([0, 1, 0]))
    Rz = R.from_rotvec(ang_z * np.array([0, 0, 1]))
    Rx, Ry, Rz = Rx.as_dcm(), Ry.as_dcm(), Rz.as_dcm()
    return Rx @ Ry @ Rz


def rotation_matrix(ang):
    rx = [[1, 0, 0], [0, np.cos(ang[0]), -np.sin(ang[0])], [0, np.sin(ang[0]), np.cos(ang[0])]]
    ry = [[np.cos(ang[1]), 0, np.sin(ang[1])], [0, 1, 0], [-np.sin(ang[1]), 0, np.cos(ang[1])]]
    rz = [[np.cos(ang[2]), -np.sin(ang[2]), 0], [np.sin(ang[2]), np.cos(ang[2]), 0], [0, 0, 1]]
    rx, ry, rz = np.array(rx), np.array(ry), np.array(rz)
    Rot = rx @ ry @ rz
    return Rot
