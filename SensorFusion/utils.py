import numpy as np


def scaleCovariance(A, P):
    """
    A: CxR
    P: CxC
    """
    R = A.shape[1]
    C = P.shape[0]
    APAt = np.zeros((R, R))
    for r in range(R):
        for j in range(r, R):
            apat = 0.
            for c in range(C):
                v = A[c][r] * P[c][c] * 0.5
                for k in range(c + 1, C):
                    v += A[k][r] * P[c][k]
                apat += 2 * v * A[c][j]

            APAt[j][r] = apat
            APAt[r][j] = apat
    return APAt


def crossMatrix(p, diag):
    r = np.diag(np.ones(3)*diag)
    r[0][1] = p[2]
    r[1][0] =-p[2]
    r[0][2] =-p[1]
    r[2][0] = p[1]
    r[1][2] = p[0]
    r[2][1] =-p[0]
    return r.T


def matrix_to_quaternion(R):
    Hx = R[0, 0]
    My = R[1, 1]
    Az = R[2, 2]

    q = np.array([0, 0, 0, 0], dtype=np.float64)
    q[1] = np.sqrt(np.max([(Hx - My - Az + 1) * 0.25, 0.]))
    q[2] = np.sqrt(np.max([(-Hx + My - Az + 1) * 0.25, 0.]))
    q[3] = np.sqrt(np.max([(-Hx - My + Az + 1) * 0.25, 0.]))
    q[0] = np.sqrt(np.max([(Hx + My + Az + 1) * 0.25, 0.]))

    q[0] *= np.sign(R[2, 1] - R[1, 2])
    q[1] *= np.sign(R[0, 2] - R[2, 0])
    q[2] *= np.sign(R[1, 0] - R[0, 1])
    # guaranteed to be unit-quaternion
    q = q[[1, 2, 3, 0]]
    return q


def  quat_to_matrix(q):
    q = q[[1, 2, 3, 0]]
    R = np.zeros((3, 3))
    sq_q1 = 2 * q[1] * q[1]
    sq_q2 = 2 * q[2] * q[2]
    sq_q3 = 2 * q[3] * q[3]
    q1_q2 = 2 * q[1] * q[2]
    q3_q0 = 2 * q[3] * q[0]
    q1_q3 = 2 * q[1] * q[3]
    q2_q0 = 2 * q[2] * q[0]
    q2_q3 = 2 * q[2] * q[3]
    q1_q0 = 2 * q[1] * q[0]
    R[0, 0] = 1 - sq_q2 - sq_q3
    R[0, 1] = q1_q2 - q3_q0
    R[0, 2] = q1_q3 + q2_q0
    R[1, 0] = q1_q2 + q3_q0
    R[1, 1] = 1 - sq_q1 - sq_q3
    R[1, 2] = q2_q3 - q1_q0
    R[2, 0] = q1_q3 - q2_q0
    R[2, 1] = q2_q3 + q1_q0
    R[2, 2] = 1 - sq_q1 - sq_q2
    return R

def normalize_quat(q):
    r = q
    if r[3] < 0:
        r = -r
    return r/np.linalg.norm(r)


def is_psd(m, tol=1e-5):
    return np.all(np.diag(m) > -1e-8) and np.all(np.abs(m - m.T) <= tol)

def dot22(A, B):
    return np.array([[A[0, 0].dot(B[0, 0]) +
                      A[0, 1].dot(B[1, 0]),
                      A[0, 0].dot(B[0, 1]) +
                      A[0, 1].dot(B[1, 1])],
                    [A[1, 0].dot(B[0, 0]) +
                     A[1, 1].dot(B[1, 0]),
                     A[1, 0].dot(B[0, 1]) +
                     A[1, 1].dot(B[1, 1])]])

def transpose22(A):
    return np.array([[A[0, 0], A[1, 0]],
                     [A[0, 1], A[1, 1]]])
