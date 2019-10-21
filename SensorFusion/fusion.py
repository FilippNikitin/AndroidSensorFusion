import numpy as np

from SensorFusion.utils import *
from SensorFusion.constants import *


class Parameter:
    def __init__(self, gyroVar=None, gyroBiasVar=None, accStdev=None, magStdev=None):
        self.gyroVar = gyroVar
        self.gyroBiasVar = gyroBiasVar
        self.accStdev = accStdev
        self.magStdev = magStdev


class Fusion:
    def __init__(self, mode):
        self.x0 = np.array([0, 0, 0, 0.], dtype=np.float64)  # quat
        self.x1 = np.array([0, 0, 0.], dtype=np.float64) #bias
        self.P = np.zeros((2, 2, 3, 3), dtype=np.float64)
        self.GQGt = np.zeros((2, 2, 3, 3), dtype=np.float64)
        self.mParam = Parameter()
        self.Phi = np.zeros((2, 2, 3, 3), dtype=np.float64)
        self.Phi[1, 1] = np.eye(3, dtype=np.float64)
        self.Ba = np.array([0, 0, 1.], dtype=np.float64)
        self.Bm = np.array([0, 1, 0.], dtype=np.float64)
        self.init(mode=mode)

    def init(self, mode=FUSION_9AXIS):
        self.mInitState = np.array([False, False, False])
        self.mGyroRate = 0.
        self.mCount = np.array([0, 0, 0])
        self.mData = np.zeros((3, 3))
        self.mMode = mode

        if (self.mMode != FUSION_NOGYRO):
            self.mParam.gyroVar = DEFAULT_GYRO_VAR
            self.mParam.gyroBiasVar = DEFAULT_GYRO_BIAS_VAR
            self.mParam.accStdev = DEFAULT_ACC_STDEV
            self.mParam.magStdev = DEFAULT_MAG_STDEV
        else:
            self.mParam.gyroVar = GEOMAG_GYRO_VAR
            self.mParam.gyroBiasVar = GEOMAG_GYRO_BIAS_VAR
            self.mParam.accStdev = GEOMAG_ACC_STDEV
            self.mParam.magStdev = GEOMAG_MAG_STDEV

    def initFusion(self, q, dT):
        # initial estimate: E{ x(t0) }
        self.x0 = q
        # process noise covariance matrix: G.Q.Gt, with
        #
        #  G = | -1 0 |        Q = | q00 q10 |
        #      |  0 1 |            | q01 q11 |
        #
        # q00 = sv^2.dt + 1/3.su^2.dt^3
        # q10 = q01 = 1/2.su^2.dt^2
        # q11 = su^2.dt
        #
        dT2 = dT * dT
        dT3 = dT2 * dT
        # variance of integrated output at 1/dT Hz (random drift)
        q00 = self.mParam.gyroVar * dT + 0.33333 * self.mParam.gyroBiasVar * dT3
        # variance of drift rate ramp
        q11 = self.mParam.gyroBiasVar * dT
        q10 = 0.5 * self.mParam.gyroBiasVar * dT2
        q01 = q10
        self.GQGt[0, 0] = q00 * np.eye(3)  # rad^2
        self.GQGt[1, 0] = -q01 * np.eye(3)
        self.GQGt[0, 1] = -q10 * np.eye(3)
        self.GQGt[1, 1] = q11 * np.eye(3)  # (rad/s)^2
        # initial covariance: Var{ x(t0) }
        # TODO: initialize P correctly

    def hasEstimate(self):
        return ((self.mInitState[MAG]) | (self.mMode == FUSION_NOMAG)) & (
                    (self.mInitState[GYRO]) | (self.mMode == FUSION_NOGYRO)) & (self.mInitState[ACC])

    def checkInitComplete(self, what, d, dT=None):
        if self.hasEstimate():
            return True

        if (what == ACC):
            self.mData[0] += d / np.linalg.norm(d)
            self.mCount[0] += 1
            self.mInitState[ACC] = True
            if (self.mMode == FUSION_NOGYRO):
                self.mGyroRate = dT

        elif (what == MAG):
            self.mData[1] += d * (1 / np.linalg.norm(d))
            self.mCount[1] += 1
            self.mInitState[MAG] = True

        elif (what == GYRO):
            self.mGyroRate = dT
            self.mData[2] += d * dT
            self.mCount[2] += 1
            self.mInitState[GYRO] = True

        if self.hasEstimate():
            # Average all the values we collected so far

            self.mData[0] *= 1.0 / self.mCount[0]
            if (self.mMode != FUSION_NOMAG):
                self.mData[1] *= 1.0 / self.mCount[1]

            self.mData[2] *= 1.0 / self.mCount[2]

            # calculate the MRPs from the data collection, this gives us
            # a rough estimate of our initial state

            R = np.ndarray((3, 3))
            up = self.mData[0]

            if (self.mMode != FUSION_NOMAG):
                east = np.cross(self.mData[1], up)
                east = east / np.linalg.norm(east)
            else:
                east = self.getOrthogonal(up)

            north = np.cross(up, east)
            ### a very strange monemt
            # R << east << north << up;
            R = np.c_[east, north, up]
            q = matrix_to_quaternion(R.T)
            self.initFusion(q, self.mGyroRate)
        return False

    def handleGyro(self, w, dT):
        if (not self.checkInitComplete(GYRO, w, dT)):
            print('Bad value')
            return BAD_VALUE
        self.predict(w, dT)

    def handleAcc(self, a, dT):
        if (not self.checkInitComplete(ACC, a, dT)):
            print('Bad value')
            return BAD_VALUE
        # ignore acceleration data if we're close to free-fall

        l = np.linalg.norm(a)
        if (l < FREE_FALL_THRESHOLD):
            print('Bad value')
            return BAD_VALUE

        l_inv = 1.0 / l
        if (self.mMode == FUSION_NOGYRO):
            # geo mag
            w_dummy = self.x1  # bias
            self.predict(w_dummy, dT)

        if (self.mMode == FUSION_NOMAG):
            m = self.getRotationMatrix().dot(self.Bm)
            self.update(m, self.Bm, self.mParam.magStdev)

        unityA = a * l_inv;

        d = np.sqrt(np.abs(l - NOMINAL_GRAVITY))
        p = l_inv * self.mParam.accStdev * np.exp(d)
        self.update(unityA, self.Ba, p)
        return NO_ERROR

    def handleMag(self, m):
        if (not self.checkInitComplete(MAG, m)):
            print('Bad value')
            return BAD_VALUE
        # the geomagnetic-field should be between 30uT and 60uT
        # reject if too large to avoid spurious magnetic sources

        magFieldSq = np.sum(m ** 2)
        if (magFieldSq > MAX_VALID_MAGNETIC_FIELD_SQ):
            print('Bad value')
            return BAD_VALUE

        elif (magFieldSq < MIN_VALID_MAGNETIC_FIELD_SQ):
            # Also reject if too small since we will get ill-defined (zero mag)
            # cross-products below
            print('Bad value')
            return BAD_VALUE
        # Orthogonalize the magnetic field to the gravity field, mapping it into
        # tangent to Earth.
        up = self.getRotationMatrix().dot(self.Ba)
        east = np.cross(m, up)
        # If the m and up vectors align, the cross product magnitude will
        # approach 0.
        # Reject this case as well to avoid div by zero problems and
        # ill-conditioning below.
        if (np.sum(east ** 2) < MIN_VALID_CROSS_PRODUCT_MAG_SQ):
            return BAD_VALUE
        # If we have created an orthogonal magnetic field successfully,
        # then pass it in as the update.
        north = np.cross(up, east)
        l_inv = 1 / np.linalg.norm(north)
        north *= l_inv
        self.update(north, self.Bm, self.mParam.magStdev * l_inv)
        return NO_ERROR

    def checkState(self):
        # P needs to stay positive semidefinite or the fusion diverges. When we
        # detect divergence, we reset the fusion.
        # TODO(braun): Instead, find the reason for the divergence and fix it.
        if (not is_psd(self.P[0, 0], tol=SYMMETRY_TOLERANCE)) | (not is_psd(self.P[1, 1], tol=SYMMETRY_TOLERANCE)):
            # ALOGW("Sensor fusion diverged; resetting state.")
            # print('State is ill')
            self.P = np.zeros((2, 2, 3, 3))

    def getAttitude(self):
        return self.x0

    def getBias(self):
        return self.x1

    def getRotationMatrix(self):
        return quat_to_matrix(self.x0)

    def getF(self, q):
        # This is used to compute the derivative of q
        # F = | [q.xyz]x |
        #     |  -q.xyz  |
        # 3 x 4
        F = np.array([[q[3], q[2], -q[1], -q[0]],
                      [-q[2], q[3], q[0], -q[1]],
                      [q[1], -q[0], q[3], -q[2]]], dtype=np.float64)
        return F.T

    def predict(self, w, dT):
        q = self.x0
        b = self.x1
        we = w - b
        if (np.linalg.norm(we) < WVEC_EPS):
            we = np.sign(we[0]) * np.ones(3) * WVEC_EPS

        # q(k+1) = O(we)*q(k)
        # --------------------
        #
        # O(w) = | cos(0.5*||w||*dT)*I33 - [psi]x                   psi |
        #        | -psi'                              cos(0.5*||w||*dT) |
        #
        # psi = sin(0.5*||w||*dT)*w / ||w||
        #
        #
        # P(k+1) = Phi(k)*P(k)*Phi(k)' + G*Q(k)*G'
        # ----------------------------------------
        #
        # G = | -I33    0 |
        #     |    0  I33 |
        #
        #  Phi = | Phi00 Phi10 |
        #        |   0     1   |
        #
        #  Phi00 =   I33
        #          - [w]x   * sin(||w||*dt)/||w||
        #          + [w]x^2 * (1-cos(||w||*dT))/||w||^2
        #
        #  Phi10 =   [w]x   * (1        - cos(||w||*dt))/||w||^2
        #          - [w]x^2 * (||w||*dT - sin(||w||*dt))/||w||^3
        #          - I33*dT

        I33 = np.eye(3)
        I33dT = np.eye(3) * dT
        wx = crossMatrix(we, 0)
        wx2 = wx.dot(wx)
        lwedT = np.linalg.norm(we) * dT
        hlwedT = 0.5 * lwedT
        ilwe = 1. / np.linalg.norm(we)
        k0 = (1 - np.cos(lwedT)) * ilwe ** 2
        k1 = np.sin(lwedT)
        k2 = np.cos(hlwedT)
        psi = np.sin(hlwedT) * ilwe * we
        O33 = crossMatrix(-psi, k2).T
        O = np.zeros((4, 4))

        O[0, :3] = O33[0]
        O[1, :3] = O33[1]
        O[2, :3] = O33[2]
        O[3, :3] = psi

        O[0, 3] = -psi[0]
        O[1, 3] = -psi[1]
        O[2, 3] = -psi[2]
        O[3, 3] = k2

        O = O.T

        self.Phi[0, 0] = I33 - wx * (k1 * ilwe) + wx2 * k0
        self.Phi[0, 1] = wx * k0 - I33dT - wx2 * (ilwe ** 3) * (lwedT - k1)

        self.x0 = O.dot(q)  # check dim!!!

        if (self.x0[3] < 0):
            self.x0 = -self.x0

        self.P = dot22(dot22(self.Phi, self.P), transpose22(self.Phi)) + self.GQGt  ### check the moment
        self.checkState()

    def update(self, z, Bi, sigma):
        q = self.x0
        # measured vector in body space: h(p) = A(p)*Bi
        A = quat_to_matrix(q)
        Bb = A.dot(Bi)
        # Sensitivity matrix H = dh(p)/dp
        # H = [ L 0 ]
        L = crossMatrix(Bb, 0)
        # gain...
        # K = P*Ht / [H*P*Ht + R]
        K = np.zeros((2, 3, 3))
        R = np.eye(3) * sigma ** 2
        S = scaleCovariance(L, self.P[0, 0]) + R
        Si = np.linalg.inv(S)
        LtSi = (L.T).dot(Si)
        K[0] = self.P[0, 0].dot(LtSi)
        K[1] = (self.P[0, 1].T).dot(LtSi)
        # update...
        # P = (I-K*H) * P
        # P -= K*H*P
        # | K0 | * | L 0 | * P = | K0*L  0 | * | P00  P10 | = | K0*L*P00  K0*L*P10 |
        # | K1 |                 | K1*L  0 |   | P01  P11 |   | K1*L*P00  K1*L*P10 |
        # Note: the Joseph form is numerically more stable and given by:
        #     P = (I-KH) * P * (I-KH)' + K*R*R'

        K0L = K[0].dot(L)
        K1L = K[1].dot(L)
        self.P[0, 0] -= K0L.dot(self.P[0, 0])
        self.P[1, 1] -= K1L.dot(self.P[0, 1])
        self.P[0, 1] -= K0L.dot(self.P[0, 1])
        self.P[1, 0] = self.P[0, 1].T
        e = z - Bb
        dq = K[0].dot(e)
        q += self.getF(q).dot(0.5 * dq)
        self.x0 = normalize_quat(q)
        if (self.mMode != FUSION_NOMAG):
            db = K[1].dot(e)
            self.x1 += db
        self.checkState()

    def getOrthogonal(self, v):
        w = np.zeros(3)
        min_id = np.argmin(np.abs(v))
        if min_id == 0:
            w[1] = v[2]
            w[2] = -v[1]
        elif min_id == 1:
            w[0] = v[2]
            w[2] = -v[0]
        else:
            w[0] = v[1]
            w[1] = -v[0]
        return w / np.linalg.norm(w)
