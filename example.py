import constants
import numpy as np

from SensorFusion.fusion import Fusion


if __name__ == '__main__':
    fusion = Fusion(constants.FUSION_9AXIS)
    acceleration = np.array([[-1.9536686,  3.7732618,  9.165004],
                             [-1.9536686,  3.7732618,  9.126697],
                             [-1.9440918,  3.7732618,  9.088389],
                             [-1.9536686,  3.7732618,  9.059659],
                             [-1.9632454,  3.7828386,  9.030929]])

    gyroscope = np.array([[0.28099802, -0.02932153,  0.5974262],
                          [0.24190264,  0.0140499,  0.61086524],
                          [0.2162463,  0.04276057,  0.62002819],
                          [0.18529579,  0.0749328,  0.63000567],
                          [0.15088372,  0.11178834,  0.6450737]])
    magnetometr = np.array([[-11.86875, -29.075, -21.99375],
                            [-12.03125, -29.25, -22.15625],
                            [-12.19375, -29.425, -22.31875],
                            [-12.35625, -29.6, -22.48125],
                            [-12.5875, -29.7, -22.4 ]])
    dt = 0.002
    fusion.handleGyro(gyroscope[0], dt)
    fusion.handleAcc(acceleration[0], dt)
    fusion.handleMag(magnetometr[0])

    ## Don't worry about printing 'Bad value'. Firstly, you need to put initial values for acc, gyro, magnet

    fusion.handleGyro(gyroscope[1], dt)
    fusion.handleAcc(acceleration[1], dt)
    fusion.handleMag(magnetometr[1])

    q0 = fusion.x0 # q = [x, y, z, w]
    print(q0)

    fusion.handleGyro(gyroscope[2], dt)
    fusion.handleAcc(acceleration[2], dt)
    fusion.handleMag(magnetometr[2])

    q1 = fusion.x0  # q = [x, y, z, w]
    print(q1)
