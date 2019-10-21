DEFAULT_GYRO_VAR = 1e-7        # (rad/s)^2 / Hz
DEFAULT_GYRO_BIAS_VAR = 1e-12  # (rad/s)^2 / s (guessed)
GEOMAG_GYRO_VAR = 1e-4         # (rad/s)^2 / Hz
GEOMAG_GYRO_BIAS_VAR = 1e-8    # (rad/s)^2 / s (guessed)

DEFAULT_ACC_STDEV  = 0.015        # m/s^2 (measured 0.08 / CDD 0.05)
DEFAULT_MAG_STDEV  = 0.1          # uT    (measured 0.7  / CDD 0.5)
GEOMAG_ACC_STDEV  = 0.05          # m/s^2 (measured 0.08 / CDD 0.05)
GEOMAG_MAG_STDEV  = 0.1           #uT    (measured 0.7  / CDD 0.5)

SYMMETRY_TOLERANCE = 1e-10

NOMINAL_GRAVITY = 9.81
FREE_FALL_THRESHOLD = 0.1 * (NOMINAL_GRAVITY)

MAX_VALID_MAGNETIC_FIELD = 100    # uT
MAX_VALID_MAGNETIC_FIELD_SQ = MAX_VALID_MAGNETIC_FIELD*MAX_VALID_MAGNETIC_FIELD

MIN_VALID_MAGNETIC_FIELD = 10 #uT
MIN_VALID_MAGNETIC_FIELD_SQ = MIN_VALID_MAGNETIC_FIELD*MIN_VALID_MAGNETIC_FIELD

MIN_VALID_CROSS_PRODUCT_MAG = 1.0e-3
MIN_VALID_CROSS_PRODUCT_MAG_SQ = MIN_VALID_CROSS_PRODUCT_MAG*MIN_VALID_CROSS_PRODUCT_MAG

SQRT_3 = 1.732
WVEC_EPS = 1e-4/SQRT_3

FUSION_9AXIS = 0 # use accel gyro mag
FUSION_NOMAG = 1 # use accel gyro (game rotation, gravity)
FUSION_NOGYRO = 2 #use accel mag (geomag rotation)
NUM_FUSION_MODE = 3

ACC = 0
MAG = 1
GYRO = 2

BAD_VALUE = -1
NO_ERROR = 0
