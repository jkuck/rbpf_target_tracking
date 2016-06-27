#adapted from:
#https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import sys
sys.path.append("/Users/jkuck/rotation3/Kalman-and-Bayesian-Filters-in-Python/code")
sys.path.append("/Users/jkuck/rotation3/Kalman-and-Bayesian-Filters-in-Python")
sys.path.append("/usr/local/lib/python2.7/site-packages/filterpy")
from scipy.stats import multivariate_normal

def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """
    
    kf = KalmanFilter(dim_x=4, dim_z=4)
    kf.x = np.array([x[0], x[1], x[2], x[3]]) # location and velocity
    kf.F = np.array([[1, dt,  0,  0],
                     [0,  1,  0,  0],
                     [0,  0,  1, dt],
                     [0,  0,  0,  1]])    # state transition matrix
    kf.H = np.array([[1,  0,  0,  0],
                     [0,  1,  0,  0],
                     [0,  0,  1,  0],
                     [0,  0,  0,  1]])     # Measurement function


    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P
    if np.isscalar(Q):
#        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        kf.Q *= Q
    else:
        kf.Q = Q

    if np.isscalar(R):
        kf.R *= R                 # measurement uncertainty
    else:
        kf.R[:] = R

    return kf


def m_step(x_predictions, x_posteriors, measurements, H):
    """
    Input:
    - x_predictions: The predicted state variables at every time step
    - x_posteriors: The posterior state variables at every time step
        after performing the Kalman filter update (with the 
        corresponding measurement).
    - measurements: The measurements at every time step.  
    * note: Time step indexing is consistent between x_predictions,
        x_posteriors, and measurements
    -H: Measurement function matrix (Hx = x' where x is in the 
        state space and x' is in the measurement space)

    Return:
    - Q: Maximum likelihood estimate of the process noise covariance 
        matrix.  This is calculated as the covariance matrix of
        (x_posteriors - x_predictions)
    - R: Maximum likelihood estimate of the measurement noise covariance 
        matrix.  This is calculated as the covariance matrix of
        (measurements - x_posteriors)
    * note: We would hope the mean of both (x_posteriors - x_predictions)
        and (measurements - x_posteriors) is zero, but this is not 
        guaranteed.  Think about adjusting motion model and measurement
        function to account for this.
    """

    Q = np.cov((x_posteriors - x_predictions).T, bias = True)
    R = np.cov((measurements.T - np.dot(H,x_posteriors.T)), bias = True)
    return (Q, R)

def calc_LL(x_posteriors, cov, measurements, H, R):
    assert(len(x_posteriors) == len(cov) and len(cov) == len(measurements))
    LL = 0
#    print "len(cov) = ", len(cov)
#    print "cov.shape = ", cov.shape
    for i in range(0, len(cov)):
#        print "shape H = ", H.shape
#        print "shape P[i] = ", cov[i].shape
#        print "shape R = ", R.shape
        S = np.dot(np.dot(H, cov[i]), H.T) + R
        state_posterior_in_meas_space = np.dot(H, x_posteriors[i])
        distribution = multivariate_normal(mean=state_posterior_in_meas_space, cov=S)
        LL += np.log(distribution.pdf(measurements[i]))
    return LL

def run_EM_on_Q_R(x0=(0.,0.,0.,0.), P=500, R=0, Q=0, dt=1.0, data=None):
    log_likelihoods = []
    for i in range(0,5):
        print "iteration ", i, ":"
        print "Q = ", Q
        print "R = ", R
        x_posteriors, x_predictions, cov, H, R = run(x0=x0, P=P, R=R, Q=Q, dt=dt, data=data)
        log_likelihoods.append(calc_LL(x_posteriors, cov, data, H, R))
        Q, R = m_step(x_predictions, x_posteriors, data, H)
    return log_likelihoods

def read_radar_data(file_name):
    f = open(file_name, 'r')

    lat_pos = []
    lng_pos = []
    lat_vel = []
    lng_vel = []

    i = 0
    for line in f:

        if i%64 == 0:
            cur_lat_pos = []
            cur_lng_pos = []
            cur_lat_vel = []
            cur_lng_vel = []
        line = line.strip()
        columns = line.split()
        cur_lat_pos.append(float(columns[0]))
        cur_lng_pos.append(float(columns[1]))
        cur_lat_vel.append(float(columns[2]))
        cur_lng_vel.append(float(columns[3]))
        #add cur time instances (composed of 64 measurements)
        #to lists of all time instances
        if i%64 == 63:
            lat_pos.append(cur_lat_pos)
            lng_pos.append(cur_lng_pos)
            lat_vel.append(cur_lat_vel)
            lng_vel.append(cur_lng_vel)
        i += 1

    f.close()

    return (np.asarray(lat_pos), np.asarray(lng_pos),
            np.asarray(lat_vel), np.asarray(lng_vel))

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def run(x0=(0.,0.,0.,0.), P=500, R=0, Q=0, dt=1.0, data=None):
    """
    `data` is a (number_of_measurements x dimension_of_measurements) 
    numpy array containing the measurements
    """

    # create the Kalman filter
    kf = pos_vel_filter(x=x0, R=R, P=P, Q=Q, dt=.09)  

    # run the kalman filter and store the results
    x_posteriors, x_predictions, cov = [], [], []
    for z in data:
        kf.predict()
        x_predictions.append(kf.x)
        kf.update(z)
        x_posteriors.append(kf.x)
        cov.append(kf.P)

    x_posteriors, x_predictions, cov = np.array(x_posteriors), np.array(x_predictions), np.array(cov)

    print "cov.shape = ", cov.shape

    cmap = get_cmap(100)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_posteriors[:,0], x_posteriors[:, 2], marker = '+', c = cmap(1))
    ax.scatter(data[:,0], data[:,2], marker = '*', c = cmap(99))
    plt.show()

    return x_posteriors, x_predictions, cov, kf.H, kf.R


(lat_pos, lng_pos, lat_vel, lng_vel) = read_radar_data('/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/utils/radar_data.txt')

print type(lat_pos)

if False:
    P = np.diag([10., 5., 10., 5.])
    Ms, Ps = run(R=1, Q=1, P=P, data=np.concatenate((np.expand_dims(lat_pos[:,54], axis=1), np.expand_dims(lng_pos[:,54], axis=1)), axis=1))

if True:
    P = np.diag([10., 5., 10., 5.])
    log_likelihoods = run_EM_on_Q_R(R=1000, Q=.00010, P=P, data=np.concatenate((np.expand_dims(lat_pos[:,54], axis=1),
                                                                                np.expand_dims(lat_vel[:,54], axis=1),
                                                                                np.expand_dims(lng_pos[:,54], axis=1),
                                                                                np.expand_dims(lng_vel[:,54], axis=1) - 35.3), axis=1))
    print log_likelihoods
    cmap = get_cmap(100)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter([i for i in range(0, len(log_likelihoods))], log_likelihoods, marker = '+', c = cmap(1))
    plt.show()


