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



#import book_plots as bp
#import mkf_internal
#from code.mkf_internal import plot_track

#from importlib.machinery import SourceFileLoader
#plot_track = SourceFileLoader("plot_track", "/Users/jkuck/rotation3/Kalman-and-Bayesian-Filters-in-Python/code/mkf_internal.py").load_module()
#foo.MyClass()

#import importlib.util
#spec = importlib.util.spec_from_file_location("code.mkf_internal.plot_track", "/Users/jkuck/rotation3/Kalman-and-Bayesian-Filters-in-Python/code/mkf_internal.py")
#plot_track = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(plot_track)
#plot_track.MyClass()

#from importlib.machinery import SourceFileLoader
#read_radar_data = SourceFileLoader("read_radar_data", "/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/utils/read_radar_data_from_txt.py").load_module()

#from read_radar_data_from_txt import read_radar_data

def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]]) # location and velocity
    kf.F = np.array([[1, dt],
                     [0,  1]])    # state transition matrix
    kf.H = np.array([[1, 0]])     # Measurement function
    kf.R *= R                   # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q = Q
    return kf

def plot_track(ps, zs, cov, std_scale=1,
               plot_P=True, y_lim=None, dt=1.,
               xlabel='time', ylabel='position',
               title='Kalman Filter'):
#def plot_track(ps, actual, zs, cov, std_scale=1,
#               plot_P=True, y_lim=None, dt=1.,
#               xlabel='time', ylabel='position',
#               title='Kalman Filter'):

#    with interactive_plot():
    count = len(zs)
    zs = np.asarray(zs)

    cov = np.asarray(cov)
    std = std_scale*np.sqrt(cov[:,0,0])
#        std_top = np.minimum(actual+std, [count + 10])
#        std_btm = np.maximum(actual-std, [-50])
#
#        std_top = actual + std
#        std_btm = actual - std
#
#        bp.plot_track(actual,c='k')

    std_top = np.minimum(zs+std, [count + 10])
    std_btm = np.maximum(zs-std, [-50])

    std_top = zs + std
    std_btm = zs - std

    bp.plot_measurements(range(1, count + 1), zs)
    bp.plot_filter(range(1, count + 1), ps)

    plt.plot(std_top, linestyle=':', color='k', lw=1, alpha=0.4)
    plt.plot(std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
    plt.fill_between(range(len(std_top)), std_top, std_btm,
                     facecolor='yellow', alpha=0.2, interpolate=True)
    plt.legend(loc=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_lim is not None:
        plt.ylim(y_lim)
    else:
        plt.ylim((-50, count + 10))

    plt.xlim((0,count))
    plt.title(title)

    if plot_P:
#        with interactive_plot():
        ax = plt.subplot(121)
        ax.set_title("$\sigma^2_x$ (pos variance)")
        plot_covariance(cov, (0, 0))
        ax = plt.subplot(122)
        ax.set_title("$\sigma^2_\dot{x}$ (vel variance)")
        plot_covariance(cov, (1, 1))

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

def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, data=None,
        count=0, do_plot=True, **kwargs):
    """
    `data` is a 1D numpy array containing the measurements
    """

    # create the Kalman filter
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

    # run the kalman filter and store the results
    xs, cov = [], []
    for z in data:
        kf.predict()
        kf.update(z)
        xs.append(kf.x)
        cov.append(kf.P)

    xs, cov = np.array(xs), np.array(cov)
    if do_plot:
        plot_track(xs[:, 0], data, cov, 
                   dt=dt, **kwargs)
    else:
        cmap = get_cmap(100)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        print xs.shape
        print data.shape
        ax.scatter([i for i in range(0,1216)], xs[:,0], marker = '+', c = cmap(1))
        ax.scatter([i for i in range(0,1216)], data, marker = '*', c = cmap(99))
        plt.show()
    return xs, cov


#####dt = .1
#####x = np.array([0., 0.]) 
#####kf = pos_vel_filter(x, P=500, R=5, Q=0.1, dt=dt)

(lat_pos, lng_pos, lat_vel, lng_vel) = read_radar_data('/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/utils/radar_data.txt')

print type(lat_pos)

if True:
    P = np.diag([500., 49.])
    Ms, Ps = run(count=50, R=1, Q=1, P=P, data=lng_pos[:,54], do_plot=False)
