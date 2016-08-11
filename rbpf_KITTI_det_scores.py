import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import stratified_resample
import filterpy
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.special import gdtrc
import random
import copy 
import math
from numpy.linalg import inv
import pickle
import sys
sys.path.insert(0, "/Users/jkuck/rotation3/clearmetrics")
import clearmetrics
sys.path.insert(0, "./KITTI_helpers")
from learn_params1 import get_clutter_probabilities_score_range_wrapper
from learn_params1 import get_meas_target_set
from learn_params1 import get_meas_target_sets_lsvm_and_regionlets
from jdk_helper_evaluate_results import eval_results

from multiple_meas_per_time_assoc_priors import HiddenState
from proposal2_helper import possible_measurement_target_associations
from proposal2_helper import memoized_birth_clutter_prior
from proposal2_helper import sample_birth_clutter_counts
from proposal2_helper import sample_target_deaths_proposal2

import cProfile
import time

#MEASURMENT_FILENAME = "KITTI_helpers/KITTI_measurements_car_lsvm_min_score_0.0.pickle"
#MEASURMENT_FILENAME = "KITTI_helpers/KITTI_measurements_car_regionlets_min_score_2.0.pickle"

#run on these sequences
SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [i for i in range(21)]
#eval_results('/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/rbpf_KITTI_results', SEQUENCES_TO_PROCESS)
#sleep(5)
#RBPF algorithmic paramters
N_PARTICLES = 100 #number of particles used in the particle filter
RESAMPLE_RATIO = 2.0 #resample when get_eff_num_particles < N_PARTICLES/RESAMPLE_RATIO

DEBUG = False

USE_PYTHON_GAUSSIAN = False #if False bug, using R_default instead of S, check USE_CONSTANT_R
USE_PROPOSAL_DISTRIBUTION_3 = True #sample measurement associations sequentially, then unassociated target deaths

#default time between succesive measurement time instances (in seconds)
default_time_step = .1 

#SCORE_INTERVALS = [i/2.0 for i in range(0, 8)]
USE_CONSTANT_R = False
#For testing why score interval for R are slow
CACHED_LIKELIHOODS = 0
NOT_CACHED_LIKELIHOODS = 0

REGIONLETS_SCORE_INTERVALS = [i for i in range(2, 20)]
LSVM_SCORE_INTERVALS = [i/2.0 for i in range(0, 8)]
SCORE_INTERVALS = [REGIONLETS_SCORE_INTERVALS, LSVM_SCORE_INTERVALS]

#(measurementTargetSetsBySequence, target_emission_probs, clutter_probabilities, birth_probabilities,\
#	meas_noise_covs) = get_meas_target_set(SCORE_INTERVALS, det_method = "regionlets", obj_class = "car", doctor_clutter_probs = True)
(measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
	MEAS_NOISE_COVS) = get_meas_target_sets_lsvm_and_regionlets(REGIONLETS_SCORE_INTERVALS, LSVM_SCORE_INTERVALS, \
    obj_class = "car", doctor_clutter_probs = True)
#from learn_params
#BIRTH_COUNT_PRIOR = [0.9371030016191306, 0.0528085689376012, 0.007223813675426578, 0.0016191306513887158, 0.000747291069871715, 0.00012454851164528583, 0, 0.00012454851164528583, 0.00012454851164528583, 0, 0, 0, 0, 0.00012454851164528583]
#from learn_params1, not counting 'ignored' ground truth
BIRTH_COUNT_PRIOR = [0.95640802092415, 0.039357329679910326, 0.0027400672561962883, 0.0008718395815170009, 0.00012454851164528583, 0.00012454851164528583, 0, 0.00024909702329057166, 0, 0, 0.00012454851164528583]

def get_score_index(score_intervals, score):
	"""
	Inputs:
	- score_intervals: a list specifying detection score ranges for which parameters have been specified
	- score: the score of a detection

	Output:
	- index: output the 0 indexed score interval this score falls into
	"""

	index = 0
	for i in range(1, len(score_intervals)):
		if(score > score_intervals[i]):
			index += 1
		else:
			break
	assert(score > score_intervals[index])
	if(index < len(score_intervals) - 1):
		assert(score < score_intervals[index+1])
	return index


#regionlet detection with score > 2.0:
#from learn_params
#P_TARGET_EMISSION = 0.813482 
#from learn_params1, not counting 'ignored' ground truth
P_TARGET_EMISSION = 0.813358070501
#death probabiltiies, for sampling AFTER associations, conditioned on un-association
#DEATH_PROBABILITIES = [-99, 0.1558803061934586, 0.24179829890643986, 0.1600831600831601, 0.10416666666666667, 0.08835341365461848, 0.04081632653061224, 0.06832298136645963, 0.06201550387596899, 0.04716981132075472, 0.056818181818181816, 0.013333333333333334, 0.028985507246376812, 0.03278688524590164, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0625, 0.03571428571428571, 0.0, 0.0, 0.043478260869565216, 0.0, 0.05555555555555555, 0.0, 0.0625, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425, 0.2222222222222222, 0.35714285714285715, 0.2222222222222222, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035, 0.02247191011235955, 0.04081632653061224, 0.05, 0.05, 0.036585365853658534, 0.013888888888888888, 0.030303030303030304, 0.03389830508474576, 0.0, 0.0, 0.0, 0.05128205128205128, 0.0, 0.06451612903225806, 0.037037037037037035, 0.0, 0.0, 0.045454545454545456, 0.0, 0.05555555555555555, 0.0, 0.0625, 0.07142857142857142, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


BORDER_DEATH_PROBABILITIES = [-99, 0.3116591928251121, 0.5483870967741935, 0.5833333333333334, 0.8571428571428571, 1.0]
NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.001880843060242297, 0.026442307692307692, 0.04918032786885246, 0.06818181818181818, 0.008]

#BORDER_DEATH_PROBABILITIES = [-99, 0.3290203327171904, 0.5868263473053892, 0.48148148148148145, 0.4375, 0.42424242424242425]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.05133928571428571, 0.006134969325153374, 0.03468208092485549, 0.025735294117647058, 0.037037037037037035]

#BORDER_DEATH_PROBABILITIES = [-99, 0.8, 0.5, 0.3, 0.4, 0.8]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.07, 0.025, 0.03, 0.03, 0.006]

#BORDER_DEATH_PROBABILITIES = [-99, 0.9430523917995444, 0.6785714285714286, 0.4444444444444444, 0.5, 1.0]
#NOT_BORDER_DEATH_PROBABILITIES = [-99, 0.08235294117647059, 0.02284263959390863, 0.04150943396226415, 0.041237113402061855, 0.00684931506849315]
#from learn_params
#CLUTTER_COUNT_PRIOR = [0.7860256569933989, 0.17523975588491716 - .001, 0.031635321957902605, 0.004857391954166148, 0.0016191306513887158, 0.0003736455349358575, 0.00024909702329057166, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0, .001/20.0]
#from learn_params1, not counting 'ignored' ground truth
CLUTTER_COUNT_PRIOR = [0.5424333167268651, 0.3045211109727239, 0.11010088429443268, 0.0298916427948686, 0.008718395815170008, 0.003113712791132146, 0.0009963880931622867, 0.00012454851164528583, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06, 5e-06]

p_clutter_likelihood = 1.0/float(1242*375)
#p_birth_likelihood = 0.035
p_birth_likelihood = 1.0/float(1242*375)


#Kalman filter defaults
#Think about doing this in a more principled way!!!
P_default = np.array([[57.54277774, 0, 			 0, 0],
 					  [0,          10, 			 0, 0],
 					  [0, 			0, 17.86392672, 0],
 					  [0, 			0, 			 0, 3]])
#P_default = np.array([[40.64558317, 0, 			 0, 0],
# 					  [0,          10, 			 0, 0],
# 					  [0, 			0, 5.56278505, 0],
# 					  [0, 			0, 			 0, 3]])

#regionlet detection with score > 2.0:
#from learn_params
R_default = np.array([[  5.60121574e+01,  -3.60666228e-02],
 					  [ -3.60666228e-02,   1.64772050e+01]])
#from learn_params1, not counting 'ignored' ground truth
#R_default = np.array([[ 40.64558317,   0.14036472],
# 					  [  0.14036472,   5.56278505]])


#learned from all GT
#Q_default = np.array([[ 84.30812679,  84.21851631,  -4.01491901,  -8.5737873 ],
# 					  [ 84.21851631,  84.22312789,  -3.56066467,  -8.07744876],
# 					  [ -4.01491901,  -3.56066467,   4.59923143,   5.19622064],
# 					  [ -8.5737873 ,  -8.07744876,   5.19622064,   6.10733628]])
#also learned from all GT
Q_default = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535],
 					  [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621],
 					  [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108],
 					  [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314]])

#measurement function matrix
H = np.array([[1.0,  0.0, 0.0, 0.0],
              [0.0,  0.0, 1.0, 0.0]])	

USE_LEARNED_DEATH_PROBABILITIES = True

#Gamma distribution parameters for calculating target death probabilities
alpha_death = 2.0
beta_death = 1.0
theta_death = 1.0/beta_death

print Q_default
print R_default

#for only displaying targets older than this
min_target_age = .2

#state parameters, during data generation uniformly sample new targets from range:
min_pos = -5.0
max_pos = 5.0
min_vel = -1.0
max_vel = 1.0

#The maximum allowed distance for a ground truth target and estimated target
#to be associated with each other when calculating MOTA and MOTP
MAX_ASSOCIATION_DIST = 1

CAMERA_PIXEL_WIDTH = 1242
CAMERA_PIXEL_HEIGHT = 375

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color




class Target:
	def __init__(self, cur_time, id_, measurement = None, width=-1, height=-1):
		if measurement is None: #for data generation
			position = np.random.uniform(min_pos,max_pos)
			velocity = np.random.uniform(min_vel,max_vel)
			self.x = np.array([[position], [velocity]])
			self.P = P_default
		else:
			self.x = np.array([[measurement[0]], [0], [measurement[1]], [0]])
			self.P = P_default

		self.width = width
		self.height = height

		assert(self.x.shape == (4, 1))
		self.birth_time = cur_time
		#Time of the last measurement data association with this target
		self.last_measurement_association = cur_time
		self.id_ = id_ #named id_ to avoid clash with built in id
		self.death_prob = -1 #calculate at every time instance

		self.all_states = [(self.x, self.width, self.height)]
		self.all_time_stamps = [cur_time]

		self.measurements = []
		self.measurement_time_stamps = []

		#if target's predicted location is offscreen, set to True and then kill
		self.offscreen = False

	def near_border(self):
		near_border = False
		x1 = self.x[0][0] - self.width/2.0
		x2 = self.x[0][0] + self.width/2.0
		y1 = self.x[2][0] - self.height/2.0
		y2 = self.x[2][0] + self.height/2.0
		if(x1 < 10 or x2 > (CAMERA_PIXEL_WIDTH - 15) or y1 < 10 or y2 > (CAMERA_PIXEL_HEIGHT - 15)):
			near_border = True
		return near_border


	def kf_update(self, measurement, width, height, cur_time, meas_noise_cov):
		""" Perform Kalman filter update step and replace predicted position for the current time step
		with the updated position in self.all_states
		Input:
		- measurement: the measurement (numpy array)
		- cur_time: time when the measurement was taken (float)
!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
		"""
		reformat_meas = np.array([[measurement[0]],
								  [measurement[1]]])
		assert(self.x.shape == (4, 1))
		if USE_CONSTANT_R:
			S = np.dot(np.dot(H, self.P), H.T) + R_default
		else:
			S = np.dot(np.dot(H, self.P), H.T) + meas_noise_cov
		K = np.dot(np.dot(self.P, H.T), inv(S))
		residual = reformat_meas - np.dot(H, self.x)
		updated_x = self.x + np.dot(K, residual)
	#	updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
		updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
		self.x = updated_x
		self.P = updated_P
		self.width = width
		self.height = height
		assert(self.all_time_stamps[-1] == cur_time and self.all_time_stamps[-2] != cur_time)
		assert(self.x.shape == (4, 1)), (self.x.shape, np.dot(K, residual).shape)

		self.all_states[-1] = (self.x, self.width, self.height)

	def kf_predict(self, dt, cur_time):
		"""
		Run kalman filter prediction on this target
		Inputs:
			-dt: time step to run prediction on
			-cur_time: the time the prediction is made for
		"""
		assert(self.all_time_stamps[-1] == (cur_time - dt))
		F = np.array([[1.0,  dt, 0.0, 0.0],
		      		  [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0,  dt],
                      [0.0, 0.0, 0.0, 1.0]])
		x_predict = np.dot(F, self.x)
		P_predict = np.dot(np.dot(F, self.P), F.T) + Q_default
		self.x = x_predict
		self.P = P_predict
		self.all_states.append((self.x, self.width, self.height))
		self.all_time_stamps.append(cur_time)

		if(self.x[0][0]<0 or self.x[0][0]>=CAMERA_PIXEL_WIDTH or \
		   self.x[2][0]<0 or self.x[2][0]>=CAMERA_PIXEL_HEIGHT):
#			print '!'*40, "TARGET IS OFFSCREEN", '!'*40
			self.offscreen = True

		assert(self.x.shape == (4, 1))



###################	def target_death_prob(self, cur_time, prev_time):
###################		""" Calculate the target death probability if this was the only target.
###################		Actual target death probability will be (return_val/number_of_targets)
###################		because we limit ourselves to killing a max of one target per measurement.
###################
###################		Input:
###################		- cur_time: The current measurement time (float)
###################		- prev_time: The previous time step when a measurement was received (float)
###################
###################		Return:
###################		- death_prob: Probability of target death if this is the only target (float)
###################		"""
###################
###################		#scipy.special.gdtrc(b, a, x) calculates 
###################		#integral(gamma_dist(k = a, theta = b))from x to infinity
###################		last_assoc = self.last_measurement_association
###################
###################		#I think this is correct
###################		death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
###################				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
###################		death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
###################		return death_prob
###################
####################		#this is used in paper's code
####################		time_step = cur_time - prev_time
####################	
####################		death_prob = gdtrc(theta_death, alpha_death, cur_time - last_assoc) \
####################				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc + time_step)
####################		death_prob /= gdtrc(theta_death, alpha_death, cur_time - last_assoc)
####################		return death_prob
	def target_death_prob(self, cur_time, prev_time):
		""" Calculate the target death probability if this was the only target.
		Actual target death probability will be (return_val/number_of_targets)
		because we limit ourselves to killing a max of one target per measurement.

		Input:
		- cur_time: The current measurement time (float)
		- prev_time: The previous time step when a measurement was received (float)

		Return:
		- death_prob: Probability of target death if this is the only target (float)
		"""

##################		#scipy.special.gdtrc(b, a, x) calculates 
##################		#integral(gamma_dist(k = a, theta = b))from x to infinity
##################		last_assoc = self.last_measurement_association
##################
##################		#I think this is correct
##################		death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
##################				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
##################		death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
##################		return death_prob
		if(self.offscreen == True):
			cur_death_prob = 1.0
		else:
			frames_since_last_assoc = int(round((cur_time - self.last_measurement_association)/default_time_step))
			assert(abs(float(frames_since_last_assoc) - (cur_time - self.last_measurement_association)/default_time_step) < .00000001)
			if(self.near_border()):
				if frames_since_last_assoc < len(BORDER_DEATH_PROBABILITIES):
					cur_death_prob = BORDER_DEATH_PROBABILITIES[frames_since_last_assoc]
				else:
					cur_death_prob = BORDER_DEATH_PROBABILITIES[-1]
#					cur_death_prob = 1.0
			else:
				if frames_since_last_assoc < len(NOT_BORDER_DEATH_PROBABILITIES):
					cur_death_prob = NOT_BORDER_DEATH_PROBABILITIES[frames_since_last_assoc]
				else:
					cur_death_prob = NOT_BORDER_DEATH_PROBABILITIES[-1]
#					cur_death_prob = 1.0

		assert(cur_death_prob >= 0.0 and cur_death_prob <= 1.0), cur_death_prob
		return cur_death_prob

class Measurement:
    #a collection of measurements at a single time instance
    def __init__(self, time = -1):
        #self.val is a list of numpy arrays of measurement x, y locations
        self.val = []
        #list of widths of each bounding box
        self.widths = []
        #list of widths of each bounding box        
        self.heights = []
        #list of scores for each individual measurement
        self.scores = []
        self.time = time

class TargetSet:
	"""
	Contains ground truth states for all targets.  Also contains all generated measurements.
	"""

	def __init__(self):
		self.living_targets = []
		self.all_targets = [] #alive and dead targets

		self.living_count = 0 #number of living targets
		self.total_count = 0 #number of living targets plus number of dead targets
		self.measurements = [] #generated measurements for a generative TargetSet 

	def create_new_target(self, measurement, width, height, cur_time):
		new_target = Target(cur_time, self.total_count, np.squeeze(measurement), width, height)
		self.living_targets.append(new_target)
		self.all_targets.append(new_target)
		self.living_count += 1
		self.total_count += 1
		assert(len(self.living_targets) == self.living_count and len(self.all_targets) == self.total_count)


	def kill_target(self, living_target_index):
		"""
		Kill target self.living_targets[living_target_index], note that living_target_index
		may not be the target's id_ (or index in all_targets)
		"""

		#kf predict was run for this time instance, but the target actually died, so remove the predicted state
		del self.living_targets[living_target_index].all_states[-1]
		del self.living_targets[living_target_index].all_time_stamps[-1]

		del self.living_targets[living_target_index]

		self.living_count -= 1
		assert(len(self.living_targets) == self.living_count and len(self.all_targets) == self.total_count)

	def plot_all_target_locations(self, title):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		for i in range(self.total_count):
			life = len(self.all_targets[i].all_states) #length of current targets life 
			locations_1D =  [self.all_targets[i].all_states[j][0] for j in range(life)]
			ax.plot(self.all_targets[i].all_time_stamps, locations_1D,
					'-o', label='Target %d' % i)

		legend = ax.legend(loc='lower left', shadow=True)
		plt.title('%s, unique targets = %d, #targets alive = %d' % \
			(title, self.total_count, self.living_count)) # subplot 211 title

	def plot_generated_measurements(self):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		time_stamps = [self.measurements[i].time for i in range(len(self.measurements))
												 for j in range(len(self.measurements[i].val))]
		locations = [self.measurements[i].val[j][0] for i in range(len(self.measurements))
													for j in range(len(self.measurements[i].val))]
		ax.plot(time_stamps, locations,'o')
		plt.title('Generated Measurements') 


	def write_targets_to_KITTI_format(self, num_frames, filename):
		f = open(filename, "w")
		for frame_idx in range(num_frames):
			timestamp = frame_idx*default_time_step
			for target in self.all_targets:
				if timestamp in target.all_time_stamps:
					x_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][0][0]
					y_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][2][0]
					width = target.all_states[target.all_time_stamps.index(timestamp)][1]
					height = target.all_states[target.all_time_stamps.index(timestamp)][2]

					left = x_pos - width/2.0
					top = y_pos - height/2.0
					right = x_pos + width/2.0
					bottom = y_pos + height/2.0		 
					f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
						(frame_idx, target.id_, left, top, right, bottom))
#					left = target.x[0][0] - target.width/2
#					top = target.x[2][0] - target.height/2
#					right = target.x[0][0] + target.width/2
#					bottom = target.x[2][0] + target.height/2		 
#					f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
#						(frame_idx, target.id_, left, top, right, bottom))
		f.close()


class Particle:
	def __init__(self, id_):
		#Targets tracked by this particle
		self.targets = TargetSet()

		self.importance_weight = 1.0/N_PARTICLES

		#cache for memoizing association likelihood computation
		self.assoc_likelihood_cache = {}

		self.id_ = id_

		#for debugging
		self.c_debug = -1
		self.imprt_re_weight_debug = -1
		self.pi_birth_debug = -1
		self.pi_clutter_debug = -1
		self.pi_targets_debug = []

	def child_copy(self, id_):
		child_particle = Particle(id_)
		child_particle.importance_weight = self.importance_weight

	def create_new_target(self, measurement, width, height, cur_time):
		self.targets.create_new_target(measurement, width, height, cur_time)

	def update_target_death_probabilities(self, cur_time, prev_time):
		for target in self.targets.living_targets:
			target.death_prob = target.target_death_prob(cur_time, prev_time)

	def sample_target_deaths(self):
		"""

		Implemented to possibly kill multiple targets at once, seems
		reasonbale but CHECK TECHNICAL DETAILS!!

		death_prob for every target should have already been calculated!!

		Input:
		- cur_time: The current measurement time (float)
		- prev_time: The previous time step when a measurement was received (float)

		"""
		original_num_targets = self.targets.living_count
		num_targets_killed = 0
		indices_to_kill = []
		for (index, cur_target) in enumerate(self.targets.living_targets):
			death_prob = cur_target.death_prob
			assert(death_prob < 1.0 and death_prob > 0.0)
			if (random.random() < death_prob):
				indices_to_kill.append(index)
				num_targets_killed += 1

		#important to delete largest index first to preserve values of the remaining indices
		for index in reversed(indices_to_kill):
			self.targets.kill_target(index)

		assert(self.targets.living_count == (original_num_targets - num_targets_killed))
		#print "targets killed = ", num_targets_killed




	def sample_data_assoc_and_death_mult_meas_per_time_proposal_distr_1(self, measurement_lists, \
		cur_time, measurement_scores):
		"""
		Input:
		- measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
			time instance from the ith measurement source (i.e. different object detection algorithms
			or different sensors)
		- measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
			measurement_list[i]

		Output:
		- measurement_associations: A list where measurement_associations[i] is a list of association values
			for each measurements in measurement_lists[i].  Association values correspond to:
			measurement_associations[i][j] = -1 -> measurement is clutter
			measurement_associations[i][j] = self.targets.living_count -> measurement is a new target
			measurement_associations[i][j] in range [0, self.targets.living_count-1] -> measurement is of
				particle.targets.living_targets[measurement_associations[i][j]]

		- imprt_re_weight: After processing this measurement the particle's
			importance weight will be:
			new_importance_weight = old_importance_weight * imprt_re_weight
		- targets_to_kill: a list containing the indices of targets that should be killed, beginning
			with the smallest index in increasing order, e.g. [0, 4, 6, 33]
		"""

		#get death probabilities for each target in a numpy array
		num_targs = self.targets.living_count
		p_target_deaths = []
		for target in self.targets.living_targets:
			p_target_deaths.append(target.death_prob)
			assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)


		(targets_to_kill, measurement_associations, proposal_probability, unassociated_target_death_probs) = \
			self.sample_proposal_distr3(measurement_lists, self.targets.living_count, p_target_deaths, \
										cur_time, measurement_scores)


		living_target_indices = []
		for i in range(self.targets.living_count):
			if(not i in targets_to_kill):
				living_target_indices.append(i)

#		exact_probability = self.get_exact_prob_hidden_and_data(measurement_list, living_target_indices, self.targets.living_count, 
#												 measurement_associations, p_target_deaths)
		exact_probability = 1.0
		for meas_source_index in range(len(measurement_lists)):
			cur_assoc_prob = self.get_exact_prob_hidden_and_data(meas_source_index, measurement_lists[meas_source_index], \
				living_target_indices, self.targets.living_count, measurement_associations[meas_source_index],\
				unassociated_target_death_probs, measurement_scores[meas_source_index], SCORE_INTERVALS[meas_source_index])
			exact_probability *= cur_assoc_prob

		exact_death_prob = self.calc_death_prior(living_target_indices, p_target_deaths)
		exact_probability *= exact_death_prob

		assert(num_targs == self.targets.living_count)
		#double check targets_to_kill is sorted
		assert(all([targets_to_kill[i] <= targets_to_kill[i+1] for i in xrange(len(targets_to_kill)-1)]))

		imprt_re_weight = exact_probability/proposal_probability

		assert(imprt_re_weight != 0.0), (exact_probability, proposal_probability)

		return (measurement_associations, targets_to_kill, imprt_re_weight)


	def associate_measurements_proposal_distr3(self, meas_source_index, measurement_list, total_target_count, \
		p_target_deaths, measurement_scores):

		"""
		Try sampling associations with each measurement sequentially
		Input:
		- measurement_list: a list of all measurements from the current time instance
		- total_target_count: the number of living targets on the previous time instace
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Output:
		- list_of_measurement_associations: list of associations for each measurement
		- proposal_probability: proposal probability of the sampled deaths and assocations
			
		"""
		list_of_measurement_associations = []
		proposal_probability = 1.0

		#sample measurement associations
		birth_count = 0
		clutter_count = 0
		remaining_meas_count = len(measurement_list)
		for (index, cur_meas) in enumerate(measurement_list):
			score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[index])
			#create proposal distribution for the current measurement
			#compute target association proposal probabilities
			proposal_distribution_list = []
			for target_index in range(total_target_count):
				cur_target_likelihood = self.memoized_assoc_likelihood(cur_meas, meas_source_index, target_index, MEAS_NOISE_COVS[meas_source_index][score_index], score_index)
				targ_likelihoods_summed_over_meas = 0.0
				for meas_index in range(len(measurement_list)):
					temp_score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[meas_index]) #score_index for the meas_index in this loop
					targ_likelihoods_summed_over_meas += self.memoized_assoc_likelihood(measurement_list[meas_index], meas_source_index, target_index,  MEAS_NOISE_COVS[meas_source_index][temp_score_index], temp_score_index)
				if((targ_likelihoods_summed_over_meas != 0.0) and (not target_index in list_of_measurement_associations)\
					and p_target_deaths[target_index] < 1.0):
					cur_target_prior = TARGET_EMISSION_PROBS[meas_source_index][score_index]*cur_target_likelihood \
									  /targ_likelihoods_summed_over_meas
#					cur_target_prior = P_TARGET_EMISSION*cur_target_likelihood \
#									  /targ_likelihoods_summed_over_meas
				else:
					cur_target_prior = 0.0

				proposal_distribution_list.append(cur_target_likelihood*cur_target_prior)

			#compute birth association proposal probability
			cur_birth_prior = 0.0
			for i in range(birth_count+1, min(len(BIRTH_PROBABILITIES[meas_source_index][score_index]), remaining_meas_count + birth_count + 1)):
				cur_birth_prior += BIRTH_PROBABILITIES[meas_source_index][score_index][i]*(i - birth_count)/remaining_meas_count 
			proposal_distribution_list.append(cur_birth_prior*p_birth_likelihood)

			#compute clutter association proposal probability
			cur_clutter_prior = 0.0
			for i in range(clutter_count+1, min(len(CLUTTER_PROBABILITIES[meas_source_index][score_index]), remaining_meas_count + clutter_count + 1)):
				cur_clutter_prior += CLUTTER_PROBABILITIES[meas_source_index][score_index][i]*(i - clutter_count)/remaining_meas_count 
			proposal_distribution_list.append(cur_clutter_prior*p_clutter_likelihood)

			#normalize the proposal distribution
			proposal_distribution = np.asarray(proposal_distribution_list)
			assert(np.sum(proposal_distribution) != 0.0), (len(proposal_distribution), proposal_distribution, birth_count, clutter_count, len(measurement_list), total_target_count)

			proposal_distribution /= float(np.sum(proposal_distribution))
			assert(len(proposal_distribution) == total_target_count+2)

			sampled_assoc_idx = np.random.choice(len(proposal_distribution),
													p=proposal_distribution)
			if(sampled_assoc_idx <= total_target_count): #target or birth association
				list_of_measurement_associations.append(sampled_assoc_idx)
				if(sampled_assoc_idx == total_target_count):
					birth_count += 1
			else: #clutter association
				assert(sampled_assoc_idx == total_target_count+1)
				list_of_measurement_associations.append(-1)
				clutter_count += 1
			proposal_probability *= proposal_distribution[sampled_assoc_idx]

			remaining_meas_count -= 1
		assert(remaining_meas_count == 0)
		return(list_of_measurement_associations, proposal_probability)

	def sample_proposal_distr3(self, measurement_lists, total_target_count, 
							   p_target_deaths, cur_time, measurement_scores):
		"""
		Try sampling associations with each measurement sequentially
		Input:
		- measurement_lists: type list, measurement_lists[i] is a list of all measurements from the current
			time instance from the ith measurement source (i.e. different object detection algorithms
			or different sensors)
		- measurement_scores: type list, measurement_scores[i] is a list containing scores for every measurement in
			measurement_list[i]
		- total_target_count: the number of living targets on the previous time instace
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Output:
		- targets_to_kill: a list of targets that have been sampled to die (not killed yet)
		- measurement_associations: type list, measurement_associations[i] is a list of associations for  
			the measurements in measurement_lists[i]
		- proposal_probability: proposal probability of the sampled deaths and assocations
			
		"""
		assert(len(measurement_lists) == len(measurement_scores))
		measurement_associations = []
		proposal_probability = 1.0
		for meas_source_index in range(len(measurement_lists)):
			(cur_associations, cur_proposal_prob) = self.associate_measurements_proposal_distr3\
				(meas_source_index, measurement_lists[meas_source_index], total_target_count, \
				 p_target_deaths, measurement_scores[meas_source_index])
			measurement_associations.append(cur_associations)
			proposal_probability *= cur_proposal_prob

		assert(len(measurement_associations) == len(measurement_lists))

############################################################################################################
		#sample target deaths from unassociated targets
		unassociated_targets = []
		unassociated_target_death_probs = []

		for i in range(total_target_count):
			target_unassociated = True
			for meas_source_index in range(len(measurement_associations)):
				if (i in measurement_associations[meas_source_index]):
					target_unassociated = False
			if target_unassociated:
				unassociated_targets.append(i)
				unassociated_target_death_probs.append(p_target_deaths[i])
			else:
				unassociated_target_death_probs.append(0.0)


		if USE_LEARNED_DEATH_PROBABILITIES:
			(targets_to_kill, death_probability) =  \
				self.sample_target_deaths_proposal3(unassociated_targets, cur_time)
		else:
			(targets_to_kill, death_probability) =  \
				self.sample_target_deaths_proposal2(unassociated_targets, cur_time)

		#probability of sampling all associations
		proposal_probability *= death_probability
		assert(proposal_probability != 0.0)

		#debug
		for meas_source_index in range(len(measurement_associations)):
			for i in range(total_target_count):
				assert(measurement_associations[meas_source_index].count(i) == 0 or \
					   measurement_associations[meas_source_index].count(i) == 1), (measurement_associations[meas_source_index],  measurement_list, total_target_count, p_target_deaths)
		#done debug

		return (targets_to_kill, measurement_associations, proposal_probability, unassociated_target_death_probs)


	def sample_target_deaths_proposal3(self, unassociated_targets, cur_time):
		"""
		Sample target deaths, given they have not been associated with a measurement, using probabilities
		learned from data.
		Also kill all targets that are offscreen.

		Inputs:
		- unassociated_targets: a list of target indices that have not been associated with a measurement

		Output:
		- targets_to_kill: a list of targets that have been sampled to die (not killed yet)
		- probability_of_deaths: the probability of the sampled deaths
		"""
		targets_to_kill = []
		probability_of_deaths = 1.0

		for target_idx in range(len(self.targets.living_targets)):
			#kill offscreen targets with probability 1.0
			if(self.targets.living_targets[target_idx].offscreen == True):
				targets_to_kill.append(target_idx)
			elif(target_idx in unassociated_targets):
				cur_death_prob = self.targets.living_targets[target_idx].death_prob
				if(random.random() < cur_death_prob):
					targets_to_kill.append(target_idx)
					probability_of_deaths *= cur_death_prob
				else:
					probability_of_deaths *= (1 - cur_death_prob)
		return (targets_to_kill, probability_of_deaths)

	def calc_death_prior(self, living_target_indices, p_target_deaths):
		death_prior = 1.0
		for (cur_target_index, cur_target_death_prob) in enumerate(p_target_deaths):
			if cur_target_index in living_target_indices:
				death_prior *= (1.0 - cur_target_death_prob)
				assert((1.0 - cur_target_death_prob) != 0.0), cur_target_death_prob
			else:
				death_prior *= cur_target_death_prob
				assert((cur_target_death_prob) != 0.0), cur_target_death_prob

		return death_prior

	def get_prior(self, living_target_indices, total_target_count, number_measurements, 
				 measurement_associations, p_target_deaths, target_emission_probs, 
				 birth_count_priors, clutter_count_priors, measurement_scores, score_intervals):
		"""
DON"T THINK THIS BELONGS IN PARTICLE, OR PARAMETERS COULD BE CLEANED UP
		REDOCUMENT

		Input: 
		- living_target_indices: a list of indices of targets from last time instance that are still alive
		- total_target_count: the number of living targets on the previous time instace
		- number_measurements: the number of measurements on this time instance
		- measurement_associations: a list of association values for each measurement. Each association has the value
			of a living target index (index from last time instance), target birth (total_target_count), 
			or clutter (-1)
		-p_target_deaths: a list of length len(number_targets) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance
		-p_target_emission: the probability that a target will emit a measurement on a 
			time instance (the same for all targets and time instances)
		-birth_count_prior: a probability distribution, specified as a list, such that
			birth_count_prior[i] = the probability of i births during any time instance
		-clutter_count_prior: a probability distribution, specified as a list, such that
			clutter_count_prior[i] = the probability of i clutter measurements during 
			any time instance
		"""

		def nCr(n,r):
		    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)

		def count_meas_orderings(M, T, b, c):
			"""
			We define target observation priors in terms of whether each target was observed and it
			is irrelevant which measurement the target is associated with.  Likewise, birth count priors
			and clutter count priors are defined in terms of total counts, not which specific measurements
			are associated with clutter and births.  This function counts the number of possible 
			measurement-association assignments given we have already chosen which targets are observed, 
			how many births occur, and how many clutter measurements are present.  The prior probability of
			observing T specific targets, b births, and c clutter observations given M measurements should
			be divided by the returned value to split the prior probability between possibilities.

			[
			*OLD EXPLANATION BELOW*:
			We view the the ordering of measurements on any time instance as arbitrary.  This
			function counts the number of possible measurement orderings given we have already
			chosen which targets are observed, how many births occur, and how many clutter 
			measurements are present.
			]
			
			Inputs:
			- M: the number of measurements
			- T: the number of observed targets
			- b: the number of birth associations
			- c: the number of clutter associations

			This must be true: M = T+b+c

			Output:
			- combinations: the number of measurement orderings as a float. The value is:
				combinations = nCr(M, T)*math.factorial(T)*nCr(M-T, b)

			"""
			assert(M == T + b + c)
			combinations = nCr(M, T)*math.factorial(T)*nCr(M-T, b)
			return float(combinations)


		assert(len(measurement_associations) == number_measurements)
		#numnber of targets from the last time instance that are still alive
		living_target_count = len(living_target_indices)
		#numnber of targets from the last time instance that died
		dead_target_count = total_target_count - living_target_count

		#count the number of unique target associations
		unique_assoc = set(measurement_associations)
		if(total_target_count in unique_assoc):
			unique_assoc.remove(total_target_count)
		if((-1) in unique_assoc):
			unique_assoc.remove((-1))

		#the number of targets we observed on this time instance
		observed_target_count = len(unique_assoc)

		#the number of target measurements by measurement score
		meas_counts_by_score = [0 for i in range(len(score_intervals))]
		for i in range(len(measurement_associations)):
			if measurement_associations[i] != -1 and measurement_associations[i] != total_target_count:
				index = get_score_index(score_intervals, measurement_scores[i])
				meas_counts_by_score[index] += 1

		#the number of targets we don't observe on this time instance
		#but are still alive on this time instance
		unobserved_target_count = living_target_count - observed_target_count
		#the number of new targets born on this time instance
		birth_count = measurement_associations.count(total_target_count)
		birth_counts_by_score = [0 for i in range(len(score_intervals))]
		for i in range(len(measurement_associations)):
			if measurement_associations[i] == total_target_count:
				index = get_score_index(score_intervals, measurement_scores[i])
				birth_counts_by_score[index] += 1
		#the number of clutter measurements on this time instance
		clutter_count = measurement_associations.count(-1)
		clutter_counts_by_score = [0 for i in range(len(score_intervals))]
		for i in range(len(measurement_associations)):
			if measurement_associations[i] == -1:
				index = get_score_index(score_intervals, measurement_scores[i])
				clutter_counts_by_score[index] += 1

		assert(observed_target_count + birth_count + clutter_count == number_measurements),\
			(number_measurements, observed_target_count, birth_count, clutter_count, \
			total_target_count, measurement_associations)

#		assert(len(p_target_deaths) == total_target_count)
		death_prior = self.calc_death_prior(living_target_indices, p_target_deaths)

		#the prior probability of this number of measurements with these associations
		#given these target deaths
		for i in range(len(score_intervals)):

			assert(0 <= clutter_counts_by_score[i] and clutter_counts_by_score[i] < len(clutter_count_priors[i])), clutter_counts_by_score[i]
			assert(0 <= birth_counts_by_score[i] and birth_counts_by_score[i] < len(birth_count_priors[i])), birth_counts_by_score[i]

		p_target_does_not_emit = 1.0 - sum(target_emission_probs)
		assoc_prior = (p_target_does_not_emit)**(unobserved_target_count) \
					  /count_meas_orderings(number_measurements, observed_target_count, \
						  					birth_count, clutter_count)
		for i in range(len(score_intervals)):
			assoc_prior *= target_emission_probs[i]**(meas_counts_by_score[i]) \
							  *birth_count_priors[i][birth_counts_by_score[i]] \
							  *clutter_count_priors[i][clutter_counts_by_score[i]] \
						  

		total_prior = death_prior * assoc_prior
		assert(total_prior != 0.0), (death_prior, assoc_prior)
#		return total_prior
		return assoc_prior

	def get_exact_prob_hidden_and_data(self, meas_source_index, measurement_list, living_target_indices, total_target_count,
									   measurement_associations, p_target_deaths, measurement_scores, score_intervals):
		"""
		REDOCUMENT, BELOW INCORRECT, not including death probability now
		Calculate p(data, associations, #measurements, deaths) as:
		p(data|deaths, associations, #measurements)*p(deaths)*p(associations, #measurements|deaths)
		Input:
		- measurement_list: a list of all measurements from the current time instance, from the measurement
			source with index meas_source_index
		- living_target_indices: a list of indices of targets from last time instance that are still alive
		- total_target_count: the number of living targets on the previous time instace
		- measurement_associations: a list of association values for each measurement. Each association has the value
			of a living target index (index from last time instance), target birth (total_target_count), 
			or clutter (-1)
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Return:
		- p(data, associations, #measurements, deaths)


		*note* p(data|deaths, associations, #measurements) is referred to as the likelihood and
		p(deaths)*p(associations, #measurements|deaths) as the prior, even though the number of measurements
		is part of the data (or an observed variable)
		"""

		prior = self.get_prior(living_target_indices, total_target_count, len(measurement_list), 
				 				   measurement_associations, p_target_deaths, TARGET_EMISSION_PROBS[meas_source_index], 
								   BIRTH_PROBABILITIES[meas_source_index], CLUTTER_PROBABILITIES[meas_source_index], measurement_scores, score_intervals)

#		hidden_state = HiddenState(living_target_indices, total_target_count, len(measurement_list), 
#				 				   measurement_associations, p_target_deaths, P_TARGET_EMISSION, 
#								   BIRTH_COUNT_PRIOR, CLUTTER_COUNT_PRIOR)
#		priorA = hidden_state.total_prior
#
#		assert(priorA == prior), (priorA, prior)

		likelihood = 1.0
		assert(len(measurement_associations) == len(measurement_list))
		for meas_index, meas_association in enumerate(measurement_associations):
			if(meas_association == total_target_count): #birth
				likelihood *= p_birth_likelihood
			elif(meas_association == -1): #clutter
				likelihood *= p_clutter_likelihood
			else:
				assert(meas_association >= 0 and meas_association < total_target_count), (meas_association, total_target_count)
				score_index = get_score_index(score_intervals, measurement_scores[meas_index])
				likelihood *= self.memoized_assoc_likelihood(measurement_list[meas_index], meas_source_index, \
											   				 meas_association, MEAS_NOISE_COVS[meas_source_index][score_index], score_index)

		assert(prior*likelihood != 0.0), (prior, likelihood)

		return prior*likelihood

	def memoized_assoc_likelihood(self, measurement, meas_source_index, target_index, meas_noise_cov, score_index):
		"""
			LSVM and regionlets produced two measurements with the same locations (centers), so using the 
			meas_source_index as part of the key is (sort of) necessary.  Currently also using the score_index, 
			could possibly be removed (not sure if this would improve speed).

			Currently saving more in the value than necessary (from debugging), can eliminate to improve
			performance (possibly noticable)
		"""


		global CACHED_LIKELIHOODS
		global NOT_CACHED_LIKELIHOODS
		if USE_CONSTANT_R:
			if((measurement[0], measurement[1], target_index, meas_source_index, score_index) in self.assoc_likelihood_cache):
				CACHED_LIKELIHOODS = CACHED_LIKELIHOODS + 1
				return self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)]
			else:
				NOT_CACHED_LIKELIHOODS = NOT_CACHED_LIKELIHOODS + 1
				target = self.targets.living_targets[target_index]
				S = np.dot(np.dot(H, target.P), H.T) + R_default
				assert(target.x.shape == (4, 1))
		
				state_mean_meas_space = np.dot(H, target.x)
				#print type(state_mean_meas_space)
				#print state_mean_meas_space
				state_mean_meas_space = np.squeeze(state_mean_meas_space)

				if USE_PYTHON_GAUSSIAN:
					distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
					assoc_likelihood = distribution.pdf(measurement)
				else:

#					S_det = np.linalg.det(S)
					S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
					S_inv = inv(S)
					LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

					offset = measurement - state_mean_meas_space
					a = -.5*np.dot(np.dot(offset, S_inv), offset)
					assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)



				self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)] = assoc_likelihood
				return assoc_likelihood

		else:
			if((measurement[0], measurement[1], target_index, meas_source_index, score_index) in self.assoc_likelihood_cache):
#			if((measurement[0], measurement[1], target_index, score_index) in self.assoc_likelihood_cache):
				CACHED_LIKELIHOODS = CACHED_LIKELIHOODS + 1
#				return self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, score_index)]
#				(assoc_likelihood, cached_score_index)	= self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, score_index)]
				(assoc_likelihood, cached_score_index, cached_measurement, cached_meas_source_index) = self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)]
				assert(cached_score_index == score_index), (cached_score_index, score_index, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
				assert(cached_meas_source_index == meas_source_index), (cached_score_index, score_index, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
#				if(cached_score_index != score_index):
#					print (cached_score_index, score_index, measurement, cached_measurement, target_index, meas_noise_cov)
#					time.sleep(2)
				return assoc_likelihood
			else:
				NOT_CACHED_LIKELIHOODS = NOT_CACHED_LIKELIHOODS + 1
				target = self.targets.living_targets[target_index]
				S = np.dot(np.dot(H, target.P), H.T) + meas_noise_cov
				assert(target.x.shape == (4, 1))
		
				state_mean_meas_space = np.dot(H, target.x)
				#print type(state_mean_meas_space)
				#print state_mean_meas_space
				state_mean_meas_space = np.squeeze(state_mean_meas_space)
				if USE_PYTHON_GAUSSIAN:
					distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
					assoc_likelihood = distribution.pdf(measurement)
				else:

##					S_det = np.linalg.det(S)

					S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
					S_inv = inv(S)
					LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

					offset = measurement - state_mean_meas_space
					a = -.5*np.dot(np.dot(offset, S_inv), offset)
					assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)

#				self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, score_index)] = assoc_likelihood
				self.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, score_index)] = (assoc_likelihood, score_index, measurement, meas_source_index)
				return assoc_likelihood

	def debug_target_creation(self):
		print
		print "Particle ", self.id_, "importance distribution:"
		print "pi_birth = ", self.pi_birth_debug, "pi_clutter = ", self.pi_clutter_debug, \
			"pi_targets = ", self.pi_targets_debug
		print "sampled association c = ", self.c_debug, "importance reweighting factor = ", self.imprt_re_weight_debug
		self.plot_all_target_locations()

	def process_meas_assoc(self, birth_value, meas_source_index, measurement_associations, measurements, \
		widths, heights, measurement_scores, cur_time):
		"""
		- meas_source_index: the index of the measurement source being processed (i.e. in SCORE_INTERVALS)

		"""
		for meas_index, meas_assoc in enumerate(measurement_associations):
			#create new target
			if(meas_assoc == birth_value):
				self.create_new_target(measurements[meas_index], widths[meas_index], heights[meas_index], cur_time)
				new_target = True 
			#update the target corresponding to the association we have sampled
			elif((meas_assoc >= 0) and (meas_assoc < birth_value)):
				assert(meas_source_index >= 0 and meas_source_index < len(SCORE_INTERVALS)), (meas_source_index, len(SCORE_INTERVALS), SCORE_INTERVALS)
				assert(meas_index >= 0 and meas_index < len(measurement_scores)), (meas_index, len(measurement_scores), measurement_scores)
				score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[meas_index])
				self.targets.living_targets[meas_assoc].kf_update(measurements[meas_index], widths[meas_index], \
					heights[meas_index], cur_time, MEAS_NOISE_COVS[meas_source_index][score_index])
			else:
				#otherwise the measurement was associated with clutter
				assert(meas_assoc == -1), ("meas_assoc = ", meas_assoc)

	#@profile
	def update_particle_with_measurement(self, cur_time, measurement_lists, widths, heights, measurement_scores):
		"""
		Input:
		- measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
			time instance from the ith measurement source (i.e. different object detection algorithms
			or different sensors)
		- measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
			measurement_list[i]
		
		-widths: a list where widths[i] is a list of bounding box widths for the corresponding measurements
		-heights: a list where heights[i] is a list of bounding box heights for the corresponding measurements

		Debugging output:
		- new_target: True if a new target was created
		"""
		new_target = False #debugging

		birth_value = self.targets.living_count

		(measurement_associations, dead_target_indices, imprt_re_weight) = \
			self.sample_data_assoc_and_death_mult_meas_per_time_proposal_distr_1(measurement_lists, \
				cur_time, measurement_scores)
		assert(len(measurement_associations) == len(measurement_lists))
		assert(imprt_re_weight != 0.0), imprt_re_weight
		self.importance_weight *= imprt_re_weight #update particle's importance weight
		#process measurement associations
		for meas_source_index in range(len(measurement_associations)):
			assert(len(measurement_associations[meas_source_index]) == len(measurement_lists[meas_source_index]) and
				   len(measurement_associations[meas_source_index]) == len(widths[meas_source_index]) and
				   len(measurement_associations[meas_source_index]) == len(heights[meas_source_index]))
			self.process_meas_assoc(birth_value, meas_source_index, measurement_associations[meas_source_index], \
				measurement_lists[meas_source_index], widths[meas_source_index], heights[meas_source_index], \
				measurement_scores[meas_source_index], cur_time)

		#process target deaths
		#double check dead_target_indices is sorted
		assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
		#important to delete larger indices first to preserve values of the remaining indices
		for index in reversed(dead_target_indices):
			self.targets.kill_target(index)

		#checking if something funny is happening
		original_num_targets = birth_value
		num_targets_born = 0
		for meas_source_index in range(len(measurement_associations)):
			num_targets_born += measurement_associations[meas_source_index].count(birth_value)
		num_targets_killed = len(dead_target_indices)
		assert(self.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
		#done checking if something funny is happening

		return new_target

	def plot_all_target_locations(self):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		for i in range(self.targets.total_count):
			life = len(self.targets.all_targets[i].all_states) #length of current targets life 
			locations_1D =  [self.targets.all_targets[i].all_states[j][0] for j in range(life)]
			ax.plot(self.targets.all_targets[i].all_time_stamps, locations_1D,
					'-o', label='Target %d' % i)

		legend = ax.legend(loc='lower left', shadow=True)
		plt.title('Particle %d, Importance Weight = %f, unique targets = %d, #targets alive = %d' % \
			(self.id_, self.importance_weight, self.targets.total_count, self.targets.living_count)) # subplot 211 title
#		plt.show()




###########assumed that the Kalman filter prediction step has already been run for this
###########target on the current time step
###########RUN PREDICTION FOR ALL TARGETS AT THE BEGINNING OF EACH TIME STEP!!!
###########@profile
##########def assoc_likelihood(measurement, target):
##########	S = np.dot(np.dot(H, target.P), H.T) + R_default
##########	assert(target.x.shape == (4, 1))
##########
##########	state_mean_meas_space = np.dot(H, target.x)
##########
##########	distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
##########	return distribution.pdf(measurement)

def normalize_importance_weights(particle_set):
	normalization_constant = 0.0
	for particle in particle_set:
		normalization_constant += particle.importance_weight
	assert(normalization_constant != 0.0), normalization_constant
	for particle in particle_set:
		particle.importance_weight /= normalization_constant


def perform_resampling(particle_set):
	assert(len(particle_set) == N_PARTICLES)
	weights = []
	for particle in particle_set:
		weights.append(particle.importance_weight)
	assert(abs(sum(weights) - 1.0) < .0000001)

	new_particles = stratified_resample(weights)
	new_particle_set = []
	for index in new_particles:
		new_particle_set.append(copy.deepcopy(particle_set[index]))
	del particle_set[:]
	for particle in new_particle_set:
		particle.importance_weight = 1.0/N_PARTICLES
		particle_set.append(particle)
	assert(len(particle_set) == N_PARTICLES)
	#testing
	weights = []
	for particle in particle_set:
		weights.append(particle.importance_weight)
		assert(particle.importance_weight == 1.0/N_PARTICLES)
	assert(abs(sum(weights) - 1.0) < .01), sum(weights)
	#done testing

def display_target_counts(particle_set, cur_time):
	target_counts = []
	for particle in particle_set:
		target_counts.append(particle.targets.living_count)
	print target_counts

	target_counts = []
	importance_weights = []
	for particle in particle_set:
		cur_target_count = 0
		for target in particle.targets.living_targets:
			if (cur_time - target.birth_time) > min_target_age:
				cur_target_count += 1
		target_counts.append(cur_target_count)
		importance_weights.append(particle.importance_weight)
	print "targets older than ", min_target_age, "seconds: ", target_counts
	print "importance weights ", min_target_age, "filler :", importance_weights


def get_eff_num_particles(particle_set):
	n_eff = 0
	weight_sum = 0
	for particle in particle_set:
		n_eff += particle.importance_weight**2
		weight_sum += particle.importance_weight

	assert(abs(weight_sum - 1.0) < .000001), (weight_sum, n_eff)
	return 1.0/n_eff



def run_rbpf_on_targetset(target_sets):
	"""
	Measurement class designed to only have 1 measurement/time instance
	Input:
	- target_sets: a list where target_sets[i] is a TargetSet containing measurements from
		the ith measurement source
	Output:
	- max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
		importance weight particle after processing all measurements
	"""
	particle_set = []
	for i in range(0, N_PARTICLES):
		particle_set.append(Particle(i))

	prev_time_stamp = -1


	#for displaying results
	time_stamps = []
	positions = []

	iter = 1 # for plotting only occasionally
	number_resamplings = 0

	#sanity check
	number_time_instances = len(target_sets[0].measurements)
	for target_set in target_sets:
		assert(len(target_set.measurements) == number_time_instances)

	for time_instance_index in range(number_time_instances):
		time_stamp = target_sets[0].measurements[time_instance_index].time
		for target_set in target_sets:
			assert(target_set.measurements[time_instance_index].time == time_stamp)
#		measurements = measurement_set.val

		measurement_lists = []
		widths = []
		heights = []
		measurement_scores = []
		for target_set in target_sets:
			measurement_lists.append(target_set.measurements[time_instance_index].val)
			widths.append(target_set.measurements[time_instance_index].widths)
			heights.append(target_set.measurements[time_instance_index].heights)
			measurement_scores.append(target_set.measurements[time_instance_index].scores)

		print "time_stamp = ", time_stamp, "living target count in first particle = ",\
		particle_set[0].targets.living_count
		for particle in particle_set:
			#update particle death probabilities
			if(prev_time_stamp != -1):
				particle.assoc_likelihood_cache = {} #clear likelihood cache
				#Run Kalman filter prediction for all living targets
				for target in particle.targets.living_targets:
					dt = time_stamp - prev_time_stamp
					assert(abs(dt - default_time_step) < .00000001), (dt, default_time_step)
					target.kf_predict(dt, time_stamp)
				#update particle death probabilities AFTER kf_predict so that targets that moved
				#off screen this time instance will be killed
				particle.update_target_death_probabilities(time_stamp, prev_time_stamp)

		new_target_list = [] #for debugging, list of booleans whether each particle created a new target
		for particle in particle_set:
			new_target = particle.update_particle_with_measurement(time_stamp, measurement_lists, widths, heights, measurement_scores)
			new_target_list.append(new_target)
		normalize_importance_weights(particle_set)
		#debugging
		if DEBUG:
			assert(len(new_target_list) == N_PARTICLES)
			for (particle_number, new_target) in enumerate(new_target_list):
				if new_target:
					print "\n\n -------Particle %d created a new target-------" % particle_number
					for particle in particle_set:
						particle.debug_target_creation()
					plt.show()
					break
		#done debugging

		if iter%100 == 0:
			print iter
			display_target_counts(particle_set, time_stamp)


		if (get_eff_num_particles(particle_set) < N_PARTICLES/RESAMPLE_RATIO):
			perform_resampling(particle_set)
			print "resampled on iter: ", iter
			number_resamplings += 1
		prev_time_stamp = time_stamp



		iter+=1

	print "resampling performed %d times" % number_resamplings

	max_imprt_weight = -1
	for particle in particle_set:
		if(particle.importance_weight > max_imprt_weight):
			max_imprt_weight = particle.importance_weight
	for particle in particle_set:
		if(particle.importance_weight == max_imprt_weight):
			max_weight_target_set = particle.targets

	return max_weight_target_set


def test_read_write_data_KITTI(target_set):
	"""
	Measurement class designed to only have 1 measurement/time instance
	Input:
	- target_set: generated TargetSet containing generated measurements and ground truth
	Output:
	- max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
		importance weight particle after processing all measurements
	"""
	output_target_set = TargetSet()

	for measurement_set in target_set.measurements:
		time_stamp = measurement_set.time
		measurements = measurement_set.val
		widths = measurement_set.widths
		heights = measurement_set.heights

		for i in range(len(measurements)):
			output_target_set.create_new_target(measurements[i], widths[i], heights[i], time_stamp)

	return output_target_set



def convert_to_clearmetrics_dictionary(target_set, all_time_stamps):
	"""
	Convert the locations of a TargetSet to clearmetrics dictionary format

	Input:
	- target_set: TargetSet to be converted

	Output:
	- target_dict: Converted locations in clearmetrics dictionary format
	"""
	target_dict = {}
	for target in target_set.all_targets:
		for t in all_time_stamps:
			if target == target_set.all_targets[0]: #this is the first target
				if t in target.all_time_stamps: #target exists at this time
					target_dict[t] = [target.all_states[target.all_time_stamps.index(t)]]
				else: #target doesn't exit at this time
					target_dict[t] = [None]
			else: #this isn't the first target
				if t in target.all_time_stamps: #target exists at this time
					target_dict[t].append(target.all_states[target.all_time_stamps.index(t)])
				else: #target doesn't exit at this time
					target_dict[t].append(None)
	return target_dict

def calc_tracking_performance(ground_truth_ts, estimated_ts):
	"""
	!!I think clearmetrics calculates #mismatches incorrectly, look into more!!
	(has to do with whether a measurement can be mismatched to a target that doesn't exist at the current time)

	Calculate MOTA and MOTP ("Evaluating Multiple Object Tracking Performance:
	The CLEAR MOT Metrics", K. Bernardin and R. Stiefelhagen)

	Inputs:
	- ground_truth_ts: TargetSet containing ground truth target locations
	- estimated_ts: TargetSet containing esimated target locations
	"""

	#convert TargetSets to dictionary format for calling clearmetrics

	all_time_stamps = [ground_truth_ts.measurements[i].time for i in range(len(ground_truth_ts.measurements))]
	ground_truth = convert_to_clearmetrics_dictionary(ground_truth_ts, all_time_stamps)
	estimated_tracks = convert_to_clearmetrics_dictionary(estimated_ts, all_time_stamps)

	clear = clearmetrics.ClearMetrics(ground_truth, estimated_tracks, MAX_ASSOCIATION_DIST)
	clear.match_sequence()
	evaluation = [clear.get_mota(),
	              clear.get_motp(),
	              clear.get_fn_count(),
	              clear.get_fp_count(),
	              clear.get_mismatches_count(),
	              clear.get_object_count(),
	              clear.get_matches_count()]
	print 'MOTA, MOTP, FN, FP, mismatches, objects, matches'
	print evaluation     
	ground_truth_ts.plot_all_target_locations("Ground Truth")         
	ground_truth_ts.plot_generated_measurements()    
	estimated_ts.plot_all_target_locations("Estimated Tracks")      
	plt.show()

#f = open(MEASURMENT_FILENAME, 'r')
#measurementTargetSetsBySequence = pickle.load(f)
#f.close()
#print '-'*80
#print measurementTargetSetsBySequence[0].measurements[0].time
#print measurementTargetSetsBySequence[0].measurements[1].time
#print measurementTargetSetsBySequence[0].measurements[2].time
#print measurementTargetSetsBySequence[0].measurements[3].time
#estimated_ts = run_rbpf_on_targetset(measurementTargetSetsBySequence[0])
#estimated_ts.write_targets_to_KITTI_format(num_frames = 154, filename = 'rbpf_training_0000_results.txt')

#estimated_ts = cProfile.run('run_rbpf_on_targetset(measurementTargetSetsBySequence[0])')


filename_mapping = "/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/KITTI_helpers/data/evaluate_tracking.seqmap"
n_frames         = []
sequence_name    = []
with open(filename_mapping, "r") as fh:
    for i,l in enumerate(fh):
        fields = l.split(" ")
        sequence_name.append("%04d" % int(fields[0]))
        n_frames.append(int(fields[3]) - int(fields[2]))
fh.close() 
print n_frames
print sequence_name     
assert(len(n_frames) == len(sequence_name) and len(n_frames) == len(measurementTargetSetsBySequence))
#for seq_idx in range(len(measurementTargetSetsBySequence)):
t0 = time.time()
for seq_idx in SEQUENCES_TO_PROCESS:
	print "Processing sequence: ", seq_idx
	estimated_ts = run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx])
#	estimated_ts = cProfile.run('run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx])')
	estimated_ts.write_targets_to_KITTI_format(num_frames = n_frames[seq_idx], \
											   filename = './rbpf_KITTI_results/%s.txt' % sequence_name[seq_idx])
t1 = time.time()

print "Cached likelihoods = ", CACHED_LIKELIHOODS
print "not cached likelihoods = ", NOT_CACHED_LIKELIHOODS
print "RBPF runtime = ", t1-t0
eval_results('/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/rbpf_KITTI_results', SEQUENCES_TO_PROCESS)
print "USE_CONSTANT_R = ", USE_CONSTANT_R
print "number of particles = ", N_PARTICLES
print "score intervals: ", SCORE_INTERVALS
print "run on sequences: ", SEQUENCES_TO_PROCESS
#test_target_set = test_read_write_data_KITTI(measurementTargetSetsBySequence[0])
#test_target_set.write_targets_to_KITTI_format(num_frames = 154, filename = 'test_read_write_0000_results.txt')

#estimated_ts = cProfile.run('run_rbpf_on_targetset(ground_truth_ts)')

#calc_tracking_performance(ground_truth_ts, estimated_ts)




