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

import cProfile

#RBPF algorithmic paramters
N_PARTICLES = 20000 #number of particles used in the particle filter
RESAMPLE_RATIO = 2.0 #resample when get_eff_num_particles < N_PARTICLES/RESAMPLE_RATIO

#algorithmic approximation parameters

#if true, sample target deaths before data associations
SAMPLE_DEATH_INDEPENDENTLY = False
#should event priors be normalized in sample_data_assoc_and_death
#when SAMPLE_DEATH_INDEPENDENTLY = False?
NORMALIZE_EVENT_PRIORS = False

PLOT = True
DEBUG = False
NUM_TARGETS = 6
WITH_NOISE = True
WITH_CLUTTER = True

#data generation parameters
meas_sigma = .2
#Multiple measurements may be generated on a single time instance if True
MULTIPLE_MEAS_PER_TIME = True 

#Pick a proposal distribution to use when MULTIPLE_MEAS_PER_TIME = True
USE_EXACT_PROPOSAL_DISTRIBUTION = False
USE_PROPOSAL_DISTRIBUTION_1 = True
assert(sum([USE_EXACT_PROPOSAL_DISTRIBUTION, USE_PROPOSAL_DISTRIBUTION_1]) == 1)

#default time between succesive measurement time instances (in seconds)
default_time_step = .01 
num_time_steps = 1500

#define parameters according to whether multiple measurements may be
#generated during a single time instance
if MULTIPLE_MEAS_PER_TIME:
	from multiple_meas_per_time_assoc_priors import enumerate_death_and_assoc_possibilities
	from multiple_meas_per_time_assoc_priors import HiddenState
	P_TARGET_EMISSION = .95
	BIRTH_COUNT_PRIOR = [0.996,.003,.001]
	CLUTTER_COUNT_PRIOR = [.7,.1,.1,.1]	
else:
	p_clutter_prior = .01 #probability of associating a measurement with clutter
	#p_birth_prior = 0.01 #probability of associating a measurement with a new target
	p_birth_prior = 0.0025 #probability of associating a measurement with a new target

p_clutter_likelihood = 0.1
#p_birth_likelihood = 0.035
p_birth_likelihood = 0.1


#Kalman filter defaults
#P_default = meas_sigma**2 #this may not be a reasonable/valid thing to do
P_default = np.array([[meas_sigma**2, 0],
 					  [0,             1]])

#P_default = np.array([[100,  0],
# 					  [  0, 10]])

R_default = meas_sigma**2
#Q_default = np.array([[ 0.04227087,  0.02025365],
# 					  [ 0.02025365,  0.00985709]])
#Q_default = Q_default/20.0 #just from testing seems to give good RMSE with clutter

process_noise_spectral_density = .1
Q_default = np.array([[1.0/3.0*default_time_step**3, 1.0/2.0*default_time_step**2],
 					  [1.0/2.0*default_time_step**2, default_time_step]])
Q_default *= process_noise_spectral_density

#process_noise_spectral_density = .1
#Q_default = np.array([[1.0/3.0*default_time_step**3, 1.0/2.0*default_time_step**2],
# 					  [1.0/2.0*default_time_step**2, default_time_step*100]])
#Q_default *= process_noise_spectral_density

#measurement function matrix
H = np.array([[1,  0]])


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


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def gen_lines():
	time_steps = np.asarray([i*default_time_step for i in range(0, num_time_steps)])

	#generate x1
	x1 = []
	for i in range(0,num_time_steps):
		x1.append(-2)

	#generate x2
	x2 = []
	for i in range(0,num_time_steps):
		x2.append(-1)

	#generate x3
	x3 = []
	for i in range(0,num_time_steps):
		x3.append(0)

	#generate x4
	x4 = []
	for i in range(0,num_time_steps):
		x4.append(1)

	#generate x5
	x5 = []
	for i in range(0,num_time_steps):
		x5.append(2)

	#generate x6
	x6 = []
	for i in range(0,num_time_steps):
		x6.append(3)



	cmap = get_cmap(100)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(time_steps, x1, marker = 'x', c = cmap(7))
	ax.scatter(time_steps, x2, marker = 'x', c = cmap(95))
	ax.scatter(time_steps, x3, marker = 'x', c = cmap(20))
	ax.scatter(time_steps, x4, marker = 'x', c = cmap(40))
	ax.scatter(time_steps, x5, marker = 'x', c = cmap(60))
	ax.scatter(time_steps, x6, marker = 'x', c = cmap(80))
	plt.show()

	return time_steps, x1, x2, x3, x4, x5, x6

def gen_data():
	time_steps = np.asarray([i*default_time_step for i in range(0, num_time_steps)])

	#generate x1
	x1 = []
	for i in range(0,800):
		if WITH_NOISE:
			x1.append(math.sin(time_steps[i]) + np.random.normal(0, meas_sigma))
		else:
			x1.append(math.sin(time_steps[i]))

	for i in range(800,num_time_steps):
		x1.append(float('nan'))

	#generate x2
	x2 = []
	for i in range(0,num_time_steps):
		if WITH_NOISE:
			x2.append(3 - 3/10.0*time_steps[i] + np.random.normal(0, meas_sigma))
		else:
			x2.append(3 - 3/10.0*time_steps[i])

	#generate x3
	x3 = []
	for i in range(0,100):
		x3.append(float('nan'))
	for i in range(100,400):
		if WITH_NOISE:
			x3.append(1/3.0*math.sin(time_steps[i]) + 4 + np.random.normal(0, meas_sigma) )
		else:
			x3.append(1/3.0*math.sin(time_steps[i]) + 4)

	for i in range(400,num_time_steps):
		x3.append(float('nan'))

	#generate x4
	x4 = []
	for i in range(0,200):
		x4.append(float('nan'))
	for i in range(200,500):
		if WITH_NOISE:
			x4.append(math.cos(time_steps[i]/2.0) - 2.0  + np.random.normal(0, meas_sigma) )
		else:
			x4.append(math.cos(time_steps[i]/2.0) - 2.0 )

	for i in range(500,num_time_steps):
		x4.append(float('nan'))

	#generate x5
	x5 = []
	for i in range(0,550):
		x5.append(float('nan'))
	for i in range(550,1000):
		if WITH_NOISE:
			x5.append(1.5*math.sin(time_steps[i]/2.8+3) - 1.3 + np.random.normal(0, meas_sigma) )
		else:
			x5.append(1.5*math.sin(time_steps[i]/2.8+3) - 1.3)

	for i in range(1000,num_time_steps):
		x5.append(float('nan'))

	#generate x6
	x6 = []
	for i in range(0,600):
		x6.append(float('nan'))
	for i in range(600,num_time_steps):
		if WITH_NOISE:
			x6.append(.5*math.sin(6.0/7*time_steps[i]-6) + 4 + np.random.normal(0, meas_sigma) )
		else:
			x6.append(.5*math.sin(6.0/7*time_steps[i]-6) + 4)


	cmap = get_cmap(100)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(time_steps, x1, marker = 'x', c = cmap(7))
	ax.scatter(time_steps, x2, marker = 'x', c = cmap(95))
	ax.scatter(time_steps, x3, marker = 'x', c = cmap(20))
	ax.scatter(time_steps, x4, marker = 'x', c = cmap(40))
	ax.scatter(time_steps, x5, marker = 'x', c = cmap(60))
	ax.scatter(time_steps, x6, marker = 'x', c = cmap(80))
	plt.show()

	return time_steps, x1, x2, x3, x4, x5, x6

def gen_orig_paper_data():
	time_steps = np.asarray([i*default_time_step for i in range(0, num_time_steps)])

	#generate x1
	x1 = []
	for i in range(0,800):
		if WITH_NOISE:
			x1.append(math.sin(time_steps[i]) + np.random.normal(0, meas_sigma))
		else:
			x1.append(math.sin(time_steps[i]))

	for i in range(800,num_time_steps):
		x1.append(float('nan'))

	#generate x2
	x2 = []
	for i in range(0,num_time_steps):
		if WITH_NOISE:
			x2.append(3 - 3/10.0*time_steps[i] + np.random.normal(0, meas_sigma))
		else:
			x2.append(3 - 3/10.0*time_steps[i])

	#generate x3
	x3 = []
	for i in range(0,100):
		x3.append(float('nan'))
	for i in range(100,400):
		if WITH_NOISE:
			x3.append(1/2.0*math.sin(.9*time_steps[i]) + 4 + np.random.normal(0, meas_sigma) )
		else:
			x3.append(1/2.0*math.sin(.9*time_steps[i]) + 4)

	for i in range(400,num_time_steps):
		x3.append(float('nan'))

	#generate x4
	x4 = []
	for i in range(0,200):
		x4.append(float('nan'))
	for i in range(200,500):
		if WITH_NOISE:
			x4.append(math.cos(time_steps[i]/2.0) - 2.0 + np.random.normal(0, meas_sigma) )
		else:
			x4.append(math.cos(time_steps[i]/2.0) - 2.0 )

	for i in range(500,num_time_steps):
		x4.append(float('nan'))

	#generate x5
	x5 = []
	for i in range(0,550):
		x5.append(float('nan'))
	for i in range(550,1000):
		if WITH_NOISE:
			x5.append(math.cos(time_steps[i]/2.0) - 2.0 + np.random.normal(0, meas_sigma) )
		else:
			x5.append(math.cos(time_steps[i]/2.0) - 2.0)

	for i in range(1000,num_time_steps):
		x5.append(float('nan'))

	#generate x6
	x6 = []
	for i in range(0,600):
		x6.append(float('nan'))
	for i in range(600,num_time_steps):
		if WITH_NOISE:
			x6.append(1/2.0*math.sin(.9*time_steps[i]) + 4 + np.random.normal(0, meas_sigma) )
		else:
			x6.append(1/2.0*math.sin(.9*time_steps[i]) + 4)


	cmap = get_cmap(100)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(time_steps, x1, marker = 'x', c = cmap(7))
	ax.scatter(time_steps, x2, marker = 'x', c = cmap(95))
	ax.scatter(time_steps, x3, marker = 'x', c = cmap(20))
	ax.scatter(time_steps, x4, marker = 'x', c = cmap(40))
	ax.scatter(time_steps, x5, marker = 'x', c = cmap(60))
	ax.scatter(time_steps, x6, marker = 'x', c = cmap(80))
	plt.show()

	return time_steps, x1, x2, x3, x4, x5, x6


def convert_data_to_rbpf_format_multiple_meas_per_time_step(time_steps, input_measurements):
	"""
	Input:
	input_measurements: list of lists where input_measurements[j][i] is the ith measurement
		from the jth target
	"""
	all_measurements = []
	for i in range(0,num_time_steps):
		cur_measurements = []
		for j in range(0, NUM_TARGETS):
			if (not math.isnan(input_measurements[j][i])):
				if WITH_CLUTTER and random.random() < .01:
					cur_measurements.append(np.array([random.random()*10 - 5]))
				else:
					cur_measurements.append(np.array([input_measurements[j][i]]))
		all_measurements.append((time_steps[i], cur_measurements))
	return all_measurements

def convert_data_to_rbpf_format_single_meas_per_time_step(time_steps, input_measurements):
	"""
	Input:
	input_measurements: list of lists where input_measurements[j][i] is the ith measurement
		from the jth target
	"""
	all_measurements = []
	for i in range(0,num_time_steps):
		cur_measurements = []
		#with probability .01 the observation is clutter
		if WITH_CLUTTER and random.random() < .01:
			cur_measurements.append(np.array([random.random()*10 - 5]))
		#with probablitiy .99 the observation is uniformly picked from
		#among the visible targets
		else:
			visible_targets = []
			for j in range(0, NUM_TARGETS):
				if (not math.isnan(input_measurements[j][i])):
					visible_targets.append(input_measurements[j][i])
			#if (not math.isnan(x1[i])):
			#		visible_targets.append(x1[i])
			#if (not math.isnan(x2[i])):
			#	visible_targets.append(x2[i])
			#if (not math.isnan(x3[i])):
			#	visible_targets.append(x3[i])
			#if (not math.isnan(x4[i])):
			#	visible_targets.append(x4[i])
			#if (not math.isnan(x5[i])):
			#	visible_targets.append(x5[i])
			#if (not math.isnan(x6[i])):
			#	visible_targets.append(x6[i])
			cur_measurements.append(np.array([np.random.choice(visible_targets)]))
		all_measurements.append((time_steps[i], cur_measurements))
	return all_measurements

def plot_rbpf_formatted_data(all_measurements):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	for (time_step, cur_meas) in all_measurements:
		for meas in cur_meas:
#			print type(meas)
#			print meas 
#			print meas.shape
			ax.scatter(time_step, meas, marker = 'x')
	plt.show()

class Target:
	def __init__(self, cur_time, id_, measurement = None):
		if measurement is None: #for data generation
			position = np.random.uniform(min_pos,max_pos)
			velocity = np.random.uniform(min_vel,max_vel)
			self.x = np.array([[position], [velocity]])
			self.P = P_default
		else:
			self.x = np.array([[measurement], [0]])
			self.P = P_default

		assert(self.x.shape == (2, 1))
		self.birth_time = cur_time
		#Time of the last measurement data association with this target
		self.last_measurement_association = cur_time
		self.id_ = id_ #named id_ to avoid clash with built in id
		self.death_prob = -1 #calculate at every time instance

		self.all_states = [self.x]
		self.all_time_stamps = [cur_time]

		self.measurements = []
		self.measurement_time_stamps = []

	def kf_update(self, measurement, cur_time):
		""" Perform Kalman filter update step and replace predicted position for the current time step
		with the updated position in self.all_states
		Input:
		- measurement: the measurement (numpy array)
		- cur_time: time when the measurement was taken (float)
!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
		"""
		assert(self.x.shape == (2, 1))
		S = np.dot(np.dot(H, self.P), H.T) + R_default
		K = np.dot(np.dot(self.P, H.T), inv(S))
		residual = measurement - np.dot(H, self.x)
		updated_x = self.x + np.dot(K, residual)
	#	updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
		updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
		self.x = updated_x
		self.P = updated_P
		assert(self.all_time_stamps[-1] == cur_time and self.all_time_stamps[-2] != cur_time)
		assert(self.x.shape == (2, 1)), (self.x.shape, np.dot(K, residual).shape)

		self.all_states[-1] = self.x

	def kf_predict(self, dt, cur_time):
		"""
		Run kalman filter prediction on this target
		Inputs:
			-dt: time step to run prediction on
			-cur_time: the time the prediction is made for
		"""
		assert(self.all_time_stamps[-1] == (cur_time - dt))
		F = np.array([[1, dt],
		      		  [0,  1]])
		x_predict = np.dot(F, self.x)
		P_predict = np.dot(np.dot(F, self.P), F.T) + Q_default
		self.x = x_predict
		self.P = P_predict
		self.all_states.append(self.x)
		self.all_time_stamps.append(cur_time)
		assert(self.x.shape == (2, 1))


	def data_gen_update_state(self, cur_time, F):
		process_noise = np.random.multivariate_normal(np.zeros(Q_default.shape[0]), Q_default)
		process_noise = np.expand_dims(process_noise, axis=1)
		self.x = np.dot(F, self.x) + process_noise 
		self.all_states.append(self.x)
		self.all_time_stamps.append(cur_time)
		assert(self.x.shape == (2, 1))

	def data_gen_measure_state(self, cur_time):
		measurement_noise = np.random.multivariate_normal(np.zeros(R_default.shape[0]), R_default)
		measurement_noise = np.expand_dims(measurement_noise, axis=1)
		measurement = np.dot(H, self.x) + measurement_noise
		self.measurements.append(measurement)
		self.measurement_time_stamps.append(cur_time)
		assert(self.x.shape == (2, 1))

		return measurement

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

		#scipy.special.gdtrc(b, a, x) calculates 
		#integral(gamma_dist(k = a, theta = b))from x to infinity
		last_assoc = self.last_measurement_association

		#I think this is correct
		death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
		death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
		return death_prob

#		#this is used in paper's code
#		time_step = cur_time - prev_time
#	
#		death_prob = gdtrc(theta_death, alpha_death, cur_time - last_assoc) \
#				   - gdtrc(theta_death, alpha_death, cur_time - last_assoc + time_step)
#		death_prob /= gdtrc(theta_death, alpha_death, cur_time - last_assoc)
#		return death_prob


class Measurement:
	def __init__(self, time = -1):
		#if MULTIPLE_MEAS_PER_TIME = False, self.val will be a numpy array
		#if MULTIPLE_MEAS_PER_TIME = True, self.val will be a list of numpy arrays
		if MULTIPLE_MEAS_PER_TIME:
			self.val = []
		else:
			self.val = np.nan
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

	def create_new_target(self, measurement, cur_time):
		new_target = Target(cur_time, self.total_count, measurement[0])
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
		if MULTIPLE_MEAS_PER_TIME:
			time_stamps = [self.measurements[i].time for i in range(len(self.measurements))
													 for j in range(len(self.measurements[i].val))]
			locations = [self.measurements[i].val[j][0] for i in range(len(self.measurements))
														for j in range(len(self.measurements[i].val))]
		else:
			time_stamps = [self.measurements[i].time for i in range(len(self.measurements))]
			locations = [self.measurements[i].val[0] for i in range(len(self.measurements))]
		ax.plot(time_stamps, locations,'o')
		plt.title('Generated Measurements') 


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

	def create_new_target(self, measurement, cur_time):
		self.targets.create_new_target(measurement, cur_time)

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


	def sample_deaths_among_specific_targets(self, at_risk_targets):
		"""
		Sample deaths (without killing !!) targets from at_risk_targets
		
		death_prob for every target should have already been calculated!!

		Input:
		- at_risk_targets: a list of targets that may be killed

		Output:
		- total_death_prob: the probability of sampled deaths
		- targets_to_kill : a list of targets that should be killed
		"""
		total_death_prob = 1.0
		targets_to_kill = []
		for index in at_risk_targets:
			cur_target = self.targets.living_targets[index]
			death_prob = cur_target.death_prob
			assert(death_prob < 1.0 and death_prob > 0.0)
			if (random.random() < death_prob):
				targets_to_kill.append(index)
				total_death_prob *= death_prob
			else:
				total_death_prob *= (1.0 - death_prob)

		return (total_death_prob, targets_to_kill)

	def sample_data_assoc(self, measurement):
		"""
		Sample only data association (target deaths have already been independently sampled)
		Input:

		Output:
		- c: The measurement-target association value.  Values of c correspond to:
			c = -1 -> clutter
			c = self.targets.living_count -> new target
			c in range [0, self.targets.living_count-1] -> particle.targets.living_targets[c]
		- normalization: After processing this measurement the particle's
			importance weight will be:
			new_importance_weight = old_importance_weight * normalization

		Cases (T = number of targets):
			1: c = clutter
				-1 option

			2: c = birth
				-1 option

			3: c = current target association
				-T options
		"""

		num_targ = self.targets.living_count
		event_priors = np.array([-9.0 for i in range(0, 2+num_targ)])
		event_likelihoods = np.array([-9.0 for i in range(0, 2+num_targ)])
		event_associations = np.array([-9 for i in range(0, 2+num_targ)])

		event_index = 0
		#case 1: c = clutter, 1 option
		event_priors[event_index] = p_clutter_prior
		event_likelihoods[event_index] = p_clutter_likelihood
		event_associations[event_index] = -1
		event_index += 1

		#case 2: c = birth, 1 option
		event_priors[event_index] = p_birth_prior 
		event_likelihoods[event_index] = p_birth_likelihood
		event_associations[event_index] = self.targets.living_count
		event_index += 1

		#case 3: c = current target association, T options
		for i in range(self.targets.living_count):
			event_priors[event_index] = (1.0 - p_birth_prior - p_clutter_prior) \
										/(self.targets.living_count)
			event_likelihoods[event_index] = assoc_likelihood(measurement, self.targets.living_targets[i])
			event_associations[event_index] = i
			event_index += 1

		assert(event_index == 2+num_targ)

		#normalize event priors (as in generative model) when no living targets
		if(self.targets.living_count == 0):
			prior_normalization = np.sum(event_priors)
			event_priors /= prior_normalization
			assert(abs(np.sum(event_priors) - 1.0 < .000001))
		#if there are living targets, priors should already be normalized
		else:
			assert(abs(np.sum(event_priors) - 1.0 < .000001))

		pi_distribution = event_priors*event_likelihoods
		normalization = np.sum(pi_distribution)
		pi_distribution /= normalization
		assert(abs(np.sum(pi_distribution) - 1.0 < .000001))

		#now sample from the importance distribution
		sampled_index = np.random.choice(len(pi_distribution), p=pi_distribution)

		assert(abs(normalization - event_likelihoods[sampled_index]*event_priors[sampled_index]/pi_distribution[sampled_index]) < .000001)
		return (event_associations[sampled_index], normalization)




	#@profile
	def sample_data_assoc_and_death(self, measurement):
		"""
		Input:

		Output:
		- c: The measurement-target association value.  Values of c correspond to:
			c = -1 -> clutter
			c = self.targets.living_count -> new target
			c in range [0, self.targets.living_count-1] -> particle.targets.living_targets[c]
		- normalization: After processing this measurement the particle's
			importance weight will be:
			new_importance_weight = old_importance_weight * normalization
		- dead_target_ind: Index of the target that was killed (max of one target can be killed),
						   -1 if no targets died


		Cases (T = number of targets):
			1: 0 deaths, c = clutter
				-1 option

			2: 0 deaths, c = birth
				-1 option

			3: 0 deaths, c = current target association
				-T options

			4: 1 death, c = clutter
				-T options

			5: 1 death, c = birth
				-T options

			6: 1 death, c = current target association (not with the target that just died)
				-T*(T-1) options
		"""

		#get death probabilities for each target in a numpy array
		death_probs = []
		for target in self.targets.living_targets:
			death_probs.append(target.death_prob)
			assert(death_probs[len(death_probs) - 1] >= 0 and death_probs[len(death_probs) - 1] <= 1)
		#if we have no targets, create length one array containing a zero so cases 1 and 2 work out
		if(len(death_probs) == 0): 
			death_probs.append(0)
		death_probs = np.asarray(death_probs)

		num_targ = self.targets.living_count
		event_priors = np.array([-9.0 for i in range(0, 2+2*num_targ+num_targ**2)])
		event_likelihoods = np.array([-9.0 for i in range(0, 2+2*num_targ+num_targ**2)])
		event_associations = np.array([-9 for i in range(0, 2+2*num_targ+num_targ**2)])
		event_deaths = np.array([-9 for i in range(0, 2+2*num_targ+num_targ**2)])

		event_index = 0
		#case 1: 0 deaths, c = clutter, 1 option
		event_priors[event_index] = np.prod(1 - death_probs) * p_clutter_prior
		event_likelihoods[event_index] = p_clutter_likelihood
		event_associations[event_index] = -1
		event_deaths[event_index] = -1
		event_index += 1

		#case 2: 0 deaths, c = birth, 1 option
		event_priors[event_index] = np.prod(1 - death_probs) * p_birth_prior 
		event_likelihoods[event_index] = p_birth_likelihood
		event_associations[event_index] = self.targets.living_count
		event_deaths[event_index] = -1
		event_index += 1

		#case 3: 0 deaths, c = current target association, T options
		for i in range(self.targets.living_count):
			event_priors[event_index] = np.prod(1 - death_probs)*(1.0 - p_birth_prior - p_clutter_prior) \
										/(self.targets.living_count)
			event_likelihoods[event_index] = assoc_likelihood(measurement, self.targets.living_targets[i])
			event_associations[event_index] = i
			event_deaths[event_index] = -1
			event_index += 1

		#case 4: 1 death, c = clutter, T options
		for i in range(self.targets.living_count):
			event_priors[event_index] = np.prod(1 - death_probs)/(1 - death_probs[i])*death_probs[i] \
										*p_clutter_prior
			event_likelihoods[event_index] = p_clutter_likelihood
			event_associations[event_index] = -1
			event_deaths[event_index] = i
			event_index += 1

		#case 5: 1 death, c = birth, T options
		for i in range(self.targets.living_count):
			event_priors[event_index] = np.prod(1 - death_probs)/(1 - death_probs[i])*death_probs[i] \
										* p_birth_prior
			event_likelihoods[event_index] = p_birth_likelihood
			event_associations[event_index] = self.targets.living_count
			event_deaths[event_index] = i
			event_index += 1

		#case 6: 1 death, c = current target association (not with the target that just died),
		#		 T*(T-1) options
		for death_index in range(self.targets.living_count):
			for assoc_index in range(self.targets.living_count):
				if(death_index != assoc_index):
					event_priors[event_index] = np.prod(1 - death_probs)/(1 - death_probs[death_index]) \
												* death_probs[death_index] \
												* (1.0 - p_birth_prior - p_clutter_prior) \
												/(self.targets.living_count-1)
					event_likelihoods[event_index] = assoc_likelihood(measurement, self.targets.living_targets[assoc_index])
					event_associations[event_index] = assoc_index
					event_deaths[event_index] = death_index
					event_index += 1

		assert(event_index == 2+2*num_targ+num_targ**2)

		#always normalize event priors (as in generative model) when no living targets
		if(self.targets.living_count == 0):
			prior_normalization = np.sum(event_priors)
			event_priors /= prior_normalization
			assert(abs(np.sum(event_priors) - 1.0 < .000001))
		elif(NORMALIZE_EVENT_PRIORS):
			prior_normalization = np.sum(event_priors)
			event_priors /= prior_normalization
			assert(abs(np.sum(event_priors) - 1.0 < .000001))

		pi_distribution = event_priors*event_likelihoods
		normalization = np.sum(pi_distribution)
		pi_distribution /= normalization
		assert(abs(np.sum(pi_distribution) - 1.0 < .000001))

		#now sample from the importance distribution
		sampled_index = np.random.choice(len(pi_distribution), p=pi_distribution)

		assert(abs(normalization - event_likelihoods[sampled_index]*event_priors[sampled_index]/pi_distribution[sampled_index]) < .000001)
		return (event_associations[sampled_index], event_deaths[sampled_index], normalization)
#		return (event_associations[sampled_index], event_deaths[sampled_index], event_likelihoods[sampled_index]*event_priors[sampled_index]/pi_distribution[sampled_index])






	def sample_data_assoc_and_death_mult_meas_per_time(self, measurement_list):
		"""
		Input:
		- measurement_list: a list of all measurements from the current time instance
		Output:
		- c: A list of association values for each measurement.  Values of c correspond to:
			c[i] = -1 -> ith measurement is clutter
			c[i] = self.targets.living_count -> ith measurement is a new target
			c[i] in range [0, self.targets.living_count-1] -> ith measurement is of
				particle.targets.living_targets[c[i]]

		- normalization: After processing this measurement the particle's
			importance weight will be:
			new_importance_weight = old_importance_weight * normalization
		- dead_target_indices: a list containing the indices of targets that died, beginning
			with the smallest index in increasing order, e.g. [0, 4, 6, 33]
		"""

		#get death probabilities for each target in a numpy array
		death_probs = []
		for target in self.targets.living_targets:
			death_probs.append(target.death_prob)
			assert(death_probs[len(death_probs) - 1] >= 0 and death_probs[len(death_probs) - 1] <= 1)

		num_targ = self.targets.living_count

		hidden_state_possibilities = enumerate_death_and_assoc_possibilities(num_targ, len(measurement_list),
										death_probs, P_TARGET_EMISSION, BIRTH_COUNT_PRIOR, CLUTTER_COUNT_PRIOR)


		#create the importance distribution
		pi_distribution = []
		for cur_hidden_state_possibility in hidden_state_possibilities:
			prior = cur_hidden_state_possibility.total_prior
			likelihood = 1.0
			cur_associations = cur_hidden_state_possibility.measurement_associations
			assert(len(cur_associations) == len(measurement_list))
			for meas_index, meas_association in enumerate(cur_associations):
				if(meas_association == num_targ): #birth
					likelihood *= p_birth_likelihood
				elif(meas_association == -1): #clutter
					likelihood *= p_clutter_likelihood
				else:
					assert(meas_association >= 0 and meas_association < num_targ), (meas_association, num_targ)
#					likelihood *= assoc_likelihood(measurement_list[meas_index], 
#												   self.targets.living_targets[meas_association])
					likelihood *= self.memoized_assoc_likelihood(measurement_list[meas_index], \
												   				 meas_association)
			pi_distribution.append(prior*likelihood)

		assert(len(pi_distribution) == len(hidden_state_possibilities))
		assert(len(pi_distribution)>0), len(pi_distribution)

		pi_distribution = np.asarray(pi_distribution)
		normalization = np.sum(pi_distribution)
		pi_distribution /= normalization
		assert(abs(np.sum(pi_distribution) - 1.0 < .000001))
		#now sample from the importance distribution
		sampled_index = np.random.choice(len(pi_distribution), p=pi_distribution)

		#sampled measurement associations
		c = hidden_state_possibilities[sampled_index].measurement_associations
		dead_target_indices = []
		for target_ind in range(num_targ):
			if(not(target_ind in hidden_state_possibilities[sampled_index].living_target_indices)):
				dead_target_indices.append(target_ind)

		return (c, dead_target_indices, normalization)


	def sample_data_assoc_and_death_mult_meas_per_time_proposal_distr_1(self, measurement_list):
		"""
		Input:
		- measurement_list: a list of all measurements from the current time instance
		Output:
		- measurement_associations: A list of association values for each measurement.  Values of c correspond to:
			c[i] = -1 -> ith measurement is clutter
			c[i] = self.targets.living_count -> ith measurement is a new target
			c[i] in range [0, self.targets.living_count-1] -> ith measurement is of
				particle.targets.living_targets[c[i]]

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

		(targets_to_kill, measurement_associations, proposal_probability) = self.sample_proposal_distr1(measurement_list, self.targets.living_count)

		living_target_indices = []
		for i in range(self.targets.living_count):
			if(not i in targets_to_kill):
				living_target_indices.append(i)

		exact_probability = self.get_exact_prob_hidden_and_data(measurement_list, living_target_indices, self.targets.living_count, 
												 measurement_associations, p_target_deaths)

		assert(num_targs == self.targets.living_count)
		#double check targets_to_kill is sorted
		assert(all([targets_to_kill[i] <= targets_to_kill[i+1] for i in xrange(len(targets_to_kill)-1)]))

		imprt_re_weight = exact_probability/proposal_probability

		return (measurement_associations, targets_to_kill, imprt_re_weight)

	def sample_proposal_distr1(self, measurement_list, total_target_count):
		"""
		Something weird about this: can get the same measurement associations by sampling in different orders
		and return different priors.  I think this may be OK, but think about. !!!!!!

		Input:
		- measurement_list: a list of all measurements from the current time instance
		- total_target_count: the number of living targets on the previous time instace

		Output:
		- targets_to_kill: a list of targets that have been sampled to die (not killed yet)
		- list_of_measurement_associations: list of associations for each measurement
		- proposal_probability: proposal probability of the sampled deaths and assocations
			
		"""
		all_measurement_associations = {} #all_measurement_associations[measurement_indx] = sampled_association_value
		proposal_dict = {}
		birth_assoc_count = 0
		clutter_assoc_count = 0
		proposal_probability = 1.0
		for meas_index, measurement in enumerate(measurement_list):
			#add measurement-target association probabilities to proposal distribution
			for target_index in range(total_target_count):
				proposal_dict[(meas_index, target_index)] = (self.memoized_assoc_likelihood(measurement, target_index),
															 P_TARGET_EMISSION / len(measurement_list))

			#add measurement-birth association probability to proposal distribution
			birth_prior = self.get_clutter_or_birth_proposal1_prior(birth_assoc_count, len(measurement_list), BIRTH_COUNT_PRIOR)
			proposal_dict[(meas_index, total_target_count)] = (p_birth_likelihood, birth_prior)
			#add measurement-clutter association probability to proposal distribution
			clutter_prior = self.get_clutter_or_birth_proposal1_prior(clutter_assoc_count, len(measurement_list), CLUTTER_COUNT_PRIOR)
			proposal_dict[(meas_index, -1)] = (p_clutter_likelihood, clutter_prior)

		for i in range(len(measurement_list)):
			#create list of proposal distribution probabilities and a list of corresponding 
			#(measurement_index, association_value) association tuples
			proposal_probabilities = []
			proposal_associations = []
			for association, (likelihood, prior) in proposal_dict.iteritems():
				proposal_probabilities.append(likelihood * prior)
				proposal_associations.append(association)
			#normalize to create a probability distribution
			proposal_probabilities = np.asarray(proposal_probabilities)
			proposal_probabilities = proposal_probabilities/float(np.sum(proposal_probabilities))

			#sample association
			assert(len(proposal_probabilities) == len(proposal_associations))
			assoc_index = np.random.choice(len(proposal_probabilities), p=proposal_probabilities)
			(sampled_measurement_ind, sampled_assoc_ind) = proposal_associations[assoc_index]
			(sampled_likelihood, sampled_prior) = proposal_dict[(sampled_measurement_ind, sampled_assoc_ind)]
			proposal_probability *= sampled_likelihood*sampled_prior
			assert(proposal_probability != 0.0), (proposal_probability, sampled_prior, len(measurement_list), sampled_assoc_ind, total_target_count, birth_assoc_count, clutter_assoc_count)
			all_measurement_associations[sampled_measurement_ind] = sampled_assoc_ind
			if(sampled_assoc_ind == total_target_count):
				birth_assoc_count += 1
			if(sampled_assoc_ind == -1):
				clutter_assoc_count += 1
			#remove invalid association values from the proposal distribution and update clutter/birth probabilities
			proposal_dict_keys_to_del = []
			for (meas_index, assoc_index), (likelihood, prior) in proposal_dict.iteritems():
				#delete further assocations with the same measurement
				if(meas_index == sampled_measurement_ind):
					proposal_dict_keys_to_del.append((meas_index, assoc_index))
				#delete further assocations with the same target
				elif(assoc_index >= 0 and assoc_index < total_target_count and assoc_index == sampled_assoc_ind):
					proposal_dict_keys_to_del.append((meas_index, assoc_index))
				#could speed this up by storing references to birth and clutter priors
				elif(sampled_assoc_ind == total_target_count and assoc_index == total_target_count): #update birth probabilities
					birth_prior = self.get_clutter_or_birth_proposal1_prior(birth_assoc_count, len(measurement_list), BIRTH_COUNT_PRIOR)
					proposal_dict[(meas_index, total_target_count)] = (p_birth_likelihood, birth_prior)
				elif(sampled_assoc_ind == -1 and assoc_index == -1): #update clutter probabilities
					clutter_prior = self.get_clutter_or_birth_proposal1_prior(clutter_assoc_count, len(measurement_list), CLUTTER_COUNT_PRIOR)
					proposal_dict[(meas_index, -1)] = (p_clutter_likelihood, clutter_prior)
			#actually delete these entries now that we done iterating over the dictionary
			for key in proposal_dict_keys_to_del:
				del proposal_dict[key]

		#sanity checks
		assert(len(all_measurement_associations) == len(measurement_list))
		#make sure all measurements have been associated
		for i in range(len(measurement_list)):
			assert(i in all_measurement_associations)
		#make sure all targets have been associated with 0 or 1 measurements
		for i in range(total_target_count):
			target_associations = sum(1 for x in all_measurement_associations.values() if (x == i))
			assert(target_associations == 0 or target_associations == 1)
		#done sanity checks

		list_of_measurement_associations = []
		for i in range(len(measurement_list)):
			list_of_measurement_associations.append(all_measurement_associations[i])

		#sample deaths of targets that were not associated with a measurement
		unassociated_targets = [] #targets that were not associated with a measurement
		for i in range(total_target_count):
			if(not i in list_of_measurement_associations):
				unassociated_targets.append(i)
		(total_death_prob, targets_to_kill) = self.sample_deaths_among_specific_targets(unassociated_targets)
		proposal_probability *= total_death_prob
		
		return (targets_to_kill, list_of_measurement_associations, proposal_probability)

	def get_clutter_or_birth_proposal1_prior(self, prv_assoc_count, meas_count, count_prior_distribution):
		"""
		doesn't really belong in Particle
		"""

		prior = 0.0
		for i in range(prv_assoc_count+1, len(count_prior_distribution)):
			prior += count_prior_distribution[i] * min(i - prv_assoc_count, meas_count - prv_assoc_count) \
					/ (meas_count - prv_assoc_count)
		return prior

	def get_exact_hidden_prior(self, measurement_list, living_target_indices, total_target_count,
									   measurement_associations, p_target_deaths):
		"""
		doesn't really belong in Particle


		Calculate p(deaths, associations, #measurements) = p(deaths)*p(associations, #measurements|deaths)
		Input:
		- measurement_list: a list of all measurements from the current time instance
		- living_target_indices: a list of indices of targets from last time instance that are still alive
		- total_target_count: the number of living targets on the previous time instace
		- measurement_associations: a list of association values for each measurement. Each association has the value
			of a living target index (index from last time instance), target birth (total_target_count), 
			or clutter (-1)
		- p_target_deaths: a list of length len(total_target_count) where 
			p_target_deaths[i] = the probability that target i has died between the last
			time instance and the current time instance

		Return:
		- p(deaths, associations, #measurements)

		*note* p(deaths)*p(associations, #measurements|deaths) is referred to as the prior, even though 
		the number of measurements is part of the data (or an observed variable)
		"""


		hidden_state = HiddenState(living_target_indices, total_target_count, len(measurement_list), 
				 				   measurement_associations, p_target_deaths, P_TARGET_EMISSION, 
								   BIRTH_COUNT_PRIOR, CLUTTER_COUNT_PRIOR)
		prior = hidden_state.total_prior
		return prior


	def get_exact_prob_hidden_and_data(self, measurement_list, living_target_indices, total_target_count,
									   measurement_associations, p_target_deaths):
		"""
		Calculate p(data, associations, #measurements, deaths) as:
		p(data|deaths, associations, #measurements)*p(deaths)*p(associations, #measurements|deaths)
		Input:
		- measurement_list: a list of all measurements from the current time instance
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


		hidden_state = HiddenState(living_target_indices, total_target_count, len(measurement_list), 
				 				   measurement_associations, p_target_deaths, P_TARGET_EMISSION, 
								   BIRTH_COUNT_PRIOR, CLUTTER_COUNT_PRIOR)
		prior = hidden_state.total_prior
		likelihood = 1.0
		assert(len(measurement_associations) == len(measurement_list))
		for meas_index, meas_association in enumerate(measurement_associations):
			if(meas_association == total_target_count): #birth
				likelihood *= p_birth_likelihood
			elif(meas_association == -1): #clutter
				likelihood *= p_clutter_likelihood
			else:
				assert(meas_association >= 0 and meas_association < total_target_count), (meas_association, total_target_count)
				likelihood *= self.memoized_assoc_likelihood(measurement_list[meas_index], \
											   				 meas_association)
		return prior*likelihood

	def memoized_assoc_likelihood(self, measurement, target_index):
		if((measurement[0][0], target_index) in self.assoc_likelihood_cache):
			return self.assoc_likelihood_cache[(measurement[0][0], target_index)]
		else:
			target = self.targets.living_targets[target_index]
			S = np.dot(np.dot(H, target.P), H.T) + R_default
			assert(target.x.shape == (2, 1))
	
			state_mean_meas_space = np.dot(H, target.x)
	
			distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)

			assoc_likelihood = distribution.pdf(measurement)
			self.assoc_likelihood_cache[(measurement[0][0], target_index)] = assoc_likelihood
			return assoc_likelihood

	def debug_target_creation(self):
		print
		print "Particle ", self.id_, "importance distribution:"
		print "pi_birth = ", self.pi_birth_debug, "pi_clutter = ", self.pi_clutter_debug, \
			"pi_targets = ", self.pi_targets_debug
		print "sampled association c = ", self.c_debug, "importance reweighting factor = ", self.imprt_re_weight_debug
		self.plot_all_target_locations()

	#@profile
	def update_particle_with_measurement(self, measurements, cur_time):
		"""
		Input:
		-measurements: if MULTIPLE_MEAS_PER_TIME = True this is a list of measurement arrays
			if MULTIPLE_MEAS_PER_TIME = False this is a single measurement array

		Debugging output:
		- new_target: True if a new target was created
		"""
		new_target = False #debugging

		if MULTIPLE_MEAS_PER_TIME:
			birth_value = self.targets.living_count

			if USE_EXACT_PROPOSAL_DISTRIBUTION:
				(measurement_associations, dead_target_indices, imprt_re_weight) = \
					self.sample_data_assoc_and_death_mult_meas_per_time(measurements)
			else:
				assert(USE_PROPOSAL_DISTRIBUTION_1 == True)
				(measurement_associations, dead_target_indices, imprt_re_weight) = \
					self.sample_data_assoc_and_death_mult_meas_per_time_proposal_distr_1(measurements)
			assert(len(measurement_associations) == len(measurements))
			self.importance_weight *= imprt_re_weight #update particle's importance weight
			#process measurement associations
			for meas_index, meas_assoc in enumerate(measurement_associations):
				#create new target
				if(meas_assoc == birth_value):
					self.create_new_target(measurements[meas_index], cur_time)
					new_target = True 
				#update the target corresponding to the association we have sampled
				elif((meas_assoc >= 0) and (meas_assoc < birth_value)):
					self.targets.living_targets[meas_assoc].kf_update(measurements[meas_index], cur_time)
				else:
					#otherwise the measurement was associated with clutter
					assert(meas_assoc == -1), ("meas_assoc = ", meas_assoc)
			#process target deaths
			#double check dead_target_indices is sorted
			assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
			#important to delete larger indices first to preserve values of the remaining indices
			for index in reversed(dead_target_indices):
				self.targets.kill_target(index)

			#checking if something funny is happening
			original_num_targets = birth_value
			num_targets_born = measurement_associations.count(birth_value)
			num_targets_killed = len(dead_target_indices)
			assert(self.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
			#done checking if something funny is happening

		else:
			if(SAMPLE_DEATH_INDEPENDENTLY):
				self.sample_target_deaths()

				#sample data association from targets
				(c, imprt_re_weight) = self.sample_data_assoc(measurements)
				#update the particles importance weight
				self.importance_weight *= imprt_re_weight

				#process c
				#create new target
				if(c == self.targets.living_count):
					self.create_new_target(measurements, cur_time)
					new_target = True 
		#			self.debug_target_creation(c, imprt_re_weight, pi_birth, pi_clutter, pi_targets)
				#update the target corresponding to the association we have sampled
				elif((c >= 0) and (c < self.targets.living_count)):
					self.targets.living_targets[c].kf_update(measurements, cur_time)

				else:
					#otherwise the measurement was associated with clutter
					assert(c == -1), ("c = ", c)

			else:

				#sample data association from targets
				(c, dead_target_ind, imprt_re_weight) = self.sample_data_assoc_and_death(measurements)
				#update the particles importance weight
				self.importance_weight *= imprt_re_weight

				#process c
				#create new target
				if(c == self.targets.living_count):
					self.create_new_target(measurements, cur_time)
					new_target = True 
		#			self.debug_target_creation(c, imprt_re_weight, pi_birth, pi_clutter, pi_targets)
				#update the target corresponding to the association we have sampled
				elif((c >= 0) and (c < self.targets.living_count)):
					self.targets.living_targets[c].kf_update(measurements, cur_time)

				else:
					#otherwise the measurement was associated with clutter
					assert(c == -1), ("c = ", c)

				#kill target if necessary
				if(dead_target_ind != -1):
					self.targets.kill_target(dead_target_ind)

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




#assumed that the Kalman filter prediction step has already been run for this
#target on the current time step
#RUN PREDICTION FOR ALL TARGETS AT THE BEGINNING OF EACH TIME STEP!!!
#@profile
def assoc_likelihood(measurement, target):
	S = np.dot(np.dot(H, target.P), H.T) + R_default
	assert(target.x.shape == (2, 1))

	state_mean_meas_space = np.dot(H, target.x)

	distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
	return distribution.pdf(measurement)

def normalize_importance_weights(particle_set):
	normalization_constant = 0.0
	for particle in particle_set:
		normalization_constant += particle.importance_weight
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

	assert(abs(weight_sum - 1.0) < .000001)
	return 1.0/n_eff


##@profile
#def run_rbpf(all_measurements):
#	"""
#	Input:
#	- all_measurements: A list of time instances, where each time instance
#		contains a time stamp and list of measurements, where each measurement
#		is a numpy array.
#	Output:
#	"""
#	particle_set = []
#	for i in range(0, N_PARTICLES):
#		particle_set.append(Particle(i))
#
#	prev_time_stamp = -1
#
#
#	#for displaying results
#	time_stamps = []
#	positions = []
#
#	iter = 1 # for plotting only occasionally
#	number_resamplings = 0
#	for (time_stamp, measurements_at_cur_time) in all_measurements:
#		for particle in particle_set:
#			#forget data associations from the previous time step
#			particle.clear_data_associations()
#			#update particle death probabilities
#			if(prev_time_stamp != -1):
#				particle.update_target_death_probabilities(time_stamp, prev_time_stamp)
#				#Run Kalman filter prediction for all living targets
#				for target in particle.targets.living_targets:
#					dt = time_stamp - prev_time_stamp
#					assert(abs(dt - default_time_step) < .00000001), (dt, default_time_step)
#					target.kf_predict(dt, time_stamp)
#		for measurement in measurements_at_cur_time:
#			new_target_list = [] #for debugging, list of booleans whether each particle created a new target
#			for particle in particle_set:
#				new_target = particle.update_particle_with_measurement(measurement, time_stamp)
#				new_target_list.append(new_target)
#			normalize_importance_weights(particle_set)
#			#debugging
#			if DEBUG:
#				assert(len(new_target_list) == N_PARTICLES)
#				for (particle_number, new_target) in enumerate(new_target_list):
#					if new_target:
#						print "\n\n -------Particle %d created a new target-------" % particle_number
#						for particle in particle_set:
#							particle.debug_target_creation()
#						plt.show()
#						break
#			#done debugging
#
#		if iter%100 == 0:
#			print iter
#			display_target_counts(particle_set, time_stamp)
#
#
#		if (get_eff_num_particles(particle_set) < N_PARTICLES/RESAMPLE_RATIO):
#			perform_resampling(particle_set)
#			print "resampled on iter: ", iter
#			number_resamplings += 1
#		prev_time_stamp = time_stamp
#
#
#		if iter%1499 == 1498:
#			max_imprt_weight = -1
#			for particle in particle_set:
#				if(particle.importance_weight > max_imprt_weight):
#					max_imprt_weight = particle.importance_weight
#			for particle in particle_set:
#				if(particle.importance_weight == max_imprt_weight):
#					particle.plot_all_target_locations()
#					plt.show()
#		iter+=1
#
#	print "resampling performed %d times" % number_resamplings


def run_rbpf_on_targetset(target_set):
	"""
	Measurement class designed to only have 1 measurement/time instance
	Input:
	- target_set: generated TargetSet containing generated measurements and ground truth
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
	for measurement_set in target_set.measurements:
		time_stamp = measurement_set.time
		measurements = measurement_set.val

		print "time_stamp = ", time_stamp, "living target count in first particle = ",\
		particle_set[0].targets.living_count
		for particle in particle_set:
			#update particle death probabilities
			if(prev_time_stamp != -1):
				particle.assoc_likelihood_cache = {} #clear likelihood cache
				particle.update_target_death_probabilities(time_stamp, prev_time_stamp)
				#Run Kalman filter prediction for all living targets
				for target in particle.targets.living_targets:
					dt = time_stamp - prev_time_stamp
					assert(abs(dt - default_time_step) < .00000001), (dt, default_time_step)
					target.kf_predict(dt, time_stamp)

		new_target_list = [] #for debugging, list of booleans whether each particle created a new target
		for particle in particle_set:
			new_target = particle.update_particle_with_measurement(measurements, time_stamp)
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

f = open("pickled_test_data.pickle", 'r')
ground_truth_ts = pickle.load(f)
f.close()
print '-'*80
print ground_truth_ts.measurements[0].time
print ground_truth_ts.measurements[1].time
print ground_truth_ts.measurements[2].time
print ground_truth_ts.measurements[3].time
estimated_ts = run_rbpf_on_targetset(ground_truth_ts)
#estimated_ts = cProfile.run('run_rbpf_on_targetset(ground_truth_ts)')

calc_tracking_performance(ground_truth_ts, estimated_ts)


#time_steps, x1, x2, x3, x4, x5, x6 = gen_orig_paper_data()
##time_steps, x1, x2, x3, x4, x5, x6 = gen_lines()
#all_measurements = convert_data_to_rbpf_format_single_meas_per_time_step(time_steps, [x2, x1, x3, x4, x5, x6])
##all_measurements = convert_data_to_rbpf_format_multiple_meas_per_time_step(time_steps, [x2, x1, x3, x4, x5, x6])
#plot_rbpf_formatted_data(all_measurements)
#
#run_rbpf(all_measurements)

