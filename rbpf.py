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

N_particles = 100 #number of particles used in the particle filter

p_clutter = .9 #probability of associating a measurement with clutter
p_birth = .1 #probability of associating a measurement with a new target


#Kalman filter defaults
P_default = np.diag([10., 5., 10., 5.])
R_default = np.array([[ 0.04499246,  0.05549335, -0.17198677, -0.00532710],
 					  [ 0.05549335,  0.10692745, -0.29580029, -0.01258159],
 					  [-0.17198677, -0.29580029,  1.54644632, -0.00529713],
 					  [-0.00532710, -0.01258159, -0.00529713,  0.00544296]])
Q_default = np.array([[ 4.05478317e-04,5.53907275e-05,-2.31993904e-03,-5.80464091e-05],
 					  [ 5.53907275e-05,4.06586702e-04, 9.80342656e-03, 3.04183540e-04],
 					  [-2.31993904e-03,9.80342656e-03, 3.23430417e-01, 7.98461527e-03],
 					  [-5.80464091e-05,3.04183540e-04, 7.98461527e-03, 3.65214796e-04]])

#Gamma distribution parameters for calculating target death probabilities
alpha_death = 2.0
beta_death = 2.0

#default time between succesive measurement time instances (in seconds)
default_time_step = .09 
#for displaying results only show targets older than this (in seconds)
min_target_age = 3

class Target:
	def __init__(self, measurement, cur_time):
		#Targets tracked by this particle
		self.kf = pos_vel_filter(np.array([measurement[0],measurement[1],measurement[2],measurement[3]]),
								 P = P_default, R = R_default, Q=Q_default)

		#Time of the last measurement data association with this target
		self.last_measurement_association = cur_time
		self.birth_time = cur_time

	def process_measurement(self, measurement, time):
		""" Perform Kalman filter predict and update step
		Input:
		- measurement: the measurement (numpy array)
		- time: time when the measurement was taken (float)

		Return:
		- x_prediction: Kalman filter state prediction (4x1 numpy array)
		- x_posterior: Kalman filter posterior state after update with measurement
			(4x1 numpy array)
		- covariance: Kalman filter state covariance matrix after update
			(4x4 numpy array)
		"""

		#set the state transition matrix with the appropriate time interval dt
		dt = time - self.last_measurement_association
		self.last_measurement_association = time
		self.kf.F = np.array([[1, dt,  0,  0],
		                 	  [0,  1,  0,  0],
		                 	  [0,  0,  1, dt],
		                 	  [0,  0,  0,  1]])

		self.kf.predict()
		x_prediction = self.kf.x
		self.kf.update(measurement)
		x_posterior = self.kf.x
		covariance = self.kf.P

		return (x_prediction, x_posterior, covariance)

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
		#integral(gamma_dist(alpha = a, betta = b))from x to infinity
		last_assoc = self.last_measurement_association
		death_prob = gdtrc(beta_death, alpha_death, prev_time - last_assoc) \
				   - gdtrc(beta_death, alpha_death, cur_time - last_assoc)
		death_prob /= gdtrc(beta_death, alpha_death, prev_time - last_assoc)
		return death_prob

class Particle:
	def __init__(self):
		#Targets tracked by this particle
		self.targets = [] 
		#Previous measurement-target data associations for the current
		#time instance (all data associations necessary for future decisions)
		self.data_associations = [] 
		self.importance_weight = 1.0/N_particles

	def create_new_target(self, measurement, cur_time):
		self.targets.append(Target(measurement, cur_time))
		#associate measurement with newly created target
		self.data_associations.append(len(self.targets) - 1) 

	#call this function before processing the first measurement of a new time
	#step
	def clear_data_associations(self):
		self.data_associations = []

	#return p(c_k,l = target_index | c_k,1:l-1)
	def assoc_prior(self, target_index):
		if (target_index in self.data_associations):
			return 0.0
		elif (len(self.targets) == len(self.data_associations)):
			return 0.0
		else:
			return 1.0/(len(self.targets) - len(self.data_associations))

	def sample_target_deaths(self, cur_time, prev_time):
		"""
		Implemented to kill possibly kill multiple targets at once, seems
		reasonbale but CHECK TECHNICAL DETAILS!!

		Input:
		- cur_time: The current measurement time (float)
		- prev_time: The previous time step when a measurement was received (float)

		"""
		original_num_targets = len(self.targets)
		num_targets_killed = 0
		for cur_target in self.targets:
			death_prob = cur_target.target_death_prob(cur_time, prev_time)
			assert(death_prob < 1.0 and death_prob > 0.0)
			if (random.random() < death_prob):
				self.targets[:] = [target for target in self.targets if (target != cur_target)]
				num_targets_killed += 1
		assert(len(self.targets) == (original_num_targets - num_targets_killed))
		print "targets killed = ", num_targets_killed

	@profile
	def sample_data_assoc(self, measurement):
		"""
		Input:

		Output:
		- c: The measurement-target association value.  Values of c correspond to:
			c = -1 -> clutter
			c = len(particle.targets) -> new target
			c in range [0, len(particle.targets)-1] -> particle.targets[c]
		- normalization: After processing this measurement the particle's
			importance weight will be:
			new_importance_weight = old_importance_weight * normalization
		"""

		#I'm not multiplying p_clutter by 1/A (seems wrong to introduce units)
		#but possibly should multiply by something
		#Also not multiplying p_birth by p(y_k' | ...), this seems less important
		#Check sometime!!
		normalization = p_birth + (1.0-p_birth)*p_clutter

		#calculate the optimal importance distribution probabilities for associating
		#the measurement with each current target, a new target, or clutter
		pi_targets = [] #current targets
		pi_birth = p_birth #new target
		pi_clutter = (1.0-p_birth)*p_clutter #clutter

		for i in range(len(self.targets)):
			cur_target_prob = assoc_likelihood(measurement, self.targets[i]) \
						   * self.assoc_prior(i) * (1.0 - p_birth)
			normalization += cur_target_prob
			pi_targets.append(cur_target_prob)

		assert(len(pi_targets) == len(self.targets))
		for i in range(len(pi_targets)):
			pi_targets[i] /= normalization
		pi_birth /= normalization
		pi_clutter /= normalization

		assert((sum(pi_targets) + pi_birth + pi_clutter - 1.0) < .0000001)

		#now sample from the optimal importance distribution
		c = -99
		rand_val = random.random()
		if(rand_val < pi_birth):
			c = len(self.targets)
		else:
			rand_val -= pi_birth
			if(rand_val < pi_clutter):
				c = -1
			else:
				rand_val -= pi_clutter
				for i in range(0, len(pi_targets)):
					if(rand_val < pi_targets[i]):
						c = i
					else:
						rand_val -= pi_targets[i]
		assert((rand_val > 0.0) and (c != -99))
		return (c, normalization)

	@profile
	def update_particle_with_measurement(self, measurement, cur_time):
		#sample data assocation from targets
		(c, imprt_re_weight) = self.sample_data_assoc(measurement)

		#update the particles importance weight
		self.importance_weight *= imprt_re_weight

		#create new target
		if(c == len(self.targets)):
			self.create_new_target(measurement, cur_time)
		#update the target corresponding to the association we have sampled
		elif((c >= 0) and (c < len(self.targets))):
			self.targets[c].process_measurement(measurement, cur_time)
		else:
			#otherwise the measurement was associated with clutter
			assert(c == -1), ("c = ", c)

def pos_vel_filter(x_init, P, Q, R):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
	Input:
	- x_init: Numpy array of initial state. x_init = 
		[lat_position, lat_velocity, lng_positon, lng_velocity]
	- P: Initial state covariance matrix. Can be a numpy array or scalar.
	- Q: Process noise covariance matrix. Can be a numpy array or scalar.
	- R: Measurement noise covariance matrix. Can be a numpy array or scalar.

	*Note: When P, Q, or R are specified as scalars, the matrices are 
		initialized to the identity matrix multiplied by the scalar.

	Return:
	- kf: Kalman filter initialized with inputs
    """
    
    kf = KalmanFilter(dim_x=4, dim_z=4)
    kf.x = x_init 
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


#assumed that the Kalman filter prediction step has already been run for this
#target on the current time step
#RUN PREDICTION FOR ALL TARGETS AT THE BEGINNING OF EACH TIME STEP!!!
@profile
def assoc_likelihood(measurement, target):
	S = np.dot(np.dot(target.kf.H, target.kf.P), target.kf.H.T) + target.kf.R
	state_mean_meas_space = np.dot(target.kf.H, target.kf.x)
	distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
	return distribution.pdf(measurement)

def normalize_importance_weights(particle_set):
	normalization_constant = 0.0
	for particle in particle_set:
		normalization_constant += particle.importance_weight
	for particle in particle_set:
		particle.importance_weight /= normalization_constant


def perform_resampling(particle_set):
	assert(len(particle_set) == N_particles)
	weights = []
	for particle in particle_set:
		weights.append(particle.importance_weight)
	assert(sum(weights) - 1.0 < .0000001)

	new_particles = stratified_resample(weights)
	new_particle_set = []
	for index in new_particles:
		new_particle_set.append(copy.deepcopy(particle_set[index]))
	del particle_set[:]
	for particle in new_particle_set:
		particle.importance_weight = 1.0/N_particles
		particle_set.append(particle)
	assert(len(particle_set) == N_particles)
	#testing
	weights = []
	for particle in particle_set:
		weights.append(particle.importance_weight)
	assert(sum(weights) - 1.0 < .01), sum(weights)
	#done testing

def display_target_counts(particle_set, cur_time):
	target_counts = []
	for particle in particle_set:
		target_counts.append(len(particle.targets))
	print target_counts

	target_counts = []
	for particle in particle_set:
		cur_target_count = 0
		for target in particle.targets:
			if (cur_time - target.birth_time) > min_target_age:
				cur_target_count += 1
		target_counts.append(cur_target_count)
	print "targets older than ", min_target_age, "seconds: ", target_counts

def get_mature_target_positions(particle_set, cur_time):
	"""
	For displaying results, get positions of all targets older than 
	min_target_age as a list of lateral positions and a list of longitudinal 
	positions
	"""
	lat_pos = []
	lng_pos = []
	for particle in particle_set:
		for target in particle.targets:
			if (cur_time - target.birth_time) > min_target_age:
				lat_pos.append(target.kf.x[0])
				lng_pos.append(target.kf.x[2])

	return (lat_pos, lng_pos)


@profile
def run_rbpf(all_measurements):
	"""
	Input:
	- all_measurements: A list of time instances, where each time instance
		contains a time stamp and list of measurements, where each measurement
		is a numpy array.
	Output:
	"""
	particle_set = []
	for i in range(0, N_particles):
		particle_set.append(Particle())

	prev_time_stamp = -1

	num_measurement_processed = 0 #just to see how fast it's running

	#for displaying results
	lat_pos = []
	lng_pos = []

	iter = 0 # for plotting only occasionally

	for (time_stamp, measurements_at_cur_time) in all_measurements:
		print num_measurement_processed
		for particle in particle_set:
			#forget data associations from the previous time step
			particle.clear_data_associations()
			#sample particle deaths after the first time step
			if(prev_time_stamp != -1):
				particle.sample_target_deaths(time_stamp, prev_time_stamp)
		for measurement in measurements_at_cur_time:
			for particle in particle_set:
				particle.update_particle_with_measurement(measurement, time_stamp)
			normalize_importance_weights(particle_set)
			perform_resampling(particle_set)

		prev_time_stamp = time_stamp
		num_measurement_processed += 1
		display_target_counts(particle_set, time_stamp)

		#display results
		(cur_lat_pos, cur_lng_pos) = get_mature_target_positions(particle_set,
																 time_stamp)
		lat_pos += cur_lat_pos
		lng_pos += cur_lng_pos

		if iter%30 == 0:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			ax.scatter(lat_pos, lng_pos)
			plt.show()
		iter+=1

def read_radar_data(file_name):
	"""
	Input:
	- file_name: name of the text file containing the radar data

	Return:
	numpy arrays
	"""
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

def convert_radar_data(lat_pos, lng_pos, lat_vel, lng_vel):
	assert(lat_pos.shape[1] == 64 and lng_pos.shape[1] == 64)
	assert(lat_pos.shape[0] == lng_pos.shape[0])
	measurements = []
	time_stamp = 0
	for time_step in range(0, lat_pos.shape[0]):
#	for time_step in range(0, 3): #for shorter runtime timing
		cur_time_step_meas = []
#		for i in range (0, lat_pos.shape[1]):
#		for i in range (54, 55): #subset of measurements per timestep
		for i in range (50, 60): #subset of measurements per timestep
			if((not math.isnan(lat_pos[time_step, i])) and \
			   (not math.isnan(lat_vel[time_step, i])) and \
			   (not math.isnan(lng_pos[time_step, i])) and \
			   (not math.isnan(lng_vel[time_step, i]))):
				cur_time_step_meas.append(np.array([lat_pos[time_step, i] * 1.5,
													lat_vel[time_step, i],
													lng_pos[time_step, i] * 1.5,
													lng_vel[time_step, i] - 35.3]))
		if(len(cur_time_step_meas) > 0):
			measurements.append((time_stamp, cur_time_step_meas))

		time_stamp += default_time_step

	return measurements



(lat_pos, lng_pos, lat_vel, lng_vel) = read_radar_data('/Users/jkuck/rotation3/Ford-Stanford-Alliance-Stefano-Sneha/jdk_filters/utils/radar_data.txt')
measurements = convert_radar_data(lat_pos, lng_pos, lat_vel, lng_vel)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(lat_pos[:,54], lng_pos[:,54])
plt.show()

run_rbpf(measurements)
