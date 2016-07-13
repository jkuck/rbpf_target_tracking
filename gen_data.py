import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import gdtrc
import random
import copy 





#default time between succesive measurement time instances (in seconds)
default_time_step = .01 
num_time_steps = 1500

#data generation parameters
process_noise_spectral_density = .1
Q_default = np.array([[1.0/3.0*default_time_step**3, 1.0/2.0*default_time_step**2],
 					  [1.0/2.0*default_time_step**2, default_time_step]])
Q_default *= process_noise_spectral_density

#process_noise_spectral_density = .1
#Q_default = np.array([[1.0/3.0*default_time_step**3, 1.0/2.0*default_time_step**2],
# 					  [1.0/2.0*default_time_step**2, default_time_step*100]])
#Q_default *= process_noise_spectral_density

print "Q = ", Q_default

meas_sigma = .2
R_default = np.array([[meas_sigma**2]])

F_default = np.array([[1, default_time_step],	# Transition matrix
		      		  [0,  				  1]])
H = np.array([[1,  0]])     			# Measurement function
P_default = np.array([[meas_sigma**2, 0],
 					  [0,             1]])

#Gamma distribution parameters for calculating target death probabilities
alpha_death = 2.0
beta_death = 1.0
theta_death = 1.0/beta_death

p_birth_prior = 0.0025 #prior probability of a target birth

p_clutter_prior = .01 #prior probability of clutter

#state parameters, uniformly sample new targets from range:
min_pos = -5.0
max_pos = 5.0
min_vel = -1.0
max_vel = 1.0


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

		self.birth_time = cur_time
		#Time of the last measurement data association with this target
		self.last_measurement_association = cur_time
		self.id_ = id_ #named id_ to avoid clash with built in id
		self.death_prob = -1 #calculate at every time instance

		self.is_alive = True

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

		S = np.dot(np.dot(H, self.P), H.T) + R_default
		K = np.dot(np.dot(self.P, H.T), inv(S))
		residual = measurement - np.dot(H, self.x)
		updated_x = self.x + np.dot(K, residual)
	#	updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
		updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
		self.x = updated_x
		self.P = updated_P
		assert(self.all_time_stamps[-1] == cur_time and self.all_time_stamps[-2] != cur_time)
		self.all_states[-1] = self.x

	def kf_predict(dt, cur_time):
		"""
		Run kalman filter prediction on this target
		Inputs:
			-dt: time step to run prediction on
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

	def data_gen_update_state(self, cur_time, F):
		process_noise = np.random.multivariate_normal(np.zeros(Q_default.shape[0]), Q_default)
		process_noise = np.expand_dims(process_noise, axis=1)
		self.x = np.dot(F, self.x) + process_noise 
		self.all_states.append(self.x)
		self.all_time_stamps.append(cur_time)

	def data_gen_measure_state(self, cur_time):
		measurement_noise = np.random.multivariate_normal(np.zeros(R_default.shape[0]), R_default)
		measurement_noise = np.expand_dims(measurement_noise, axis=1)
		measurement = np.dot(H, self.x) + measurement_noise
		self.measurements.append(measurement)
		self.measurement_time_stamps.append(cur_time)
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
	def __init__(self, val = 0, time = -1):
		self.val = val
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
		self.measurements = []

	#define equality, copied from http://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
	#maybe double check this is correct sometime
	#this doesn't actually check if two different instances have the same content
	def __eq__(self, other):
		return (isinstance(other, self.__class__)
			and self.__dict__ == other.__dict__)

	def __str__(self):
		return "Living Target Count = %d Total Target Count = %d" % (self.living_count, self.total_count)

	def kill_targets(self, cur_time, prev_time):
		"""
		Kill every target with probability given by the death model between prev_time and cur_time
		"""
		assert(len(self.living_targets) == self.living_count)
		target_indices_to_kill = []
		for i in range(self.living_count):
			death_prob = self.living_targets[i].target_death_prob(cur_time, prev_time)
			if(random.random() < death_prob): 
				target_indices_to_kill.append(i)
		#kill targets here
		#reverse indices to delete targets with larger indices first (preserving
		# the correct indices for remaining targets to kill)
		for index_to_kill in reversed(target_indices_to_kill):
			self.living_targets[index_to_kill].is_alive = False
			del self.living_targets[index_to_kill]
		self.living_count -= len(target_indices_to_kill)
		assert(len(self.living_targets) == self.living_count)

	def create_target(self, cur_time):
		new_target = Target(cur_time, self.total_count)
		self.living_targets.append(new_target)
		self.all_targets.append(new_target)
		self.living_count += 1
		self.total_count += 1

	def move_targets(self, cur_time, prev_time):
		"""
		Move living targets according to the motion model for a time step of cur_time - prev_time
		"""
		dt = cur_time - prev_time
		F= np.array([[1, dt],	# Transition matrix
		      		 [0,  1]])

		for target in self.living_targets:
			target.data_gen_update_state(cur_time, F)

	def create_measurement(self, index, cur_time):
		"""
		Create a measurement of a current target, newborn target, or clutter.  First create a new target if
		the measurement is of a newborn target.  

		0 <= index < self.living_count corresponds to a measurement of the indicated current target
		index = self.living_count corresponds to a measurement of a new target
		index = self.living_count + 1 corresponds to a measurement of clutter
		"""
		cur_meas = Measurement(time = cur_time)
		if(index == self.living_count + 1): #clutter
			cur_meas.val = np.random.uniform(min_pos,max_pos,R_default.shape[0])
			cur_meas.val = np.expand_dims(cur_meas.val, axis=1)

		elif(index == self.living_count): #newborn target
			self.create_target(cur_time)
			cur_meas.val = self.living_targets[index].data_gen_measure_state(cur_time)

		else: #current target
			assert(index >= 0 and index < self.living_count)
			cur_meas.val = self.living_targets[index].data_gen_measure_state(cur_time)

		self.measurements.append(cur_meas)

	def save_data(self, ground_truth_file, measurements_file):
		"""
		Save the generated data.  Ground truth data will be saved with the tab separated format:

		time_stamp \t living_target_count \t target_state_size \t target_0_id \t target_0_state[0] 
		\t ... \t target_0_state[target_state_size-1] \t ... \t target_(living_target_count-1)_id
		\t target_(living_target_count-1)_state[0] \ttarget_(living_target_count-1)_state[target_state_size-1]

		Where living_target_count is the number of targets alive during the current time instance and
		target_state_size is the number of dimensions in the target state space.  If the state space has
		2 dimensions (position and velocity) and there are 3 living targets with id's 0 and 3 and 5
		a line would look like:

		.03	3	2	0	1.23	.41	3	4.2 -1.2	5	2.1	1.2

		Measurements will be saved with the tab separated format:

		time_stamp \t measurement_0_0 \t ... \t measurement_0_(meas_size - 1) 
		\t measurement_(meas_count - 1)_0 \t ... \t measurement_(meas_count - 1)_(meas_size - 1)

		Where meas_size is the number of dimensions in measurement space and meas_count is the number of
		measurements generated during the current time instance.

		"""
		return -1

	def pickle_data(self, pickle_file_name):
		f = open(pickle_file_name, 'w')
		pickle.dump(self, f)
		f.close()


def generate_1_meas_per_time_instance():
	"""
	Generate data with exactly one measurement every time instance.
	Each measurement may have originated from a currently living target,
	a newborn target, or clutter with probabilities:
		If there are no targets currently alive:
			p(newborn target) = p_birth_prior/(p_birth_prior + p_clutter_prior)
			p(clutter) = p_clutter_prior/(p_clutter_prior + p_clutter_prior)
		If there are currently living targets:
			p(target i) = (1-p_birth_prior-p_clutter_prior)/living_target_count (for i in [1,living_target_count])
			p(newborn target) = p_birth_prior
			p(clutter) = p_clutter_prior
	"""

	targets = TargetSet()

	for i in range(num_time_steps):
		for target in targets.living_targets:
			assert(target.is_alive)
		if(i != 0):
			#sample target deaths
			targets.kill_targets(i*default_time_step, (i-1)*default_time_step)
			#move all targets, according to motion model, for one time step
			targets.move_targets(i*default_time_step, (i-1)*default_time_step)

		#create distribution to sample measurement association
		association_distribution = []
		#assign uniform probability to association with any of the current targets
		for j in range(targets.living_count):
			association_distribution.append((1.0 - p_birth_prior - p_clutter_prior)/targets.living_count)
		association_distribution.append(p_birth_prior)
		association_distribution.append(p_clutter_prior)
		#normlized probability distribution if there are no living targets
		if(targets.living_count == 0):
			normalization = sum(association_distribution)
			for j in range(len(association_distribution)):
				association_distribution[j] = float(association_distribution[j])/normalization
		assert(abs(sum(association_distribution) - 1.0) < .00000001), (targets.living_count, association_distribution)

		# sample whether the measurement is a new target, clutter, or current target
		sampled_index = np.random.choice(len(association_distribution), p=association_distribution)
		targets.create_measurement(sampled_index, i*default_time_step)
		assert(len(targets.all_targets) == targets.total_count)

	return targets




targets = generate_1_meas_per_time_instance()
print '-'*80
print targets.measurements[0].time
print targets.measurements[1].time
print targets.measurements[2].time
print targets.measurements[3].time
print '-'*80

targets.pickle_data("pickled_test_data.pickle")
f = open("pickled_test_data.pickle", 'r')
loaded_targets = pickle.load(f)
f.close()

print targets
print loaded_targets

for key, value in targets.__dict__.items():
	if value != loaded_targets.__dict__[key]:
		print "not equal for ", key
		#print "targets value: ", targets.__dict__[key]
		#print "loaded_targets value: ", loaded_targets.__dict__[key]

for i in range(10):
	print "target's        ", i, "th measurement = ", targets.measurements[i].val, targets.measurements[i].val
	print "loaded_target's ", i, "th measurement = ", loaded_targets.measurements[i].val, loaded_targets.measurements[i].val



print isinstance(targets, loaded_targets.__class__)
print (targets.__dict__ == loaded_targets.__dict__)
print "Are targets and loaded targets equal? : ", (targets == loaded_targets)
loaded_targets.living_targets[0].x[0] += .03
print "Are targets and loaded targets equal after changing loaded_targets? : ", (targets == loaded_targets)





#print "number of targets = ", targets.living_count
#
#plottable_meas = []
#for meas in measurements:
#	plottable_meas.append(meas[0])
#
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(measurement_time_stamps, plottable_meas, 'o')
#plt.title('Measurements') # subplot 211 title
#plt.show()
#
#

