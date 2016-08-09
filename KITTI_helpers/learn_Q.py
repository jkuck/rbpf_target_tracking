import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal

#Number of iterations to run EM 
EM_ITERS = 3

meas_sigma = 3
default_time_step = 0.1
NUM_TIME_STEPS = 100

#Kalman filter defaults
#P_default = meas_sigma**2 #this may not be a reasonable/valid thing to do
P_default = np.array([[40.64558317, 0, 			 0, 0, 0, 0],
 					  [0,          10, 			 0, 0, 0, 0],
 					  [0, 			0,  5.56278505, 0, 0, 0],
 					  [0, 			0, 			 0, 3, 0, 0],
 					  [0, 			0, 			 0, 0, 90, 0],
 					  [0, 			0, 			 0, 0, 0, 20]])

#P_default = np.array([[ 4888.46193689,  3252.85225574,  -139.60283818,  -684.35206995],
# [ 3252.85225574,  2165.89830482,  -103.43734755,  -462.74360811],
# [ -139.60283818,  -103.43734755,    83.16469482,    74.85882445],
# [ -684.35206995,  -462.74360811,    74.85882445,   134.44922666]])

#P_default = np.array([[ 2.45361284,  2.1363416 ,  1.12297189,  1.51060596],
# [ 2.1363416 ,  2.62559297,  1.15038   ,  1.41361091],
# [ 1.12297189,  1.15038   ,  1.37026071,  1.55870007],
# [ 1.51060596,  1.41361091,  1.55870007,  2.23242019]])

#P_default = np.array([[0, 0],
# 					  [0, 0]])

#P_default = np.array([[100,  0],
# 					  [  0, 10]])

#R_default = np.array([[meas_sigma**2]])
R_default = np.array([[0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
					  [0.0, 0.0, 0.0, 0.0]])
#Q_default_gt = np.array([[ 0.04227087,  0.02025365],
# 					  [ 0.02025365,  0.00985709]])
#Q_default_gt = Q_default_gt/20.0 #just from testing seems to give good RMSE with clutter

#process_noise_spectral_density = .1
#Q_default_gt = np.array([[1.0/3.0*default_time_step**3, 1.0/2.0*default_time_step**2],
# 					  [1.0/2.0*default_time_step**2, default_time_step]])
#Q_default_gt *= process_noise_spectral_density
Q_default_gt = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535, 0.0, 0.0],
 					  [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621, 0.0, 0.0],
 					  [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108, 0.0, 0.0],
 					  [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314, 0.0, 0.0],
 					  [  		 0.0,           0.0,           0.0,           0.0, 30.0, 30.0],
 					  [  		 0.0,           0.0,           0.0,           0.0, 30.0, 30.0],])

#initial estimate of Q when trying to learn from data
#Q_init_estimate = np.array([[.50, .0050],
# 					     	[.0050, .0050]])
#Q_init_estimate = np.array([[1.0, 1.0],
#                         [1.0, 1.0]])
#Q_init_estimate = np.array([[1.0, 0.0, 0.0, 0.0],
#                            [0.0, 1.0, 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]])

Q_init_helper = np.random.rand(6,6)
Q_init_estimate = np.dot(Q_init_helper.T, Q_init_helper)
#Q_init_estimate = Q_default_gt

#process_noise_spectral_density = .1
#Q_default_gt = np.array([[1.0/3.0*default_time_step**3, 1.0/2.0*default_time_step**2],
# 					  [1.0/2.0*default_time_step**2, default_time_step*100]])
#Q_default_gt *= process_noise_spectral_density

#measurement function matrix
H = np.array([[1.0,  0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0,  0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 0.0, 1.0]])	




class Target:
	def __init__(self, measurement):
		self.x = np.array([[measurement[0]], [0], [measurement[1]], [0], [measurement[2]], [measurement[3]]])
		self.P = P_default

		assert(self.x.shape == (6, 1))
		#Time of the last measurement data association with this target
		self.death_prob = -1 #calculate at every time instance

		self.all_states = [self.x]

		self.all_predicted_states = []
		self.all_updated_states = []

		self.all_predicted_P = []

		self.measurements = []
		self.measurement_time_stamps = []

	def print_info(self):
		print '-'*80
		print "x = "
		print self.x
		print "P = "
		print self.P
		print '-'*80

	def kf_update(self, measurement):
		""" Perform Kalman filter update step and replace predicted position for the current time step
		with the updated position in self.all_states
		Input:
		- measurement: the measurement (numpy array)
!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
		"""
		assert(self.x.shape == (6, 1))
		S = np.dot(np.dot(H, self.P), H.T) + R_default
		K = np.dot(np.dot(self.P, H.T), inv(S))
		residual = measurement - np.dot(H, self.x)
		assert(np.dot(H, self.x).shape == (4,1)), (np.dot(H, self.x).shape, H.shape, self.x.shape, measurement.shape)
		updated_x = self.x + np.dot(K, residual)
	#	updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
		updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
		self.x = updated_x
		self.P = updated_P
		assert(self.x.shape == (6, 1)), (self.x.shape, np.dot(K, residual).shape, measurement.shape, S.shape, K.shape, residual.shape, np.dot(H, self.x).shape)

		self.all_states[-1] = self.x
		self.all_updated_states.append(self.x)

	def kf_predict(self, dt, Q_estimate = Q_default_gt):
		"""
		Run kalman filter prediction on this target
		Inputs:
			-dt: time step to run prediction on
		"""
		F = np.array([[1.0,  dt, 0.0, 0.0, 0.0, 0.0],
		      		  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0,  dt, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
		x_predict = np.dot(F, self.x)
		P_predict = np.dot(np.dot(F, self.P), F.T) + Q_estimate
		self.x = x_predict
		self.P = P_predict
		self.all_states.append(self.x)
		self.all_predicted_states.append(self.x)
		self.all_predicted_P.append(self.P)
		assert(self.x.shape == (6, 1))
#		print self.P


	def data_gen_update_state(self):
		dt = default_time_step
		F = np.array([[1.0,  dt, 0.0, 0.0, 0.0, 0.0],
		      		  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0,  dt, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
		process_noise = np.random.multivariate_normal(np.zeros(Q_default_gt.shape[0]), Q_default_gt)
		process_noise = np.expand_dims(process_noise, axis=1)
		self.x = np.dot(F, self.x) + process_noise 
		self.all_states.append(self.x)

	def data_gen_measure_state(self):
		measurement_noise = np.random.multivariate_normal(np.zeros(R_default.shape[0]), R_default)
		measurement_noise = np.expand_dims(measurement_noise, axis=1)
		measurement = np.dot(H, self.x) + measurement_noise
		self.measurements.append(measurement)
		print "measurement.shape = ", measurement.shape
		return measurement

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def run_kf(Q_estimate, generative_target, initialize_with_first_gt_measurement = True, meas_0=np.array([0, 0])):
    """
    """
    assert (initialize_with_first_gt_measurement == True) #fix code if otherwise desired
    if initialize_with_first_gt_measurement:
    	estimated_target = Target(np.squeeze(generative_target.measurements[0]))
    	estimated_target
    else:
	    estimated_target = Target(meas_0)

    # run the kalman filter with etimated Q
    x_posteriors, x_predictions, cov = [], [], []
#    for meas in generative_target.measurements:
#        estimated_target.kf_predict(default_time_step, Q_estimate)
#        estimated_target.kf_update(meas)
    assert(len(generative_target.measurements) == len(generative_target.measurement_time_stamps))
    for index in range(len(generative_target.measurements)-1):
    	dt = generative_target.measurement_time_stamps[index + 1] - generative_target.measurement_time_stamps[index]
        if(abs(dt - default_time_step) > .00000001):
        	print dt
        estimated_target.kf_predict(default_time_step, Q_estimate)
        estimated_target.kf_update(generative_target.measurements[index + 1])

#    estimated_x_positions = np.array([estimated_target.all_predicted_states[i][0][0] for i in range(len(estimated_target.all_predicted_states))])
#    estimated_y_positions = np.array([estimated_target.all_predicted_states[i][2][0] for i in range(len(estimated_target.all_predicted_states))])
#    ground_truth_x_measurements = np.array([generative_target.measurements[i][0][0] for i in range(len(generative_target.measurements))])
#    ground_truth_y_measurements = np.array([generative_target.measurements[i][1][0] for i in range(len(generative_target.measurements))])
#    assert(len(ground_truth_x_measurements) == len(estimated_x_positions)), (len(ground_truth_x_measurements), len(estimated_x_positions))
#    time_stamps = np.array([i*default_time_step for i in range(NUM_TIME_STEPS)])
#    assert(len(ground_truth_measurements) == len(time_stamps))
#    assert(len(estimated_positions) == len(time_stamps))

#    cmap = get_cmap(100)
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)
#    ax.scatter(estimated_x_positions, estimated_y_positions, marker = '+', c = cmap(1))
#    ax.scatter(ground_truth_x_measurements, ground_truth_y_measurements, marker = '*', c = cmap(99))
#    plt.show()

    return estimated_target

def calc_LL(x_estimates, estimate_cov, measurements):
    assert(len(x_estimates) == len(estimate_cov) and len(estimate_cov) == len(measurements)), (len(x_estimates), len(estimate_cov), len(measurements))
    LL = 0
#    print "len(estimate_cov) = ", len(estimate_cov)
#    print "estimate_cov.shape = ", estimate_cov.shape
    for i in range(0, len(estimate_cov)):
#        print "shape H = ", H.shape
#        print "shape P[i] = ", estimate_cov[i].shape
#        print "shape R_default = ", R_default.shape
        S = np.dot(np.dot(H, estimate_cov[i]), H.T) + R_default
        state_posterior_in_meas_space = np.dot(H, x_estimates[i])
        #print S.shape
        #print S
        #print state_posterior_in_meas_space.shape
        #print np.squeeze(state_posterior_in_meas_space)
        #print "!"*80
        distribution = multivariate_normal(mean=np.squeeze(state_posterior_in_meas_space), cov=S)
        likelihood = distribution.pdf(np.squeeze(measurements[i]))
        #assert(likelihood > 0), (likelihood, measurements[i], state_posterior_in_meas_space, S)
        if(likelihood > 0):
        	LL += np.log(likelihood)
    return LL

def m_step(x_predictions, x_posteriors, measurements):
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

def run_EM_on_Q_single_target(Q_estimate, generative_target):
	log_likelihoods = []
	for i in range(0,EM_ITERS):
		print "iteration ", i, ":"
		print "Q_estimate = ", Q_estimate
		estimated_target = run_kf(Q_estimate, generative_target, x0)
		log_likelihoods.append(calc_LL(estimated_target.all_predicted_states, estimated_target.all_predicted_P, generative_target.measurements))
		print np.asarray(estimated_target.all_predicted_states).shape
		print np.asarray(estimated_target.all_updated_states).shape
		print np.asarray(generative_target.measurements).shape


#        use_measured_positions_as_truth = np.asarray(np.squeeze(estimated_target.all_updated_states))
#        for i in range(len(use_measured_positions_as_truth)):
#        	use_measured_positions_as_truth[i][0] = generative_target.measurements[i]
#        Q_estimate, R = m_step(np.asarray(np.squeeze(estimated_target.all_predicted_states)), \
#        					   use_measured_positions_as_truth, \
#        					   np.asarray(np.squeeze(generative_target.measurements)))

#        print "!!!!!!!", np.asarray(np.squeeze(estimated_target.all_updated_states)).shape
#
#        F = np.array([[1, default_time_step],
#              		  [0,  1]])
#        test_EM_pred = np.zeros((len(generative_target.measurements), 2))
#        for i in range(len(test_EM_pred)):
#        	test_EM_pred[i] = np.squeeze(np.dot(F, generative_target.all_states[i]))
#        Q_estimate, R = m_step(np.asarray(test_EM_pred), \
#        					   np.asarray(np.squeeze(generative_target.all_states[1:])), \
#        					   np.asarray(np.squeeze(generative_target.measurements)))
		print '#'*20
		print np.squeeze(estimated_target.all_predicted_states).shape
		print np.squeeze(estimated_target.all_updated_states).shape
		print '#'*20
		Q_estimate, R = m_step(np.asarray(np.squeeze(estimated_target.all_predicted_states)), \
							   np.asarray(np.squeeze(estimated_target.all_updated_states)), \
							   np.asarray(np.squeeze(generative_target.measurements)))
	return log_likelihoods

def run_EM_on_Q_multiple_targets(all_gt_targets):
	"""
	Input:
	- Q_estimate: initial estimate of the process noise covariance matrix Q
	- all_targets: a list of ground truth target measurements
	"""
	log_likelihoods_by_iter = []
	Q_estimate = Q_init_estimate
	for i in range(0,EM_ITERS):
		print "iteration ", i, ":"
		print "Q_estimate = ", Q_estimate
		estimated_targets = []
		cur_iter_LL = 0

		target_idx = 0
		number_targets_zero_states = 0
		for cur_gt_target in all_gt_targets:
			cur_estimated_target = run_kf(Q_estimate, cur_gt_target, initialize_with_first_gt_measurement=True)
			cur_iter_LL += calc_LL(cur_estimated_target.all_predicted_states, cur_estimated_target.all_predicted_P, cur_gt_target.measurements[1:])
			if target_idx == 0:
				all_targets_predicted_states = np.asarray(np.squeeze(cur_estimated_target.all_predicted_states))
				all_targets_updated_states = np.asarray(np.squeeze(cur_estimated_target.all_updated_states))
				all_targets_gt_measurements = np.asarray(np.squeeze(cur_gt_target.measurements[1:]))
			elif len(np.asarray(np.squeeze(cur_estimated_target.all_predicted_states)).shape) == 2:
#				print "target_idx = ", target_idx
#				print all_targets_predicted_states.shape
#				print np.asarray(np.squeeze(cur_estimated_target.all_predicted_states)).shape
#				print '-'*80
				all_targets_predicted_states = np.concatenate((all_targets_predicted_states, 
															   np.asarray(np.squeeze(cur_estimated_target.all_predicted_states))),
															  axis=0)
				all_targets_updated_states = np.concatenate((all_targets_updated_states,
															 np.asarray(np.squeeze(cur_estimated_target.all_updated_states))),
														 	axis=0)
				all_targets_gt_measurements = np.concatenate((all_targets_gt_measurements,
															  np.asarray(np.squeeze(cur_gt_target.measurements[1:]))),
															 axis=0)
			else:
				number_targets_zero_states+=1
			target_idx += 1

		print "during Q EM, number_targets_zero_states = ", number_targets_zero_states
		log_likelihoods_by_iter.append(cur_iter_LL)
		Q_estimate, R = m_step(all_targets_predicted_states, \
							   all_targets_updated_states, \
							   all_targets_gt_measurements)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter([i for i in range(0, len(log_likelihoods_by_iter))], log_likelihoods_by_iter, marker = '+')
	plt.show()

	return (log_likelihoods_by_iter, Q_estimate)
#target containing ground truth states and generated measurements
#generative_target = Target(np.array([0, 0]))
#for i in range(NUM_TIME_STEPS):
#	generative_target.data_gen_update_state()
#	generative_target.data_gen_measure_state()
#
#run_kf(Q_init_estimate, generative_target)

###################generative_target_list = []
###################for num_targets in range(3):
###################	cur_generative_target = Target(np.array([0, 0]))
###################	for i in range(NUM_TIME_STEPS):
###################		cur_generative_target.data_gen_update_state()
###################		cur_generative_target.data_gen_measure_state()
###################	generative_target_list.append(cur_generative_target)
###################
###################log_likelihoods = run_EM_on_Q_multiple_targets(generative_target_list)
###################print log_likelihoods

#target = Target(3)
#target.print_info()
#
#target.kf_predict(1)
#target.print_info()
#target.kf_update(4)
#target.print_info()
#
#target.kf_predict(1)
#target.print_info()
#target.kf_update(5)
#target.print_info()
#
#target.kf_predict(1)
#target.print_info()
#target.kf_update(6)
#target.print_info()

