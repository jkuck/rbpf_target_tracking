import subprocess
import os
import errno

NUM_RUNS=10
#SEQUENCES_TO_PROCESS = [i for i in range(21)]
SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [0]
#NUM_PARTICLES_TO_TEST = [25, 100]
NUM_PARTICLES_TO_TEST = [100]
DIRECTORY_OF_ALL_RESULTS = '/scratch/users/kuck/rbpf_results'
CUR_EXPERIMENT_BATCH_NAME = '100_particle_missing_result_test'
RUN_EVALUATION = False



def get_description_of_run(include_ignored_gt, include_dontcare_in_gt, 
						   use_regionlets_and_lsvm, sort_dets_on_intervals):
	if (not include_ignored_gt) and (not include_dontcare_in_gt) and use_regionlets_and_lsvm and sort_dets_on_intervals:
		description_of_run = "lsvm_and_regionlets_with_score_intervals"

	elif (not include_ignored_gt) and (not include_dontcare_in_gt) and use_regionlets_and_lsvm and (not sort_dets_on_intervals):
		description_of_run = "lsvm_and_regionlets_no_score_intervals"
		
	elif (not include_ignored_gt) and (not include_dontcare_in_gt) and (not use_regionlets_and_lsvm) and (sort_dets_on_intervals):
			description_of_run = "regionlets_only_with_score_intervals"

	elif (not include_ignored_gt) and (not include_dontcare_in_gt) and (not use_regionlets_and_lsvm) and (not sort_dets_on_intervals):
		description_of_run = "regionlets_only_no_score_intervals"



	elif (include_ignored_gt) and (not include_dontcare_in_gt) and use_regionlets_and_lsvm and sort_dets_on_intervals:
		description_of_run = "lsvm_and_regionlets_include_ignored_gt"

	elif (not include_ignored_gt) and (include_dontcare_in_gt) and use_regionlets_and_lsvm and sort_dets_on_intervals:
		description_of_run = "lsvm_and_regionlets_include_dontcare_in_gt"

	elif (include_ignored_gt) and (include_dontcare_in_gt) and use_regionlets_and_lsvm and sort_dets_on_intervals:
		description_of_run = "lsvm_and_regionlets_include_ignored_and_dontcare_in_gt"

	else:
		print "Unexpected combination of boolean arguments"
		sys.exit(1);

	return description_of_run

def run_complete(run_idx, seq_idx, num_particles, include_ignored_gt, include_dontcare_in_gt, 
	use_regionlets_and_lsvm, sort_dets_on_intervals):
	"""
	Output:
		- complete: bool, True if this run has been completed already
	"""
	description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
						   						use_regionlets_and_lsvm, sort_dets_on_intervals)
	results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
	results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)
	indicate_run_complete_filename = '%s/results_by_run/run_%d/seq_%d_done.txt' % (results_folder, run_idx, seq_idx)
	complete = os.path.isfile(indicate_run_complete_filename)
	return complete

def setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
					   	 use_regionlets_and_lsvm, sort_dets_on_intervals):
	description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
					   						use_regionlets_and_lsvm, sort_dets_on_intervals)
	results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
	results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)

	for cur_run_idx in range(1, NUM_RUNS + 1):
		file_name = '%s/results_by_run/run_%d/%s.txt' % (results_folder, cur_run_idx, 'random_name')
		if not os.path.exists(os.path.dirname(file_name)):
			try:
				os.makedirs(os.path.dirname(file_name))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

def submit_single_qsub_job(num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
	use_regionlets_and_lsvm=True, sort_dets_on_intervals=True, run_idx=-1, seq_idx=-1, peripheral='run'):
	if not run_complete(run_idx, seq_idx, num_particles, include_ignored_gt, include_dontcare_in_gt, 
						use_regionlets_and_lsvm, sort_dets_on_intervals):
#		args = ['qsub', '-q', 'atlas', '-l', 'nodes=1:ppn=1', '-v',
#				'num_particles=%d,include_ignored_gt=%s,'
#				'include_dontcare_in_gt=%s,use_regionlets_and_lsvm=%s,sort_dets_on_intervals=%s,'
#				'RUN_IDX=%d,NUM_RUNS=%d,SEQ_IDX=%d,PERIPHERAL=%s' %
#				(num_particles, include_ignored_gt, include_dontcare_in_gt, use_regionlets_and_lsvm, \
#				 sort_dets_on_intervals, run_idx, NUM_RUNS, seq_idx, peripheral), 
#		 		'setup_rbpf_python_venv.sh']
#		p = subprocess.Popen(args)


		command = 'sbatch --export=NUM_PARTICLES=%d,INCLUDE_IGNORED_GT=%s,INCLUDE_DONTCARE_IN_GT=%s,'\
				  'USE_REGIONLETS_AND_LSVM=%s,SORT_DETS_ON_INTERVALS=%s,RUN_IDX=%d,NUM_RUNS=%d,'\
				  'SEQ_IDX=%d,PERIPHERAL=%s submit_single_rbpf_job_sherlock.sbatch' \
		 		 % (num_particles, include_ignored_gt, include_dontcare_in_gt, use_regionlets_and_lsvm, \
				 sort_dets_on_intervals, run_idx, NUM_RUNS, seq_idx, peripheral)
		os.system(command)

#		args = ['run_experiment_interactive_mode.sh', '%d' % num_particles, '%s' % include_ignored_gt,\
#				'%s' % include_dontcare_in_gt, '%s' % use_regionlets_and_lsvm,\
#				'%s' % sort_dets_on_intervals, '%d' % run_idx, '%d' % NUM_RUNS,\
#				'%d' % seq_idx, '%s' % peripheral]
#		p = subprocess.Popen(args)

def submit_single_experiment(num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
	use_regionlets_and_lsvm=True, sort_dets_on_intervals=True):
	setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
					   	 use_regionlets_and_lsvm, sort_dets_on_intervals)
	for run_idx in range(1, NUM_RUNS+1):
		for seq_idx in SEQUENCES_TO_PROCESS:
			submit_single_qsub_job(num_particles=num_particles, include_ignored_gt=include_ignored_gt, 
				include_dontcare_in_gt=include_dontcare_in_gt, use_regionlets_and_lsvm=use_regionlets_and_lsvm, 
				sort_dets_on_intervals=sort_dets_on_intervals, run_idx=run_idx, seq_idx=seq_idx, peripheral='run')
	if RUN_EVALUATION:
		submit_single_qsub_job(num_particles=num_particles, include_ignored_gt=include_ignored_gt, 
				include_dontcare_in_gt=include_dontcare_in_gt, use_regionlets_and_lsvm=use_regionlets_and_lsvm, 
				sort_dets_on_intervals=sort_dets_on_intervals, run_idx=-1, seq_idx=-1, peripheral='evaluate')


if __name__ == "__main__":
	#lsvm_and_regionlets_with_score_intervals
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=False, include_dontcare_in_gt=False, 
								use_regionlets_and_lsvm=True, sort_dets_on_intervals=True)

	#lsvm_and_regionlets_no_score_intervals
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=False, include_dontcare_in_gt=False, 
								use_regionlets_and_lsvm=True, sort_dets_on_intervals=False)

	#regionlets_only_with_score_intervals
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=False, include_dontcare_in_gt=False, 
								use_regionlets_and_lsvm=False, sort_dets_on_intervals=True)

	#regionlets_only_no_score_intervals
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=False, include_dontcare_in_gt=False, 
								use_regionlets_and_lsvm=False, sort_dets_on_intervals=False)

	#lsvm_and_regionlets_include_ignored_gt
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=True, include_dontcare_in_gt=False, 
								use_regionlets_and_lsvm=True, sort_dets_on_intervals=True)

	#lsvm_and_regionlets_include_dontcare_in_gt
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=False, include_dontcare_in_gt=True, 
								use_regionlets_and_lsvm=True, sort_dets_on_intervals=True)

	#lsvm_and_regionlets_include_ignored_and_dontcare_in_gt
	for num_particles in NUM_PARTICLES_TO_TEST:
		submit_single_experiment(num_particles=num_particles, 
								include_ignored_gt=True, include_dontcare_in_gt=True, 
								use_regionlets_and_lsvm=True, sort_dets_on_intervals=True)

