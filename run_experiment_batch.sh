NUM_RUNS=1
RUN_WITH_1600='False'

#lsvm_and_regionlets_with_score_intervals

SETUP=$(qsub -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=-1,NUM_RUNS=$NUM_RUNS,PERIPHERAL=setup setup_rbpf_python_venv.sh)
for i in `seq 1 $NUM_RUNS`;
do
	RUN=$(qsub -W depend=afterok:$SETUP -t 0-20 -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,PERIPHERAL=run setup_rbpf_python_venv.sh)
done
qsub -W depend=afterok:$RUN -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=-1,NUM_RUNS=$NUM_RUNS,PERIPHERAL=evaluate setup_rbpf_python_venv.sh
#qsub -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=-1,NUM_RUNS=$NUM_RUNS,PERIPHERAL=evaluate setup_rbpf_python_venv.sh

##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True setup_rbpf_python_venv.sh
##########
###########lsvm_and_regionlets_no_score_intervals
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=False setup_rbpf_python_venv.sh
##########
###########regionlets_only_with_score_intervals
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=True setup_rbpf_python_venv.sh
##########
###########regionlets_only_no_score_intervals
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=False,include_dontcare_in_gt=False,use_regionlets_and_lsvm=False,sort_dets_on_intervals=False setup_rbpf_python_venv.sh
##########
##########
###########lsvm_and_regionlets_include_ignored_gt
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=25,include_ignored_gt=True,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=True,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=True,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=True,include_dontcare_in_gt=False,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True setup_rbpf_python_venv.sh
##########
###########lsvm_and_regionlets_include_dontcare_in_gt
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=25,include_ignored_gt=False,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=False,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=False,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=False,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True setup_rbpf_python_venv.sh
##########
###########lsvm_and_regionlets_include_ignored_and_dontcare_in_gt
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=25,include_ignored_gt=True,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=100,include_ignored_gt=True,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
##########for i in `seq 1 $NUM_RUNS`;
##########do
##########	for j in `seq 0 20`;
##########	do
##########		qsub -q atlas -v num_particles=400,include_ignored_gt=True,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS,SEQ_IDX=$j setup_rbpf_python_venv.sh
##########	done
##########done
##########
###########qsub -q atlas -v num_particles=1600,include_ignored_gt=True,include_dontcare_in_gt=True,use_regionlets_and_lsvm=True,sort_dets_on_intervals=True setup_rbpf_python_venv.sh
##########