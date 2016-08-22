NUM_RUNS=100
RUN_WITH_1600='False'

#lsvm_and_regionlets_with_score_intervals
for i in `seq 1 $NUM_RUNS`;
do
	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
done

for i in `seq 1 $NUM_RUNS`;
do
	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
done

for i in `seq 1 $NUM_RUNS`;
do
	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
done

if [ $RUN_WITH_1600 == 'True' ]; then
	for i in `seq 1 $NUM_RUNS`;
	do
	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
	done
fi	

###############lsvm_and_regionlets_no_score_intervals
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############if [ $RUN_WITH_1600 == 'True' ]; then
##############	for i in `seq 1 $NUM_RUNS`;
##############	do
##############	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############	done
##############fi	
##############
###############regionlets_only_with_score_intervals
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############if [ $RUN_WITH_1600 == 'True' ]; then
##############	for i in `seq 1 $NUM_RUNS`;
##############	do
##############	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############	done
##############fi	
##############
###############regionlets_only_no_score_intervals
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############for i in `seq 1 $NUM_RUNS`;
##############do
##############	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############done
##############
##############if [ $RUN_WITH_1600 == 'True' ]; then
##############	for i in `seq 1 $NUM_RUNS`;
##############	do
##############	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=False,SORT_DETS_ON_INTERVALS=False,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##############	done
##############fi	

#lsvm_and_regionlets_include_ignored_gt
for i in `seq 1 $NUM_RUNS`;
do
	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
done

for i in `seq 1 $NUM_RUNS`;
do
	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
done

for i in `seq 1 $NUM_RUNS`;
do
	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
done

if [ $RUN_WITH_1600 == 'True' ]; then
	for i in `seq 1 $NUM_RUNS`;
	do
	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=False,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
	done
fi	

###################lsvm_and_regionlets_include_dontcare_in_gt
##################for i in `seq 1 $NUM_RUNS`;
##################do
##################	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################done
##################
##################for i in `seq 1 $NUM_RUNS`;
##################do
##################	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################done
##################
##################for i in `seq 1 $NUM_RUNS`;
##################do
##################	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################done
##################
##################if [ $RUN_WITH_1600 == 'True' ]; then
##################	for i in `seq 1 $NUM_RUNS`;
##################	do
##################	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=False,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################	done
##################fi	
##################
###################lsvm_and_regionlets_include_ignored_and_dontcare_in_gt
##################for i in `seq 1 $NUM_RUNS`;
##################do
##################	sbatch --export=NUM_PARTICLES=25,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################done
##################
##################for i in `seq 1 $NUM_RUNS`;
##################do
##################	sbatch --export=NUM_PARTICLES=100,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################done
##################
##################for i in `seq 1 $NUM_RUNS`;
##################do
##################	sbatch --export=NUM_PARTICLES=400,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################done
##################
##################if [ $RUN_WITH_1600 == 'True' ]; then
##################	for i in `seq 1 $NUM_RUNS`;
##################	do
##################	sbatch --export=NUM_PARTICLES=1600,INCLUDE_IGNORED_GT=True,INCLUDE_DONTCARE_IN_GT=True,USE_REGIONLETS_AND_LSVM=True,SORT_DETS_ON_INTERVALS=True,RUN_IDX=$i,NUM_RUNS=$NUM_RUNS submit_single_rbpf_job_sherlock.sbatch
##################	done
##################fi	