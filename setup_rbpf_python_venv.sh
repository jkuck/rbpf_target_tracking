#!/bin/bash

cd /atlas/u/jkuck/rbpf_target_tracking
#env >> ./job_env
#set >> ./job_env

#### Only need to run following two lines if script never run before!
#pip install virtualenv
#virtualenv venv
source venv/bin/activate

#pip install numpy
#pip install filterpy
#pip install munkres

#x='string' #just a random string
#if [ -z ${PBS_ARRAYID+x} ]; 
#then 
#	SEQ_IDX=-1
#else 
#	SEQ_IDX=$PBS_ARRAYID
#fi

python rbpf_KITTI_det_scores.py $num_particles $include_ignored_gt $include_dontcare_in_gt $use_regionlets_and_lsvm $sort_dets_on_intervals $RUN_IDX $NUM_RUNS $SEQ_IDX $PERIPHERAL

deactivate
#cd /atlas/u/jkuck/rbpf_target_tracking
#source venv/bin/activate
#python rbpf_KITTI_det_scores.py 100 False False True True 1 10 11 run

#qsub -I -q atlas -l nodes=atlas1.stanford.edu:ppn=1