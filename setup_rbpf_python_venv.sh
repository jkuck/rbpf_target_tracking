#!/bin/bash

#### Only run need to run following two lines if script never run before!
####pip install virtualenv
####virtualenv venv
source venv/bin/activate

pip install numpy
pip install filterpy
pip install munkres

cd /afs/cs.stanford.edu/u/jkuck/rotation3/rbpf_target_tracking
python rbpf_KITTI_det_scores.py $num_particles $include_ignored_gt $include_dontcare_in_gt $use_regionlets_and_lsvm $sort_dets_on_intervals $RUN_IDX $NUM_RUNS
