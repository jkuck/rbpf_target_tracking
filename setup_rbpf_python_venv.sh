#!/bin/bash

ml load python/2.7.5

pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install numpy
pip install filterpy
pip install munkres

cd /afs/cs.stanford.edu/u/jkuck/rotation3/rbpf_target_tracking
python rbpf_KITTI_det_scores.py $num_particles