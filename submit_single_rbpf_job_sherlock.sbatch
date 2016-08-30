#!/bin/bash
if [ $NUM_PARTICLES -gt "100" ]; then
    TIME=30:00 
#    TIME=4:00:00 
elif [ $NUM_PARTICLES -gt "90" ]; then
    TIME=15:00 
#    TIME=1:00:00 
else
    TIME=6:00 
#    TIME=15:00 
fi
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=$10:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################


ml load python/2.7.5

#pip install virtualenv
#virtualenv venv
source venv/bin/activate

#pip install numpy
#pip install filterpy
#pip install munkres


echo $NUM_PARTICLES
echo $INCLUDE_IGNORED_GT
echo $INCLUDE_DONTCARE_IN_GT
echo $USE_REGIONLETS_AND_LSVM
echo $SORT_DETS_ON_INTERVALS

python rbpf_KITTI_det_scores.py $NUM_PARTICLES $INCLUDE_IGNORED_GT $INCLUDE_DONTCARE_IN_GT $USE_REGIONLETS_AND_LSVM $SORT_DETS_ON_INTERVALS $RUN_IDX $NUM_RUNS $SEQ_IDX $PERIPHERAL

deactivate