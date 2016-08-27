#!/bin/bash
if [ $NUM_PARTICLES -gt "100" ]; then
#    TIME=20:00 
    TIME=4:00:00 
elif [ $NUM_PARTICLES -gt "90" ]; then
#    TIME=10:00 
    TIME=1:00:00 
else
#    TIME=3:00 
    TIME=15:00 
fi
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=12:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################


ml load python/2.7.5

pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install numpy
pip install filterpy
pip install munkres

cd /home/kuck/rbpf_target_tracking

python rbpf_KITTI_det_scores.py
