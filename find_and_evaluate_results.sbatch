#!/bin/bash

#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=3:00
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

search_directory='/scratch/users/kuck/rbpf_results/9_04_comparisons'

python KITTI_helpers/jdk_helper_evaluate_results.py $search_directory

deactivate