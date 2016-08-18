#!/bin/bash
# Use -gt 1 to consume two arguments per pass in the loop (e.g. each
# argument has a corresponding value to go with it).
# Use -gt 0 to consume one or more arguments per pass in the loop (e.g.
# some arguments don't have a corresponding value to go with it such
# as in the --default example).
# note: if this is set to -gt 0 the /etc/hosts part is not recognized ( may be a bug )
while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -n|--num_particles)
    NUM_PARTICLES="$2"
    shift # past argument
    ;;
    -igt|--include_ignored_gt)
    INCLUDE_IGNORED_GT="$2"
    shift # past argument
    ;;
    -dcgt|--include_dontcare_in_gt)
    INCLUDE_DONTCARE_IN_GT="$2"
    shift # past argument
    ;;
    -rl|--use_regionlets_and_lsvm)
    USE_REGIONLETS_AND_LSVM="$2"
    shift # past argument
    ;;
    -sdi|--sort_dets_on_intervals)
    SORT_DETS_ON_INTERVALS="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

if [ $NUM_PARTICLES -gt "100" ]; then
    TIME=20:00:00 
elif [ $NUM_PARTICLES -gt "90" ]; then
    TIME=4:00:00 
else
    TIME=1:00:00 
fi


#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=$TIME
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

cd /afs/cs.stanford.edu/u/jkuck/rotation3/rbpf_target_tracking
python rbpf_KITTI_det_scores.py $NUM_PARTICLES $INCLUDE_IGNORED_GT $INCLUDE_DONTCARE_IN_GT $USE_REGIONLETS_AND_LSVM $SORT_DETS_ON_INTERVALS
