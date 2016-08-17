#lsvm_and_regionlets_with_score_intervals
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True

#lsvm_and_regionlets_no_score_intervals
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals False
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals False
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals False
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals False

#regionlets_only_with_score_intervals
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals True
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals True

#regionlets_only_no_score_intervals
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals False
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals False
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals False
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt False --include_dontcare_in_gt False --use_regionlets_and_lsvm False --sort_dets_on_intervals False


#lsvm_and_regionlets_include_ignored_gt
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt True --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt True --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt True --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt True --include_dontcare_in_gt False --use_regionlets_and_lsvm True --sort_dets_on_intervals True

#lsvm_and_regionlets_include_dontcare_in_gt
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt False --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt False --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt False --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt False --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True

#lsvm_and_regionlets_include_ignored_and_dontcare_in_gt
./submit_single_rbpf_job_sherlock.sh --num_particles 25 --include_ignored_gt True --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 100 --include_ignored_gt True --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
./submit_single_rbpf_job_sherlock.sh --num_particles 400 --include_ignored_gt True --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
#./submit_single_rbpf_job_sherlock.sh --num_particles 1600 --include_ignored_gt True --include_dontcare_in_gt True --use_regionlets_and_lsvm True --sort_dets_on_intervals True
