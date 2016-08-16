#qsub -q atlas -v num_particles=25 setup_rbpf_python_venv.sh
#qsub -q atlas -v num_particles=100 setup_rbpf_python_venv.sh
#qsub -q atlas -v num_particles=400 setup_rbpf_python_venv.sh
qsub -q atlas -v num_particles=1600 setup_rbpf_python_venv.sh
