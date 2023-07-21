#!/bin/sh

#PBS -l select=1:ncpus=1:mem=7gb:cluster=galgo2

# Launch script:
# qsub -J 1-240 -v CWD=$PWD execute.sh

# Make sure that the script is run
# in the current working directory
cd $CWD

# Ensure that mamba and conda are available
source /home/Luna.Jimenez/mambaforge-pypy3/etc/profile.d/conda.sh
source /home/Luna.Jimenez/mambaforge-pypy3/etc/profile.d/mamba.sh

mamba activate bayes_torch2

python hc_nn.py $(sed -n ${PBS_ARRAY_INDEX}p experiment_list.txt)
