#!/bin/sh

#PBS -l select=1:ncpus=1:mem=7gb:cluster=galgo2

# Launch script:
# qsub -J 1-240 execute.sh

# Make sure that the script is run
# in the current working directory
cd $CWD

mamba activate bayes_torch2

python hc_nn.py -c $(sed -n ${PBS_ARRAY_INDEX}p arguments_linux.txt)
