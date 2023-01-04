#!/bin/bash

#$ -cwd -V
#$ -l h_rt=00:20:00
#$ -pe smp 8
#$ -l h_vmem=3G

export OMP_NUM_THREADS=1

source ${CLUSTER_CONDA_CONFIG}
conda activate damage_env

d=$1

date

python ${SCRIPT_HOME}/mosaic/python_scripts/mosaic_region.py $d

date
