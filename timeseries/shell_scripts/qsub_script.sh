#!/bin/bash

#$ -cwd -V
# -l h_rt=00:05:00
#AIS:
# -l h_rt=10:00:00
#ASE:
#$ -l h_rt=03:00:00
# -l nodes=1 ##need to request processes on same node so I got access to local storage
#$ -pe smp 12
#$ -l h_vmem=5G


export OMP_NUM_THREADS=1

source ${CLUSTER_CONDA_CONFIG}
conda activate fracdiff_maps

#directory of daily mosaics
mosaic_directory=""

ulx=$1
uly=$2
lrx=$3
lry=$4
job_proc_index=$5
spacing=$6
buffer=$7

date

python ../python_scripts/create_data_for_fracture_change_map_arc_lr.py $ulx $uly $lrx $lry ${job_proc_index} ${mosaic_directory} ${spacing} ${buffer}

date
