#!/bin/bash


# Arguments:
# $1: scene granule name
# $2: path
# $3: frame
# $4: date
# $5: temporary directory
# $6: save directory
# $7: download script name
# $8: bs script name
# $9: damage 1a script name
# $10: damage 1b script name
# $11: postprocess 1b script name
# $12: process 1a
# $13: process 1b
# $14: machine

if [ ${14} = "arc3" ]; then
	time="01:45:00"
else
	time="00:40:00"
fi

cat >"qsub_script_${2}_${3}_${4}.sh"<<EOF
#!/bin/bash

#$ -cwd -V
#$ -l h_rt=${time}
#$ -pe smp 8
#$ -l h_vmem=3G

export OMP_NUM_THREADS=8

date

bash ${SCRIPT_HOME}/create_fracture_maps/shell_scripts/process_scene.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}

date
EOF

