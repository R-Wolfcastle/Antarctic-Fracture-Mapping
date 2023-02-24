#!/bin/bash

source ${CLUSTER_CONDA_CONFIG}
conda activate damage_env

## Arguments:
## $1: scene granule name
## $2: path
## $3: frame
## $4: date
## $5: temporary directory
## $6: save directory
## $7: download script name
## $8: bs script name
## $9: damage 1a script name
## $10: damage 1b script name
## $11: parallel structure script name
## $12: processs 1a
## $13: process 1b
## $14: machine

#must be executed in the correct directory:
if [[ "$(pwd)" != $5 ]]; then
	echo "Not executing in the correct directory"
	exit
fi

if [[ -z $(find . -path "./${4}.slc_clip_1.tc.gc.gamma.dB.mli.tiff" -print -quit) ]]; then
	if [[ $(find ${6} -name "${4}.slc_clip_1.tc.gc.gamma.dB.mli.tiff" -print -quit) ]]; then
    	full_run="False"

    	cp ${6}/${4}.slc_clip_1.tc.gc.gamma.dB.mli.tiff .
	else
		full_run="True"

		bash ${7} ${1} || exit

		bash ${8} $1 $2 $3 $4 ${14} || exit
	fi
else full_run="False"
fi

# sar_filepath=$(find . -path "./${4}.slc_clip_1.tc.gc.gamma.mli.bmp.tiff" -print -quit)
sar_filepath=$(find . -path "./${4}.slc_clip_1.tc.gc.gamma.dB.mli.tiff" -print -quit)

#unset GOMP_CPU_AFFINITY KMP_AFFINITY #prevent processes all running on the same CPU core!
num_procs_py=4
num_omp_threads=2

if [ ${12} = "true" ]; then
	if [[ ! -f "${5}/${4}.damage_probs_1a.tif" ]]; then
		if [[ ! -f "${6}/${4}.damage_probs_1a.tif" ]]; then
			echo "Processing Type 1a"
			#export OMP_NUM_THREADS=1 #sack off openMP
			python ${9} $sar_filepath $num_procs_py $num_omp_threads || exit
			#export OMP_NUM_THREADS=8
		fi
	fi
fi

if [ ${13} = "true" ]; then
	if [[ ! -f "${5}/${4}.damage_probs_1b.tif" ]]; then
		if [[ -f "${6}/${4}.damage_probs_1b.tif" ]]; then
			cp "${6}/${4}.damage_probs_1b.tif" ${5}
		else
			echo "Processing Type 1b"
			#export OMP_NUM_THREADS=1
			python ${10} $sar_filepath $num_procs_py $num_omp_threads || exit
			#export OMP_NUM_THREADS=8
		fi
		if [[ ! -f "${5}/${4}.damage_probs_1b_filtered.tif" ]]; then
			echo "Postprocessing Type 1b"
        	python ${11} "./${4}.damage_probs_1b.tif" || exit
		fi

	elif [[ ! -f "${5}/${4}.damage_probs_1b_filtered.tif" ]]; then
		echo "Postprocessing Type 1b"
		python ${11} "./${4}.damage_probs_1b.tif" || exit
	fi
fi

#if [ $full_run = "True" ]; then
	date=${4}

	rm -rf *.SAFE
	rm -rf *.zip

	rm -f ${date}_coffs
	rm -f ${date}_slc.ccp
	rm -f ${date}_coffsets
	rm -f ${date}.slc_clip_1
	rm -f ${date}.dem
	# rm -f ${date}.dem_par
	rm -f ${date}.diff_par
	rm -f ${date}.inc
	rm -f ${date}.inc.bmp
	rm -f ${date}.iw1.slc
	rm -f ${date}.iw1.slc.par
	rm -f ${date}.slc_clip_1.par
	rm -f ${date}.iw1.slc.TOPS_par
	rm -f ${date}.iw2.slc
	rm -f ${date}.iw2.slc.par
	rm -f ${date}.iw2.slc.TOPS_par
	rm -f ${date}.iw3.slc
	rm -f ${date}.iw3.slc.par
	rm -f ${date}.iw3.slc.TOPS_par 
	rm -f ${date}.ls_map
	rm -f ${date}.lt
	rm -f ${date}.lt_fine
	rm -f ${date}.offnadir
	rm -f ${date}_offsets
	rm -f ${date}_slc.offs
	rm -f ${date}.slc.par
	rm -f pix
	rm -f psi
	rm -f SLC_tab
	rm -f u
	rm -f v
	rm -f ${date}.slc
	rm -f ${date}.dem_par
	rm -f ${date}.slc_clip_1.tc.gc.gamma.dB.mli
	rm -f ${date}.pix_gamma0
	rm -f ${date}.slc_clip_1.tc.gc.gamma.mli
	rm -f ${date}.pix_gamma0_ratio
	rm -f ${date}.slc_clip_1.tc.gc.mli
	rm -f ${date}.pix_sigma0
	rm -f ${date}.slc_clip_1.tc.mli
	rm -f ${date}.pix_sigma0_ratio
	rm -f bl.txt
	rm -f ${date}.sim_sar
	rm -f br.txt
	rm -f ${date}.slc_clip_1.mli
	rm -f tl.txt
	rm -f ${date}.slc_clip_1.mli.par
	rm -f tr.txt
	rm -f ${date}.pix_gamma0.bmp
	rm -f ${date}.pix_sigma0.bmp
	rm -f ${date}.slc_clip_1.gc.mli
	rm -f ${date}.slc_clip_1.gc.mli.bmp
	rm -f ${date}.slc_clip_1.gc.mli.bmp.tiff
	rm -f ${date}.slc_clip_1.gc.mli.tiff
	rm -f ${date}.slc_clip_1.mli.bmp
	rm -f ${date}.slc_clip_1.tc.gc.gamma.dB.mli.bmp
	rm -f ${date}.slc_clip_1.tc.gc.gamma.mli.bmp
	rm -f ${date}.slc_clip_1.tc.gc.gamma.mli.bmp.tiff
	rm -f ${date}.slc_clip_1.tc.gc.mli.tiff
	rm -f ${date}.slc_clip_1.tc.mli.bmp
	rm -f *.py
	rm -f *.sh
#fi

mv --no-clobber ${9} ${10} ${11} ${6}/../../../
mv * $6 || exit

