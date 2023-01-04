#!/bin/bash

# INPUTS
#	$1=image date (yyyymmdd)
#	$2=orbit data type
#	$3=platform (S1A or S1B)

# orbit directory
if [ $2 == 1 ]; then
	Orbit_dir=${ORBIT_DIR}
elif [ $2 == 2 ]; then
	Orbit_dir=${ORBIT_DIR}
	# Get start and end of sensing time of day
	time_start=`grep title: $1.mli.par | cut -c36-41`
	time_end=`grep title: $1.mli.par | cut -c52-57`
fi

# date of OPOD orbit file to use - look for file starting 1 day before image to ensure full period covered
if [ $2 == 1 ]; then
	orb_date_temp=`date -d $1 +%s`
	orb_date_temp=`echo ${orb_date_temp}-86400 | bc -l`
	orb_date_temp=`date -d @${orb_date_temp} +'%F'`
	orb_date=`echo ${orb_date_temp} | tr -d '-'`
elif [ $2 == 2 ]; then
	orb_date=$1
	start_precise=${orb_date}${time_start}
	end_precise=${orb_date}${time_end}
fi
 
 
# Update orbits
if [ $2 == 1 ]; then
	
	# Get orbit name
	orb_name=${Orbit_dir}/S1*_OPER_AUX_POEORB_OPOD_********T******_V${orb_date}T******_********T******.EOF
	
	# In some cases, there are multiple versions of the same orbit file (i.e. valid for the same day, but released at different times)
	# In these cases, get the more recent file
	orb_name_arr=(${orb_name// / }) # convert to indexed array
	num_orb_files=(${#orb_name_arr[@]})
	if (( num_orb_files > 1 )); then
		for ((i=1; i<=$num_orb_files; i++)); do
			idx=$( echo $(( $i - 1 )) ) # bash indexing starts from 0
			tmp_name=${orb_name_arr[$idx]}
			release_date=`echo $tmp_name | cut -c60-67`
			if [ $i == 1 ]; then
				latest_release=$release_date
				idx2use=$idx
			else
				if (( release_date > latest_release )); then
					latest_release=$release_date
					idx2use=$idx
				fi
			fi
		done
	fi
	orb_name_use=${orb_name_arr[$idx2use]}

	# Check we found a file
	if [[ -z "${orb_name_use// }" ]]; then
		echo "Couldn't find an orbit file, exiting"
		exit 0
	fi

	# Extract Sentinel-1 OPOD state vectors and copy into the ISP image parameter file
	echo "Found orbit file: ${orb_name_use}"
	S1_OPOD_vec $1.slc.par ${orb_name_use}
	
elif [ $2 == 2 ]; then
	
	check=0
	for i in ${Orbit_dir}/*V${orb_date}*.EOF; do
		orb_name=`basename $i`
		orbit_valid_start_date=`echo $orb_name | cut -c 43-50` 
		orbit_valid_start_time=`echo $orb_name | cut -c 52-57`
		orbit_valid_start_precise=${orbit_valid_start_date}${orbit_valid_start_time}
		orbit_valid_end_date=`echo $orb_name | cut -c 59-66` 
		orbit_valid_end_time=`echo $orb_name | cut -c 68-73`
		orbit_valid_end_precise=${orbit_valid_end_date}${orbit_valid_end_time}
		if (( orbit_valid_start_precise < start_precise )); then
			if (( orbit_valid_end_precise > end_precise )); then
				if [ $check == 0 ]; then
					orbit_name=$orb_name
					check=1
				fi
			fi
		fi
	done
	# Extract Sentinel-1 OPOD state vectors and copy into the ISP image parameter file
	echo "Found orbit file: ${orbit_name}"
	S1_OPOD_vec $1.slc.par ${Orbit_dir}/${orbit_name}

fi 
