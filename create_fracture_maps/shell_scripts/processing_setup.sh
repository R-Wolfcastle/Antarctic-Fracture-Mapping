#!/bin/bash

machine=$( echo $( hostname ) | awk -F '[.]' '{printf $2}' )
echo "Machine: ${machine}"

Y1=2015
M1=01
D1=01

#NOTE
Y2=2022
M2=07
D2=01

stime=("$Y1"-"$M1"-"$D1"T00:00:01)
etime=("$Y2"-"$M2"-"$D2"T23:59:59)
# etime=("$Y2"-"$M2"-"$D2"T07:37:30)
# etime=("$Y2"-"$M2"-"$D2"T09:37:30)

t1="${Y1}${M1}${D1}"
t2="${Y2}${M2}${D2}"
timetag="${t1}_${t2}"

proc_subset="false"
subset_timetag="${sub_t1}_${sub_t2}"

mainPath="$(pwd)"
beamMode="IW"
processingLevel="SLC"

tempDirStub="${TEMP_OUTPUT_BASEDIR}/Antarctica_${timetag}"
mkdir -p $tempDirStub
cd $tempDirStub || exit

saveDirStub="${OUTPUT_BASEDIR}/Antarctica_${timetag}"
mkdir -p $saveDirStub

polygon=$(head -n 1 ../../antarctic_locations/coastline_vertices.txt)

process_1a="true"
process_1b="true"

kml_fn="map-overlay_AIS.kml"

if [[ ! -f "${mainPath}/${kml_fn}" ]]; then
	wget -O ${mainPath}/${kml_fn} https://api.daac.asf.alaska.edu/services/search/param?platform=S1\&polygon=${polygon}\&start=${stime}UTC\&end=${etime}UTC\&beamMode=${beamMode}\&processingLevel=${processingLevel}\&output=KML
fi

num_images=$(grep -o -i '<name>S1' ${mainPath}/${kml_fn} | wc -l)

dates=( $( grep -o -P '(?<=<name>S1[AB]_IW_SLC__[0-9A-Z]{4}_)[0-9]*(?=T.*</name>)' ${mainPath}/${kml_fn} ) )
echo "extracted $(echo ${#dates[@]}) dates"
granule_names=( $( grep -o -P '(?<=<name>)S1.*(?=</name>)' ${mainPath}/${kml_fn} ) )
echo "extracted $(echo ${#granule_names[@]}) granule_names"
paths=( $( grep -o -P '(?<=<li>Path: ).*(?=</li>)' ${mainPath}/${kml_fn} ) )
echo "extracted $(echo ${#paths[@]}) paths"
frames=( $( grep -o -P '(?<=<li>Frame: ).*(?=</li>)' ${mainPath}/${kml_fn} ) )
echo "extracted $(echo ${#frames[@]}) frames"


counter=1
for ((j=1; j<=$num_images; j++)); do
#	#if [[ $counter -gt 10000 ]]; then exit 1; fi

	date="${dates[${j}]}"
	if [ ! $proc_subset = "true" ] || [ "$date" -ge "${sub_t1}" ] && [ "$date" -le "${sub_t2}" ]; then #pretty sure and is before or in logical order precedence...
		path="${paths[${j}]}"
		frame="${frames[${j}]}"
		granule_name="${granule_names[${j}]}"
		echo $frame $path $granule_name $date

		saveDir=${saveDirStub}/${path}/${frame}/${date}
	
		if [[ ! -f "${saveDir}/${date}.damage_probs_1b_filtered.tif" ]]; then
			mkdir -p $saveDir
	
			tempDir=${tempDirStub}/${path}/${frame}/${date}
			mkdir -p $tempDir
			cd $tempDir || exit
	
			download_script_name="download_scene.sh"
			bs_script_name="generate_ard_bs.sh"
			damage_1a_script_name="extract_damage_par_db_1a.py"
			damage_1b_script_name="extract_damage_par_db_1b.py"
			parallel_structure_filtering_script_name="psf.py"

            ${SCRIPT_HOME}    
			cp --no-clobber ${SCRIPT_HOME}/create_fracture_maps/shell_scripts/${download_script_name} .
			cp --no-clobber ${SCRIPT_HOME}/create_fracture_maps/shell_scripts/${bs_script_name} .
			if [ $process_1a = "true" ]; then
				cp --no-clobber ${SCRIPT_HOME}/create_fracture_maps/python_scripts/${damage_1a_script_name} .
			fi
			if [ $process_1b = "true" ]; then
		    	cp --no-clobber ${SCRIPT_HOME}/create_fracture_maps/python_scripts/${damage_1b_script_name} .
		    	cp --no-clobber ${SCRIPT_HOME}/create_fracture_maps/python_scripts/${parallel_structure_filtering_script_name} .
			fi
	
			bash ${SCRIPT_HOME}/create_fracture_maps/shell_scripts/make_qsub_script.sh $granule_name \
																						$path \
																						$frame \
																						$date \
																						$tempDir \
																						$saveDir \
																						${download_script_name} \
																						${bs_script_name} \
																						${damage_1a_script_name} \
		                                                                                ${damage_1b_script_name} \
		                                                                                ${parallel_structure_filtering_script_name} \
		                                                                                ${process_1a} \
		                                                                                ${process_1b} \
																						${machine}
			qsub qsub_script_${path}_${frame}_${date}.sh
			cd ${mainPath}
			let counter++
   		fi
	fi
done

#cp ${kml_fn} ${saveDirStub}/
