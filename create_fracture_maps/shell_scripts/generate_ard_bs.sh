#!/bin/bash

module rm gamma
#if [ ${14} = 'arc4' ]; then
	module add test gamma/20201216
#else
#	module load gamma/20190613
#fi

generate_mlis () {

	zip_filepath=$(find . -path "./*${date}T*.zip" -print -quit)

	echo "Zip Filepath: ${zip_filepath}"

	unzip ${zip_filepath}

	filepath=$(find . -path "./*${date}T*.SAFE" -print -quit)

	# ultra-smooth DEM from REMA 1km
	dem_par=${DEM_PAR_FILEPATH}
	dem=${DEM_FILEPATH}

	outres=50
	demresN="$(awk '$1 == "post_north:" {print $2}' $dem_par)"
	ovrfactN="$(echo $demresN $outres | awk '{print -1*($1/$2)}')"
	demresE="$(awk '$1 == "post_east:" {print $2}' $dem_par)"
	ovrfactE="$(echo $demresE $outres | awk '{print $1/$2}')"
	echo $ovrfactN
	echo $ovrfactE


	filenm="$(echo $filepath | awk -F'[/]' '{print $2}')"
	echo "filenm: $filenm"
	datetrack="$(echo $filenm | awk -F'[/._]' '{print $6}')"
	echo "datetrack: $datetrack"

	date="$(echo $datetrack | awk -F'[T]' '{print $1}')"


	scene=$date
	echo "scene: $scene"
	satellite="$(echo $filenm | awk -F'[/._]' '{print $1}')"
	orbit_data_type=1

	cd $filenm || exit 1
	pwd
	for file in $(find . -regex '.*/[a-z0-9\-]*\.tiff'); do
		echo $file
		filename="$(echo $file | awk -F'[/.]' '{print $4}')"
		echo "filename: $filename"
		SWATH="$(echo $filename | awk -F'[-t]' '{print $2}')"
		echo "SWATH: $SWATH"
		DATE="$(echo $filename | awk -F'[-t]' '{print $5}')"
		echo "DATE: $DATE"
		cd ../
		par_S1_SLC ./*/*/$filename.tiff ./*/*/$filename.xml ./*/*/*/calibration-$filename.xml ./*/*/*/noise-$filename.xml $DATE.$SWATH.slc.par $DATE.$SWATH.slc $DATE.$SWATH.slc.TOPS_par
		cd $filenm
	done
	cd ../

	echo "$scene.iw1.slc $scene.iw1.slc.par $scene.iw1.slc.TOPS_par" > SLC_tab
	echo "$scene.iw2.slc $scene.iw2.slc.par $scene.iw2.slc.TOPS_par" >> SLC_tab
	echo "$scene.iw3.slc $scene.iw3.slc.par $scene.iw3.slc.TOPS_par" >> SLC_tab

	SLC_mosaic_S1_TOPS SLC_tab $scene.slc $scene.slc.par 10 2

	# Update orbits to reflect precise orbit information
    bash ${SCRIPT_HOME}/create_fracture_maps/shell_scripts/update_orbits_arc4.sh $date ${orbit_data_type} ${satellite}


	########## EXTRACTING AIO #############################################
	if [ "$extract_aoi" = true ]; then
		### TG terminus Noel AOI
        #minx="-1610800"
        #miny="-487400"
        #maxx="-1508400"
        #maxy="-411800"

		bash ${SCRIPT_HOME}/create_fracture_maps/shell_scripts/clip_slc_to_aoi.sh $scene.slc $scene.slc.par $dem_par $minx $miny $maxx $maxy $scene.slc_clip_1 $scene.slc_clip_1.par

	else
		SLC_copy $scene.slc $scene.slc.par $scene.slc_clip_1 $scene.slc_clip_1.par 1 1.0 - - - -
	fi
	######################################################################


	# multi_look and generate image
	multi_look $scene.slc_clip_1 $scene.slc_clip_1.par $scene.slc_clip_1.mli $scene.slc_clip_1.mli.par 10 2 - - - -
	scene_range_samples="$(grep range_samples $scene.slc_clip_1.mli.par | awk -F'[:]' '{print $2}')"
	scene_azimuth_lines="$(grep azimuth_lines $scene.slc_clip_1.mli.par | awk -F'[:]' '{print $2}')"
	#raspwr $scene.slc_clip_1.mli $scene_range_samples - - - - - - - $scene.slc_clip_1.mli.bmp - -
	
	# Generate geocoding lookup table
	#20201206:
	gc_map2 $scene.slc_clip_1.mli.par $dem_par $dem $scene.dem_par $scene.dem $scene.lt $ovrfactN $ovrfactE $scene.ls_map - $scene.inc - $scene.offnadir $scene.sim_sar u v psi pix - - 0 8 0 - 0
	#20191213
	# gc_map2 $scene.slc_clip_1.mli.par $dem_par $dem $scene.dem_par $scene.dem $scene.lt $ovrfactN $ovrfactE $scene.ls_map - $scene.inc - $scene.offnadir - 0 0 8 - - -
	# gc_map2 <MLI_par> <DEM_par> 				<DEM [DEM_seg_par] [DEM_seg] [lookup_ta [lat_ovr] [lon_ovr] [ls_map] [ls_map_rdc] [inc] [res] [offnadir] [r_ovr] [ls_scaling] [mask] [frame] [DIFF_par] [ref_flg]


	# gc_map2 <MLI_par> <DEM_par> <DEM> [DEM_seg_par] [DEM_seg] [lookup_table] [lat_ovr] [lon_ovr] [ls_map] [ls_map_rdc] [inc] - - - - - - - - - [mask] [frame] [ls_scaling] [DIFF_par] [ref_flg]
	# Refine geocoding lookup table (calculates simulated backscatter images, pix_gamma0 & pix_sigma0, based on DEM)
	pixel_area ${scene}.slc_clip_1.mli.par $scene.dem_par $scene.dem $scene.lt ${scene}.ls_map ${scene}.inc ${scene}.pix_sigma0 ${scene}.pix_gamma0 - - ${scene}.pix_sigma0_ratio ${scene}.pix_gamma0_ratio

	scene_DEM_width="$(grep width: $scene.dem_par | cut -c22-26)"
	scene_DEM_lng="$(grep nlines: $scene.dem_par | cut -c22-26)"

	if [ "$fine_geocoding" = true ]; then
		echo "Executing fine geocoding"
		########## APPLY TERRAIN CORRECTION TO BACKSCATTER COEFFICIENTS ####################
		# Based on the simulated and real MLI images, determine and apply a correction to the geocoding lookup table
		create_diff_par ${scene}.slc_clip_1.mli.par - ${scene}.diff_par 1 0
		#	- Cross-correlate real and simulated MLI images to determine apparent offset induced uncertainty in orbital positioning
		#		NOTE: not used later, so could remove?
		rng_win=256
		azi_win=64
		nr=64
		naz=64
		offset_pwrm ${scene}.pix_sigma0 $scene.slc_clip_1.mli ${scene}.diff_par ${scene}_slc.offs ${scene}_slc.ccp ${rng_win} ${azi_win} ${scene}_offsets 1 ${nr} ${naz} 0.2
		# offset_pwrm <MLI-1> <MLI-2> <DIFF_par> <offs> <ccp> [rwin] [azwin] [offsets] [n_ovr] [nr] [naz] [thres] [lanczos] [bw_frac] [pflag] [pltflg] [ccs] [std_mean]
		#	- Calculate a polynomial (constant offset) to characterise the apparent offset
		offset_fitm ${scene}_slc.offs ${scene}_slc.ccp ${scene}.diff_par ${scene}_coffs ${scene}_coffsets 0.2 1
		#	- Refine geocoding lookup table
		gc_map_fine ${scene}.lt $scene_DEM_width ${scene}.diff_par $scene.lt_fine 1

		# Apply pixel area again using the refined lookup table, so that simulated image uses refined geometry
		pixel_area ${scene}.slc_clip_1.mli.par $scene.dem_par ${scene}.dem ${scene}.lt_fine ${scene}.ls_map ${scene}.inc ${scene}.pix_sigma0 ${scene}.pix_gamma0 - - ${scene}.pix_sigma0_ratio ${scene}.pix_gamma0_ratio
	fi

	##raspwr ${scene}.pix_sigma0 $scene_range_samples - - - - - - - ${scene}.pix_sigma0.bmp - -
	##raspwr ${scene}.pix_gamma0 $scene_range_samples - - - - - - - ${scene}.pix_gamma0.bmp - -
	##raspwr ${scene}.inc $scene_DEM_width - - - - - - - ${scene}.inc.bmp - -
	######################################################################################

	product $scene.slc_clip_1.mli ${scene}.pix_sigma0_ratio $scene.slc_clip_1.tc.mli $scene_range_samples 1 1 -
	## geocode_back $scene.slc_clip_1.tc.mli $scene_range_samples $scene.lt_fine $scene.gc.slc_clip_1.tc.mli $dem_width $dem_lng 2 0
	##raspwr $scene.slc_clip_1.tc.mli $scene_range_samples - - - - - - - $scene.slc_clip_1.tc.mli.bmp - -
	#data2geotiff $scene.dem_par $scene.slc_clip_1.tc.mli.bmp 0 $scene.slc_clip_1.tc.mli.bmp.tiff

	## convert sigma to gamma (see Flattening Gamma paper if you can't remember which is which https://ieeexplore.ieee.org/document/5752845)
	if [ "$fine_geocoding" = true ]; then
		geocode_back $scene.slc_clip_1.tc.mli ${scene_range_samples} ${scene}.lt_fine $scene.slc_clip_1.tc.gc.mli ${scene_DEM_width} ${scene_DEM_lng} 2 0
	else
		geocode_back $scene.slc_clip_1.mli ${scene_range_samples} ${scene}.lt $scene.slc_clip_1.gc.mli ${scene_DEM_width} ${scene_DEM_lng} 2 0
		geocode_back $scene.slc_clip_1.tc.mli ${scene_range_samples} ${scene}.lt $scene.slc_clip_1.tc.gc.mli ${scene_DEM_width} ${scene_DEM_lng} 2 0
	fi

	## make uncorrected bs image:
	##not sure I need all of these as I only use $scene.slc_clip_1.tc.gc.gamma.mli.bmp.tiff
	#data2geotiff $scene.dem_par $scene.slc_clip_1.gc.mli 2 $scene.slc_clip_1.gc.mli.tiff
	##raspwr $scene.slc_clip_1.gc.mli $scene_DEM_width - - - - - - - $scene.slc_clip_1.gc.mli.bmp - -
	#data2geotiff $scene.dem_par $scene.slc_clip_1.gc.mli.bmp 0 $scene.slc_clip_1.gc.mli.bmp.tiff

	sigma2gamma $scene.slc_clip_1.tc.gc.mli $scene.inc $scene.slc_clip_1.tc.gc.gamma.mli $scene_DEM_width
	data2geotiff $scene.dem_par $scene.slc_clip_1.tc.gc.mli 2 $scene.slc_clip_1.tc.gc.mli.tiff
	##raspwr $scene.slc_clip_1.tc.gc.gamma.mli $scene_DEM_width - - - - - - - $scene.slc_clip_1.tc.gc.gamma.mli.bmp - -
	#data2geotiff $scene.dem_par $scene.slc_clip_1.tc.gc.gamma.mli.bmp 0 $scene.slc_clip_1.tc.gc.gamma.mli.bmp.tiff
	
	## Convert to decibels (dB)
	float_math $scene.slc_clip_1.tc.gc.gamma.mli - $scene.slc_clip_1.tc.gc.gamma.dB.mli ${scene_range_samples} 4
	data2geotiff $scene.dem_par $scene.slc_clip_1.tc.gc.gamma.dB.mli 2 $scene.slc_clip_1.tc.gc.gamma.dB.mli.tiff
	##raspwr $scene.slc_clip_1.tc.gc.gamma.dB.mli $scene_DEM_width - - - - - - - $scene.slc_clip_1.tc.gc.gamma.dB.mli.bmp - -
}


fine_geocoding=false
extract_aoi=false

generate_mlis
