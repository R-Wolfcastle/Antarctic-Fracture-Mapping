#!/bin/bash

if [ -z $1 ]; then echo "Argument is the file name, e.g. S1A....zip"; exit; fi

granule_name=${1}.zip

date="$(echo $granule_name | awk -F'[_T]' '{print $6}')"
satellite="$(echo $granule_name | awk '{print substr($1,1,3)}')"

download_scene () {
	#this script will download a given file (zip file) from alaska facility
	asfuser=${ASF_UNAME}
	asfpasswd=${ASF_PWD}

	echo ${asfpasswd}

	URL=https://datapool.asf.alaska.edu/SLC

	granule_name=$1
	satellite=$2

	if [ $satellite == "S1A" ]; then
	    satstr='SA'
	elif [ $satellite == "S1B" ]; then
	    satstr='SB'
	fi

	wget -c --http-user=$asfuser --http-password=$asfpasswd $URL/$satstr/$granule_name
}


download_scene $granule_name $satellite
