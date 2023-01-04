#!/bin/bash

damage_type="bs"

wd=$(pwd)

txt_files_dir="${DATA_SUPERDIR}/date_fp_txt_files"

file_basename="damage_probs_1b_filtered.tif"
if [ ${damage_type} = "1a" ]; then
    file_basename="damage_probs_1a_sqrt.tif"
elif [ ${damage_type} = "bs" ]; then
    file_basename="slc_clip_1.tc.gc.gamma.dB.mli.tiff"
fi



mother_txt_file="${DATA_SUPERDIR}/all_${damage_type}_fps.txt"


cd ${DATA_SUPERDIR}
rm -f ${mother_txt_file}
$(find . -name "*${file_basename}" > ${mother_txt_file})
cd ${wd}

#exit

startdate=20140101
#startdate=20210101
enddate=20220701
date=
n=0
until [ "$date" = "$enddate" ]; do  
    ((n++))
    date=$(date -d "$startdate + $n days" +%Y%m%d)
    echo $date

    $(grep -o -P "\./Antarctica_[0-9]{8}_[0-9]{8}/[0-9]*/[0-9]*/${date}/${date}.${file_basename}" ${mother_txt_file}  > "${txt_files_dir}/${date}_${damage_type}_fps.txt")
    #printf "%s\n" "${fps[@]}" > ${txt_files_dir}/${date}_${damage_type}_fps.txt

    #$( find ${DATA_SUPERDIR} -name ${date}.${file_basename} > ${txt_files_dir}/${date}_${damage_type}_fps.txt )
done

