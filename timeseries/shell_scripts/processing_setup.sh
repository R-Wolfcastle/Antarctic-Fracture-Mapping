#!/bin/bash

#Note: around 60G of mosaics - kind of not that much really!
#and around 16G of output hdf5... 


CHOP_UP_COORD="Y"


#Done manually for speed, but need to sort out:
#AIS:
#ulx=-3000000
#uly=2500000
#lrx=3000000
#lry=-2500000

#ASE:
ulx=-1735000
uly=-190000
lrx=-1425000
lry=-715000

dx=$((lrx-ulx))
dy=$((uly-lry))

echo $dx
echo $dy

#AIS:
#grid_spacing=5000
#buffer_size=10000

#ASE:
grid_spacing=2500
buffer_size=5000


if [ $CHOP_UP_COORD = "X" ]; then

  num_x_grid_points=$((dx/grid_spacing))
  echo ${num_x_grid_points}
  
  #For AIS, could do 50 odd.
  #For ASE, 5 is good enough :)
  #Make sure the maths works out! Should make this less hacky...
  num_job_chunks=6
  
  grid_points_per_chunk=$((num_x_grid_points/num_job_chunks))
  echo ${grid_points_per_chunk}
  
  
  ulxs=()
  lrxs=()
  
  startval=0
  endval=$(( num_job_chunks-1 ))
  for((i=$startval;i<=$endval;++i)) do
      ulxs+=($(( ulx + grid_points_per_chunk * i * grid_spacing )))
      lrxs+=($(( ulx + grid_points_per_chunk * (i+1) * grid_spacing )))
  done
  
  for j in "${!ulxs[@]}"; do
      qsub qsub_script.sh "${ulxs[j]}" $uly "${lrxs[j]}" $lry $j ${grid_spacing} ${buffer_size}
  done

else

  num_y_grid_points=$((dy/grid_spacing))
  echo ${num_y_grid_points}

  num_job_chunks=10

  grid_points_per_chunk=$((num_y_grid_points/num_job_chunks))
  echo ${grid_points_per_chunk}

  ulxs=()
  lrxs=()

  startval=0
  endval=$(( num_job_chunks-1 ))
  for((i=$startval;i<=$endval;++i)) do
      ulys+=($(( uly - grid_points_per_chunk * i * grid_spacing )))
      lrys+=($(( uly - grid_points_per_chunk * (i+1) * grid_spacing )))
  done

  for j in "${!ulys[@]}"; do
      echo $ulx "${ulys[j]}" $lrx "${lrys[j]}" $j ${grid_spacing} ${buffer_size}
      qsub qsub_script.sh $ulx "${ulys[j]}" $lrx "${lrys[j]}" $j ${grid_spacing} ${buffer_size}
  done 

fi

