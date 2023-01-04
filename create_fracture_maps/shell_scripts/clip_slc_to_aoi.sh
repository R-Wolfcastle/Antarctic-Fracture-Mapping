#!/bin/bash

max() {
  [ "$#" -gt 0 ] || return
  typeset i max="$1"; shift
  for i do
    if [ "$((i > max))" -ne 0 ]; then
      max=$i
    fi
  done
  printf '%s\n' "$max"
}
min() {
  [ "$#" -gt 0 ] || return
  typeset i min="$1"; shift
  for i do
    if [ "$((i < min))" -ne 0 ]; then
      min=$i
    fi
  done
  printf '%s\n' "$min"
}


# ### These are hard-coded based on the Thwaites AOI
# minx="-1608825"
# miny="-507475"
# maxx="-1494375"
# maxy="-389475"

infile=$1
inparfile=$2
dempar=$3
minx=$4
miny=$5
maxx=$6
maxy=$7
outfile=$8
outparfile=$9

if [ ! "$minx" = "none" ] && [ ! "$miny" = "none" ] && [ ! "$maxx" = "none" ] && [ ! "$maxy" = "none" ]; then

  echo "++++++++++++++++++++++++++++++++++++++++++"
  echo "++++++++++++++++++++++++++++++++++++++++++"
  echo "$infile $inparfile $dempar $minx $miny $maxx $maxy $outfile $outparfile"
  echo "++++++++++++++++++++++++++++++++++++++++++"
  echo "++++++++++++++++++++++++++++++++++++++++++"


  coord_to_sarpix $inparfile - $dempar $miny $minx 0 | tee bl.txt
  coord_to_sarpix $inparfile - $dempar $maxy $minx 0 | tee tl.txt
  coord_to_sarpix $inparfile - $dempar $maxy $maxx 0 | tee tr.txt
  coord_to_sarpix $inparfile - $dempar $miny $maxx 0 | tee br.txt

  blr="$(grep \(int\): bl.txt | cut -c44-48)"
  blc="$(grep \(int\): bl.txt | cut -c57-61)"

  tlr="$(grep \(int\): tl.txt | cut -c44-48)"
  tlc="$(grep \(int\): tl.txt | cut -c57-61)"

  brr="$(grep \(int\): br.txt | cut -c44-48)"
  brc="$(grep \(int\): br.txt | cut -c57-61)"

  trr="$(grep \(int\): tr.txt | cut -c44-48)"
  trc="$(grep \(int\): tr.txt | cut -c57-61)"

  echo "++++++++++++++++++++++++++++++++++++++++++"
  echo "++++++++++++++++++++++++++++++++++++++++++"
  echo "$blr $blc $tlr $tlc $brr $brc $trr $trc"
  echo "++++++++++++++++++++++++++++++++++++++++++"
  echo "++++++++++++++++++++++++++++++++++++++++++"

  corner_ranges=($blr $tlr $trr $brr)
  max_range="$(max "${corner_ranges[@]}")"
  min_range="$(min "${corner_ranges[@]}")"

  corner_cols=($blc $tlc $trc $brc)
  max_az="$(max "${corner_cols[@]}")"
  min_az="$(min "${corner_cols[@]}")"

  start_range=$min_range
  range_span=$(expr $max_range - $min_range)

  start_az=$min_az
  az_span=$(expr $max_az - $min_az)


  SLC_copy $infile $inparfile $outfile $outparfile 1 1.0 $start_range $range_span $start_az $az_span

else

  SLC_copy $infile $inparfile $outfile $outparfile 1 1.0 - - - -
  
fi
