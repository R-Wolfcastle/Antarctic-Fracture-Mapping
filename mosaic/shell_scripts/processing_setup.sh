#!/bin/bash/

#requires GNU date program

startdate=20190101
enddate=20200101

d=
n=0
until [ "$d" = "$enddate" ]; do
  d=$(date -d "$startdate + $n days" +%Y%m%d)
  qsub ${SCRIPT_HOME}/mosaic/shell_scripts/qsub_script.sh $d
  ((n++))
done



