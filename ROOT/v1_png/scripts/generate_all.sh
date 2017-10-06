#!/bin/bash

#Mass-generate images from all data files in directory

datadirpath=$1
outputdir=${2:-"~"}
dir=$(pwd -P)

echo "Data directory="$1
echo "Output directory="$outputdir


for file in $datadirpath/*.dst.root; do
     ./imageExtractor -i $file -c littlecam.cfg -o $outputdir
 done

echo "Done!"
