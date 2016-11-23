#!/bin/bash

###########################################
#Script for Easy Testing of imageExtractor#
###########################################
#
#NOTE: Be sure to run in directory containing imageExtractor, littlecam.cfg,
#and normalizeImages.sh
#
#Description: Runs imageExtractor on desired imagefile, creates (if
#necessary), directory structure to store images, uses normalizeImages.sh
#script to create normalized copies and stores in appropriate directory for
#viewing. 
#
#Finally, creates some symbolic links for fast and convenient movement
#between imageExtractor directory and image directories:
#
#In all newly created subdirectories containing images, creates "back"
#symbolic link to the imageExtractor directory.
#
#In imageExtractor directory, creates/updates link "goto" which points to
#most recently generated normalized image directory
#
##################################################################################
#
#Arguments:
#
#First argument = full file path of dst data file to run imageExtractor on
#
#Second argument= path to directory for image storage (if none provided,
#defaults to  ~)
#
####################################################################################
#
#EX: for a datafile /data/mad2251/gamma/gamma_20deg_0deg_run1000___cta-prod3-sct_desert-2150m-Paranal-SCT.dst.root
# and output directory ~/imageExtractorOutput
#
#will save original images in:
#
#~/imageExtractorOutput/images/gamma/1000/
#
#and normalized images in:
#
#~/imageExtractorOutput/imagesNormalized/gamma/1000/
#
##########################################################################################

datafilepath=$1
outputdir=${2:-"~"}
dir=$(pwd)

echo "Data file path="$1
echo "Output directory="$outputdir

eventtype=$( echo $1 | awk -F"/" '{print $NF}' | awk -F"_" '{print $1}' )

echo "Event Type="$eventtype

run=$( echo $1 | awk -F"/" '{print $NF}' | awk -F"_" '{print $4}' | grep -o '[0-9]\+')

echo "Run Number="$run

echo "Creating output directories..."

if [ ! -d "$outputdir" ]
then
    mkdir "$outputdir"
fi

if [ ! -d "$outputdir""/images/$eventtype/$run" ]
then
    mkdir -p "$outputdir""/images/$eventtype/$run"
fi

if [ ! -d "$outputdir""/imagesNormalized/$eventtype/$run" ]
then
    mkdir -p "$outputdir""/imagesNormalized/$eventtype/$run"
fi

echo "Running imageExtractor..."

./imageExtractor -i "$datafilepath" -c littlecam.cfg -o "$outputdir""/images/$eventtype/$run"

cd "$outputdir""/images/$eventtype/$run"

echo "Generating normalized images..."

"$dir"/normalizeImages.sh "$outputdir""/imagesNormalized/$eventtype/$run"

echo "Creating/updating links..."

ln -sn "$dir" back

cd "$outputdir""/imagesNormalized/$eventtype/$run"

ln -sn "$dir" back

cd "$dir"

ln -snf "$outputdir""/imagesNormalized/$eventtype/$run" goto

echo "Done!"


