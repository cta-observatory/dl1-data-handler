#!/bin/bash

optype=${1:-"cp"}

traindir=${2:-""}
valdir=${3:-""}
testdir=${4:-""}

valpercent=20
testpercent=0

images=( *.png )
numImages=${#images[@]}

echo "Total # of Images="$numImages
echo

echo "Training Set % = "$((100 - valpercent  - testpercent ))
echo "Validation Set % = "$valpercent
echo "Test Set % = "$testpercent
echo 

valnum=$((valpercent * numImages / 100))
testnum=$((testpercent * numImages / 100))
trainnum=$((numImages - valnum - testnum))

echo "Training Set # ="$trainnum
echo "Validation Set # = "$valnum
echo "Test Set # = "$testnum 
echo

#echo "Creating directories..."
#
#if [ (! -d "./train")  ]
#then
#    mkdir "./train"
#fi
#
#if [ ! -d  "./val"-a ($valpercent -gt "0") ]
#then
#    mkdir "./val"
#fi
#
#if [ ! -d  "./test" -a ($testpercent -gt "0") ]
#then
#    mkdir "./test"
#fi

if [ $optype = "mv" ]
then

echo "Moving images..."

if [ $valpercent -gt "0" ]
then
-type f -name '*.png'
find -type f -name '*.png' |sort -R |tail -$valnum |while read file; do
        mv $file $valdir
     done
fi

if [ $testpercent -gt "0" ]
then

find -type f -name '*.png' |sort -R |tail -$testnum |while read file; do
        mv $file $testdir 
     done
fi

find -type f -name '*.png' |while read file; do
        mv $file $traindir 
     done

elif [ $optype = "cp" ]
then

echo "Copying images..."

find -type f -name '*.png' |while read file; do
        cp $file $traindir
     done

cd $traindir

if [ $valpercent -gt "0" ]
then

find -type f -name '*.png' |sort -R |tail -$valnum |while read file; do
        mv $file $valdir
     done
fi

if [ $testpercent -gt "0" ]
then

find -type f -name '*.png' |sort -R |tail -$testnum |while read file; do
        mv $file $testdir
     done
fi

fi

echo "Done!"


