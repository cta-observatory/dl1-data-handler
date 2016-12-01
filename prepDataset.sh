#!/bin/bash

valpercent=20
testpercent=20

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

echo "Training Set # =$((numImages - valnum - testnum))"
echo "Validation Set # = "$valnum
echo "Test Set # = "$testnum 
echo

echo "Creating directories..."

if [ ! -d "./train" ]
then
    mkdir "./train"
fi

if [ ! -d  "./val" ]
then
    mkdir "./val"
fi

if [ ! -d  "./test" ]
then
    mkdir "./test"
fi

echo "Moving images..."

ls *.png |sort -R |tail -$valnum |while read file; do
        mv $file ./val  
     done

ls *.png |sort -R |tail -$testnum |while read file; do
        mv $file ./test  
     done

ls *.png |while read file; do
        mv $file ./train  
     done

echo "Done!"
