#!/bin/bash

#argument 1 = (wildcard) path to simtel files
#argument 2 = output .h5 filename/path
#argument 3 = pickled cuts/bins file
#argument 4 = config file

#run image_extractor on each simtel file to append to output HDF5 file

echo "Extracting images from simtel files..."

for f in $1
do
    #echo "python /home/gemini/code/imageExtractor/ctapipe/image_extractor.py $f ./output_temp.h5 $3" 
    echo "$f"
    python "/home/shevek/bsk2133/image-extractor/image_extractor.py" "--bins_cuts_dict" "$3" "$f" "./output_temp.h5" "$4" 
done

echo "Done!"

#shuffle final hdf5 file and split it

echo "Shuffling and splitting dataset..."

python "/home/shevek/bsk2133/image-extractor/scripts/shuffle_split_events.py -shuffle -no_test" "./output_temp.h5" "$2"
#rm "./output_temp.h5"
    
echo "Done!"


