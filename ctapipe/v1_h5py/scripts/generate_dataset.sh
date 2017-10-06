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
    python "/home/gemini/code/imageExtractor/ctapipe/image_extractor.py" "$f" "./output_temp.h5" "$3" "$4"
done

echo "Done!"

#shuffle final hdf5 file, then delete original file

echo "Shuffling dataset..."

python "/home/gemini/code/imageExtractor/ctapipe/shuffle_events.py" "./output_temp.h5" "$2" "$4"
#rm "./output_temp.h5"
    
echo "Done!"


