imageExtractor.C
Author: Daniel Nieto (nieto@nevis.columbia.edu)

Additional work by: Bryan Kim (bryan.sanghyuk.kim@gmail.com)

Description:

 Read simulated CTA data in DST format and generates events' images in eps
and png formats.Uses exiv2 to mark images with relevant metadata.


Dependencies:
- OpenCV 2.4.13
- ROOT
- Exiv2 0.25

Compilation instructions:
- Run "make" to compile imageExtractor
- Run "make clean" to remove imageExtractor
- Run "make all" to recompile imageExtractor

Additional Scripts:
- normalizeImages.sh
- output.sh


TO DO LIST:

* Blank image issue:
  - check telescope selection?
  - blank sections of trace matrix?

* Blurs/streaks:
  - data issue - only present in new data files
  - channel encoding issue? (offset)

* Image duplication issue
  - check getEntry behavior?

* Add additional metadata

* Prepare scripts to sort images based on metadata

