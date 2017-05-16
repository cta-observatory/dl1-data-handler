#!/bin/bash

#scripts to find all duplicate images in a directory

identify -format "%# %f\n" *.png |sort -rnk3 | awk -F"[. ]" 'a[$1]++'
