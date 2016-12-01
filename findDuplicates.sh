#!/bin/bash

identify -format "%# %f\n" *.png |sort -rnk3 | awk -F"[. ]" 'a[$1]++'
