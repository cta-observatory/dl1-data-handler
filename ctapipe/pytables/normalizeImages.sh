#!/bin/bash

filepath=$1
ext=png
flag=[NORM]

for i in *.$ext; do

    convert  -normalize $i ${i%.$ext}$flag.$ext

    done

for j in *$flag*; do

    mv $j $filepath

    done
