#!/bin/sh

for f in input/*.rgb
do
  echo $f
  ./sobel.out $f /dev/null
done
