#!/bin/sh

for f in input/*.rgb
do
  echo $f
  ./jpeg.out $f /dev/null
done
