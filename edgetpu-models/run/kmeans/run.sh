#!/bin/sh

for f in input/*.rgb
do
  echo $f
  ./kmeans.out $f /dev/null
done
