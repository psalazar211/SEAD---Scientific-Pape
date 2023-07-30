#!/bin/sh

for f in input/*.rgb
do
  echo $f
  ./sobel.tf.out $f /dev/null
done
