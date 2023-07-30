#!/bin/sh

for f in input/*.rgb
do
  echo $f
  ./jpeg.tf.out $f /dev/null
done
