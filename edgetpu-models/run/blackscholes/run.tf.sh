#!/bin/sh

for f in input/*.data
do
  echo $f
  ./blackscholes.tf.out $f /dev/null
done
