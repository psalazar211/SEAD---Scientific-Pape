#!/bin/sh

for f in input/*.data
do
  echo $f
  ./jmeint.tf.out $f /dev/null
done
