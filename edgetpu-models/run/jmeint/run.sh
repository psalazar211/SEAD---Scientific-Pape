#!/bin/sh

for f in input/*.data
do
  echo $f
  ./jmeint.out $f /dev/null
done
