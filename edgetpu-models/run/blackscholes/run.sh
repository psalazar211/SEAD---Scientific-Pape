#!/bin/sh

for f in input/*.data
do
  echo $f
  ./blackscholes.out $f /dev/null
done
