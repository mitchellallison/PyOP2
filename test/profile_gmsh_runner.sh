#!/bin/bash

discretisations=(CG1 DG1)
test=test_extruded_rhs_assembly
layers=(1 2 4 10 30 50 100)
mesh=square
iterations=1

for disc_1 in "${discretisations[@]}"; do
  for disc_2 in "${discretisations[@]}"; do
    for layer in "${layers[@]}"; do
       ./profile.sh $test $disc_1-$disc_2 $mesh $layer $iterations
    done
  done
done
