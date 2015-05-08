#!/bin/bash

discretisations=(CG1 DG1)
test=test_extruded_rhs_assembly
layers=(1 2 3 4 8 10 15 30 45 60)
mesh_size=100x100
iterations=10

for disc_1 in "${discretisations[@]}"; do
  for disc_2 in "${discretisations[@]}"; do
    for layer in "${layers[@]}"; do
       ./profile.sh $test $disc_1-$disc_2 $mesh_size $layer $iterations
    done
  done
done
