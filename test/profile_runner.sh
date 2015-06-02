#!/bin/bash

discretisations=(CG1 DG1)
test=test_extruded_rhs_assembly
layers=(1 2 4 8 16 32 64 96 128 160 192 224 256)
mesh_size=square
iterations=15

for disc_1 in "${discretisations[@]}"; do
  for disc_2 in "${discretisations[@]}"; do
    for layer in "${layers[@]}"; do
       ./profile.sh $test $disc_1-$disc_2 $mesh_size $layer $iterations
    done
  done
done
