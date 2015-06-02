#!/bin/bash

test=test_extruded_rhs_assembly
layers=(1 2 4 8 16 32 64 96 128 160 192 224 256)
mesh_size=square
iterations=3
disc_1=CG2
disc_2=DG0

for layer in "${layers[@]}"; do
   ./profile.sh $test $disc_1-$disc_2 $mesh_size $layer $iterations
done
