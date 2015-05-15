#!/bin/bash

discretisations=(CG1)
test=test_extruded_rhs_assembly
layers=(1 2 4 10 30 50 100)
mesh=square
iterations=1

for disc_1 in "${discretisations[@]}"; do
  for disc_2 in "${discretisations[@]}"; do
    for layer in "${layers[@]}"; do
      for partition_scale in `seq 0 12`; do
        selected_test="$test[opencl-greedy-$disc_1-$disc_2-$mesh-$layer]"
        scale=`echo "2^$partition_scale" | bc`
        echo "### Profiling $mesh mesh with $layer layers with the partition scaled by 1/$scale ###"
        for (( i=0; i < $iterations; i++ )); do
          echo "PYOP2_PROFILING=1 PYOPENCL_CTX=0 PYOP2_PARTITION_SCALE=$partition_scale py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=opencl -k $selected_test --profile=$scale" | bash
        done
      done
    done
  done
done
