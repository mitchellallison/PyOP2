#!/bin/bash

discretisations=(CG1 DG1)
test=test_extruded_rhs_assembly
layers=(1 2 4 8 16 32 64 96 128 160 192 224 256)
partition_sizes=(2 3 4 5 5.585 6 7 8 9 10)
execution_schemes=(C D)
mesh=square
iterations=5

for disc_1 in "${discretisations[@]}"; do
  for disc_2 in "${discretisations[@]}"; do
    for layer in "${layers[@]}"; do
      for partition_size in "${partition_sizes[@]}"; do
        selected_test="$test[opencl-greedy-$disc_1-$disc_2-$mesh-$layer]"
        partition_size_eval=`printf "%.0f" $(echo "e(l(2)*(10+$partition_size))" | bc -l)`
        echo "### Profiling $mesh mesh with $layer layers with the partition scaled by 1/$partition_size_eval ###"
        for (( i=0; i < $iterations; i++ )); do
          command="PYOP2_EXECUTION_SCHEME=0 PYOP2_PROFILING=1 PYOPENCL_CTX=0 PYOP2_PARTITION_SIZE=$partition_size_eval py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=opencl -k $selected_test --profile='$partition_size_eval ${execution_schemes[0]}'"
          for (( scheme=1; scheme < 2; scheme++ )); do
            command="$command && PYOP2_EXECUTION_SCHEME=$scheme PYOP2_PROFILING=1 PYOPENCL_CTX=0 PYOP2_PARTITION_SIZE=$partition_size_eval py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=opencl -k $selected_test --profile='$partition_size_eval ${execution_schemes[$scheme]}'"
          done
          echo $command | bash
        done
      done
    done
  done
done
