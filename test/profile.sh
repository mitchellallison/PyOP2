#!/bin/bash

if [ $# != 5 ]; then
  echo "Must provide a test, a discretisation, mesh size, layer size and iteration count to profile."
else
  test_name=$1
  discretisation=$2
  mesh_size=$3
  layer_size=$4
  iterations=$5

  backends=(sequential openmp opencl opencl)
  profile_names=(Sequential OpenMP OpenCL_CPU OpenCL_GPU)
  machines=(pixel01 pixel01 pixel01 graphic02)
  execution_types=(greedy lazy)

  for i in "${!backends[@]}"; do
    for execution_type in "${execution_types[@]}"; do
      backend=${backends[$i]}
      profile_name=${profile_names[$i]}
      machine=${machines[$i]}
      selected_test="$test_name[$backend-$execution_type-$discretisation-$mesh_size-$layer_size]"
      echo "### Profiling $profile_name, using backend $backend on $machine ###"
      for (( i=0; i < $iterations; i++ )); do
        echo "PYOP2_PROFILING=1 PYOPENCL_CTX=0 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=$profile_name -sx" | ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $machine bash -l;
      done
    done
  done
fi
