#!/bin/bash

if [ $# != 4 ]; then
  echo "Must provide a test, a discretisation, mesh size and layer size to profile."
else
  test_name=$1
  discretisation=$2
  mesh_size=$3
  layer_size=$4

  iterations=1

  backends=(sequential opencl opencl)
  profile_names=(Sequential OpenCL_CPU OpenCL_GPU)
  machines=(edge02 edge02 graphic06)

  for i in "${!backends[@]}"; do
    backend=${backends[$i]}
    profile_name=${profile_names[$i]}
    machine=${machines[$i]}
    selected_test="$test_name[$backend-greedy-$discretisation-$mesh_size-$layer_size]"
    echo "### Profiling $profile_name, using backend $backend on $machine ###"
    for (( i=0; i < $iterations; i++ )); do
      echo "PYOPENCL_CTX=0 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=$profile_name -sx" | ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $machine bash -l;
    done
  done
fi
