#!/bin/bash

if [ $# != 5 ]; then
  echo "Must provide a test, a discretisation, mesh size, layer size and iteration count to profile."
else
  test_name=$1
  discretisation=$2
  mesh_size=$3
  layer_size=$4
  iterations=$5

  backends=(sequential openmp sequential opencl opencl)
  profile_names=(Sequential OpenMP MPI OpenCL_CPU OpenCL_GPU)
  machines=(pixel05 pixel05 pixel05 pixel05 graphic06)

  for i in "${!backends[@]}"; do
    backend=${backends[$i]}
    profile_name=${profile_names[$i]}
    machine=${machines[$i]}
    selected_test="$test_name[$backend-greedy-$discretisation-$mesh_size-$layer_size]"
    echo "### Profiling $profile_name, using backend $backend on $machine ###"
    for (( i=0; i < $iterations; i++ )); do
      command=""
      if [ $profile_name == "MPI" ]
      then
        command="PYOP2_PROFILING=1 mpirun"
        for (( i=0; i < 8; i++ )); do
          command="$command -n 1 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=${profile_name}_$i -sx :"
        done
        command=${command::-1}
      else
          command="PYOP2_PROFILING=1 PYOPENCL_CTX=0 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=$profile_name -sx"
      fi
      echo $command | ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $machine bash -l;
    done
  done
fi
