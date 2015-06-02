#!/bin/bash

if [ $# != 5 ]; then
  echo "Must provide a test, a discretisation, mesh size, layer size and iteration count to profile."
else
  test_name=$1
  discretisation=$2
  mesh_size=$3
  layer_size=$4
  iterations=$5

  backends=(opencl opencl)
  profile_names=(OpenCL_CPU OpenCL_GPU)
  machines=(pixel05 graphic06)
  execution_schemes=(C D)

  for i in "${!backends[@]}"; do
    backend=${backends[$i]}
    profile_name=${profile_names[$i]}
    machine=${machines[$i]}
    selected_test="$test_name[$backend-greedy-$discretisation-$mesh_size-$layer_size]"
    execution_scheme=${execution_schemes[$i]}
    echo "### Profiling $profile_name, using backend $backend on $machine ###"
    for (( i=0; i < $iterations; i++ )); do
      command=""
      if [ $profile_name == "MPI" ]
      then
        command="PYOP2_PROFILING=1 mpirun"
        for (( cpu=0; cpu < 8; cpu++ )); do
          command="$command -n 1 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=${profile_name}_$cpu -sx :"
        done
        command=${command::-1}
      elif [ $backend == 'opencl' ]
      then
          command="PYOP2_EXECUTION_SCHEME=0 PYOP2_PROFILING=1 PYOPENCL_CTX=0 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=$profile_name\ Scheme\ ${execution_schemes[0]} -sx"
          for (( scheme=1; scheme < 2; scheme++ )); do
            command="$command && PYOP2_EXECUTION_SCHEME=$scheme PYOP2_PROFILING=1 PYOPENCL_CTX=0 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=$profile_name\ Scheme\ ${execution_schemes[$scheme]} -sx"
          done
      else
          command="PYOP2_PROFILING=1 PYOPENCL_CTX=0 py.test -v /homes/mka211/Documents/IndividualProject/PyOP2/test/unit/test_opencl_extrusion.py --backend=$backend -k $selected_test --profile=$profile_name -sx"
      fi
      echo $command | ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $machine bash -l;
    done
  done
fi
