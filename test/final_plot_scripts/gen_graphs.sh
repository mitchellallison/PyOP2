if [ $# != 3 ]; then
  echo "Usage: ./gen_graphs.sh GENERAL_PROFILE_DIR PARTITION_PROFILE_DIR OUTPUT_DIR."
  exit 1
fi
profile_dir=$1
partition_dir=$2
output_dir=$3
./plot_parloop_layer_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layers.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layer_speedup_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layers_speedup.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_layer_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_layers.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_stacked.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_stacked_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layer_partition_size.py $partition_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_maximum_valuable_bandwidth.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_gflops.py $profile_dir test_extruded_rhs_assembly square $output_dir
