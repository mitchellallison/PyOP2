if [ $# != 2 ]; then
  echo "Usage: ./gen_graphs.sh PROFILE_DIR OUTPUT_DIR."
  exit 1
fi
profile_dir=$1
output_dir=$2
./plot_parloop_layer_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layers.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layer_speedup_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_parloop_layers_speedup.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_layer_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_layers.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_stacked.py $profile_dir test_extruded_rhs_assembly square $output_dir
./plot_runtime_stacked_preliminary.py $profile_dir test_extruded_rhs_assembly square $output_dir
