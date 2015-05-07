#!/usr/bin/env python
import sys
import os

from matplotlib import rc
import pandas

if len(sys.argv) != 6:
    print "Usage: ./plot_parloop_increasing_layers.py FILE_DIR TEST_NAME DISCRETISATION MESH-SIZE OUTPUT_DIR."
    exit()

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

file_dir = os.path.abspath(sys.argv[1])
test_name = sys.argv[2]
discretisation = sys.argv[3]
mesh_size = sys.argv[4]
graph_dir = os.path.abspath(sys.argv[5])

backends = ["'Sequential'", "'OpenMP'", "'OpenCL_CPU'", "'OpenCL_GPU'"]
backend_headers = ["Sequential", "OpenMP", "OpenCL CPU", "OpenCL GPU"]

df = pandas.DataFrame(columns=backend_headers)

for layer in [1, 2, 3, 4, 8, 10, 15, 30, 45, 60]:
    file_name = "{}[greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
    file_path = os.path.join(file_dir, file_name)
    input_file = open(file_path)

    data = pandas.read_csv(input_file)

    # Without runtime
    parloop_profile_data = data.groupby('Profile')
    parloop_mean = parloop_profile_data['ParLoop kernel'].mean()
    df.loc[layer] = parloop_mean[backends].tolist()

plot = df.plot(kind='line')
plot.set_xlabel('Layer Count')
plot.set_ylabel('Time (Seconds)')
title = plot.set_title('Time spent in parallel loop against layer count while performing an extruded RHS assembly with discretisation {} and a mesh size of {}'.format(discretisation, mesh_size), fontsize=14, fontweight='bold')
title.set_y(1.03)
figure = plot.get_figure()
figure.savefig(os.path.join(graph_dir, "{}-{}-{}-parloop-increasing-layers.pdf".format(test_name, discretisation, mesh_size)))
