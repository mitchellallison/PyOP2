#!/usr/bin/env python
import sys
import os

import matplotlib.pyplot as plt
import pandas

if len(sys.argv) != 6:
    print "Usage: ./plot_parloop_increasing_layers.py FILE_DIR TEST_NAME DISCRETISATION MESH-SIZE OUTPUT_DIR."
    exit()

file_dir = os.path.abspath(sys.argv[1])
test_name = sys.argv[2]
discretisation = sys.argv[3]
mesh_size = sys.argv[4]
graph_dir = os.path.abspath(sys.argv[5])

backends = ["'Sequential'", "'OpenMP'", "'OpenCL_CPU'", "'OpenCL_GPU'"]

df = pandas.DataFrame(columns=backends)

for layer in [60, 90]:
    file_name = "{}[greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
    file_path = os.path.join(file_dir, file_name)
    input_file = open(file_path)

    data = pandas.read_csv(input_file)

    # Without runtime
    parloop_profile_data = data.groupby('Profile')
    parloop_mean = parloop_profile_data['ParLoop kernel'].mean()
    df.loc[layer] = parloop_mean[backends].tolist()
    from IPython import embed; embed()

figure = df.plot(kind='line').get_figure()
figure.savefig(os.path.join(graph_dir, "{}-{}-{}-parloop-increasing-layers.pdf".format(test_name, discretisation, mesh_size)))
