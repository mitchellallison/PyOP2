#!/usr/bin/env python
import sys
import os

from matplotlib import rc
import pandas


def plot(df, backend):
    with open(os.path.join(graph_dir, "{}-{}-{}-runtime-stacked-table.tex".format(test_name, mesh_size, backend)), 'w') as f:
        latex = df.to_latex()
        f.write(latex)

if len(sys.argv) != 5:
    print "Usage: {} FILE_DIR TEST_NAME MESH-SIZE OUTPUT_DIR.".format(sys.argv[0])
    exit()

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

file_dir = os.path.abspath(sys.argv[1])
test_name = sys.argv[2]
mesh_size = sys.argv[3]
graph_dir = os.path.abspath(sys.argv[4])

discretisations = ["CG1", "DG1"]

disc_cross_product = ["{}-{}".format(disc_1, disc_2) for disc_1 in discretisations for disc_2 in discretisations]

backends = ["'OpenCL GPU Scheme C'"]
backend_headers = map(lambda backend: backend[1:-1], backends)

layer = 256

opencl_df = pandas.DataFrame(columns=disc_cross_product)
for disc_1 in discretisations:
    for disc_2 in discretisations:
        discretisation = "{}-{}".format(disc_1, disc_2)
        opencl_file_name = "{}[opencl-greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
        opencl_file_path = os.path.join(file_dir, opencl_file_name)
        try:
            opencl_data = pandas.read_csv(open(opencl_file_path))
            profile_data = opencl_data.groupby('Profile')
            parloop_min = profile_data.min()
            transposed_min = parloop_min.transpose()
            if backends[0] in transposed_min:
                opencl_data = parloop_min.transpose()[backends[0]]
                opencl_df["{}-{}".format(disc_1, disc_2)] = opencl_data
        except IOError:
            pass
opencl_df = opencl_df.drop("Assemble cells")
plot(opencl_df, backend_headers[0])
