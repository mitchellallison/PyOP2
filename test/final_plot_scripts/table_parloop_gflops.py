#!/usr/bin/env python
import sys
import os

from matplotlib import rc
import pandas

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

backends = ["'OpenCL GPU Scheme A'", "'OpenCL GPU Scheme B'", "'OpenCL GPU Scheme C'", "'OpenCL GPU Scheme D'"]
backend_headers = map(lambda backend: backend[1:-1], backends)

df = pandas.DataFrame(columns=backend_headers)

ops_per_kernel = 78
dofs_per_cell = 6
num_cells = 4000000
# Total flops = number of ops per kernel for each dof, multiplied by the number of dofs
total_flops = dofs_per_cell * num_cells * ops_per_kernel

for disc_1 in discretisations:
    for disc_2 in discretisations:
        discretisation = "{}-{}".format(disc_1, disc_2)
        for i, layer in enumerate([1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]):
            timings = []

            opencl_file_name = "{}[opencl-greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
            opencl_file_path = os.path.join(file_dir, opencl_file_name)
            try:
                opencl_data = pandas.read_csv(open(opencl_file_path))
                profile_data = opencl_data.groupby('Profile')
                parloop_profile_data = profile_data['Assembly kernel']
                parloop_min = parloop_profile_data.min()
                timings.extend(map(lambda runtime: (total_flops / runtime) / 2**30, parloop_min[backends]))
            except IOError:
                timings.extend([float('NaN') for i in range(4)])

            df.loc[layer] = timings

        with open(os.path.join(graph_dir, "{}-{}-{}-parloop-gflops-increasing-layers-table.tex".format(test_name, discretisation, mesh_size)), 'w') as f:
            latex = df.to_latex()
            f.write(latex)
