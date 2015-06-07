#!/usr/bin/env python
import sys
import os

from matplotlib import rc
import pandas


def plot(df, backend):
    plot = df.transpose().plot(kind='bar', stacked=True, colormap='gist_ncar')
    plot.set_xlabel('Discretisation')
    plot.set_ylabel('Time (Seconds)')
    handles, labels = plot.get_legend_handles_labels()
    lgd = plot.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
    figure = plot.get_figure()
    figure.savefig(os.path.join(graph_dir, "{}-{}-{}-runtime-stacked.pdf".format(test_name, mesh_size, backend)), bbox_extra_artists=(lgd,), bbox_inches='tight')

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

backends = ["'OpenCL GPU Scheme D'"]
backend_headers = map(lambda backend: backend[1:-1], backends)

sequential_df = pandas.DataFrame(columns=disc_cross_product)
mpi_df = pandas.DataFrame(columns=disc_cross_product)

for disc_1 in discretisations:
    for disc_2 in discretisations:
        discretisation = "{}-{}".format(disc_1, disc_2)
        layer = 128
        timings = []
        sequential_file_name = "{}[sequential-greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
        sequential_file_path = os.path.join(file_dir, sequential_file_name)
        try:
            sequential_data = pandas.read_csv(open(sequential_file_path))
            profile_data = sequential_data.groupby('Profile')
            parloop_min = profile_data.min()
            sequential_data = parloop_min.transpose()[backends[0]]
            sequential_df["{}-{}".format(disc_1, disc_2)] = sequential_data
            mpi_data = parloop_min.transpose()["'MPI_1'"]
            mpi_df["{}-{}".format(disc_1, disc_2)] = mpi_data
        except IOError:
            pass

sequential_df = sequential_df.drop("Assemble cells")
plot(sequential_df, backend_headers[0])
mpi_df = mpi_df.drop("Assemble cells")
plot(mpi_df, backend_headers[1])

for i in range(2, len(backends)):
    opencl_df = pandas.DataFrame(columns=disc_cross_product)
    for disc_1 in discretisations:
        for disc_2 in discretisations:
            opencl_file_name = "{}[opencl-greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
            opencl_file_path = os.path.join(file_dir, opencl_file_name)
            try:
                opencl_data = pandas.read_csv(open(opencl_file_path))
                profile_data = opencl_data.groupby('Profile')
                parloop_min = profile_data.min()
                transposed_min = parloop_min.transpose()
                if backends[i] in transposed_min:
                    opencl_data = parloop_min.transpose()[backends[i]]
                    opencl_df["{}-{}".format(disc_1, disc_2)] = opencl_data
            except IOError:
                pass
    opencl_df = opencl_df.drop("Assemble cells")
    plot(opencl_df, backend_headers[i])
