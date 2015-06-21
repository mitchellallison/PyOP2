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

#mvbw = {}
#mvbw["CG1", "CG1"] = [
#  160009600,
#  124197480,
#  103916200,
#  93407760,
#  87771680,
#  85092480,
#  84247800,
#  81472240,
#  84856200,
#  81015200,
#  83846920,
#  79416000,
#  82198880
#]
#
#mvbw["CG1", "DG1"] = [
#  160009600,
#  140757144,
#  128856088,
#  122467952,
#  118749920,
#  117066624,
#  116910024,
#  113389200,
#  118272440,
#  113018720,
#  117038136,
#  110900032,
#  114822560,
#]
#
#mvbw["DG1", "CG1"] = [
#  479556672,
#  372082968,
#  311147160,
#  279458640,
#  262295040,
#  253880352,
#  250790280,
#  242079408,
#  251806968,
#  240031680,
#  248187192,
#  234727200,
#  242772480,
#]

mvbw = [
  479556672,
  471271128,
  460425624,
  453225072,
  447264480,
  444412800,
  444870600,
  431292048,
  449585592,
  429076320,
  444015672,
  420141888,
  434720160,
]

for disc_1 in discretisations:
    for disc_2 in discretisations:
        discretisation = "{}-{}".format(disc_1, disc_2)
        for i, layer in enumerate([1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]):
            timings = []

            data_volume = mvbw[i]

            opencl_file_name = "{}[opencl-greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
            opencl_file_path = os.path.join(file_dir, opencl_file_name)
            try:
                opencl_data = pandas.read_csv(open(opencl_file_path))
                profile_data = opencl_data.groupby('Profile')
                parloop_profile_data = profile_data['Assembly kernel']
                parloop_min = parloop_profile_data.min()
                timings.extend(map(lambda runtime: (data_volume / runtime) / 2**30, parloop_min[backends]))
            except IOError:
                timings.extend([float('NaN') for i in range(4)])

            df.loc[layer] = timings

        plot = df.plot(kind='line', colormap='gist_rainbow', marker='.')
        plot.set_xlabel('Layer Count')
        plot.set_ylabel('Maximum Valuable Bandwidth (GB/s)')
        figure = plot.get_figure()
        figure.tight_layout()
        figure.savefig(os.path.join(graph_dir, "{}-{}-{}-parloop-mvbw-increasing-layers.pdf".format(test_name, discretisation, mesh_size)))
