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

sizes = [2, 3, 4, 5, 5.585, 6, 7, 8, 9, 10]
scales = [2.0**(i + 10) for i in sizes]
scales_KB = [2.0**i for i in sizes]
scale_labels = ["{0:.0f}".format(i) for i in scales]

discretisations = ["CG1", "DG1"]

execution_schemes = ["C", "D"]

for disc_1 in discretisations:
    for disc_2 in discretisations:
        execution_scheme_parloop_c_df = pandas.DataFrame(columns=scales_KB)
        execution_scheme_parloop_d_df = pandas.DataFrame(columns=scales_KB)

        execution_scheme_plan_c_df = pandas.DataFrame(columns=scales_KB)
        execution_scheme_plan_d_df = pandas.DataFrame(columns=scales_KB)

        execution_schemes_parloop_df = [execution_scheme_parloop_c_df, execution_scheme_parloop_d_df]
        execution_schemes_plan_df = [execution_scheme_plan_c_df, execution_scheme_plan_d_df]

        discretisation = "{}-{}".format(disc_1, disc_2)
        for layer in [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]:
                opencl_file_name = "{}[opencl-greedy-{}-{}-{}]".format(test_name, discretisation, mesh_size, layer)
                opencl_file_path = os.path.join(file_dir, opencl_file_name)
                try:
                    opencl_data = pandas.read_csv(open(opencl_file_path))
                    profile_data = opencl_data.groupby('Profile')
                    parloop_profile_data = profile_data['Assembly kernel']
                    parloop_min = parloop_profile_data.min()
                    plan_profile_data = profile_data['Plan construction']
                    plan_min = plan_profile_data.min()
                    for i, execution_scheme in enumerate(execution_schemes):
                        labels = map(lambda label: "'{} {}'".format(label, execution_scheme), scale_labels)
                        execution_schemes_parloop_df[i].loc[layer] = parloop_min[labels].tolist()
                        execution_schemes_plan_df[i].loc[layer] = plan_min[labels].tolist()
                except IOError:
                    pass

        for i, execution_scheme in enumerate(execution_schemes):
            with open(os.path.join(graph_dir, "{}-{}-{}-Scheme_{}-parloop-partition-size-increasing-layers-table.tex".format(test_name, discretisation, mesh_size, execution_scheme)), 'w') as f:
                latex = execution_schemes_parloop_df[i].transpose().to_latex()
                f.write(latex)

            with open(os.path.join(graph_dir, "{}-{}-{}-Scheme_{}-plan-partition-size-increasing-layers-table.tex".format(test_name, discretisation, mesh_size, execution_scheme)), 'w') as f:
                latex = execution_schemes_plan_df[i].transpose().to_latex()
                f.write(latex)
