#!/usr/bin/python
import sys
import os

import matplotlib.pyplot as plt
import pandas

if len(sys.argv) != 3:
    print "Usage: ./ploy.py FILE_PATH OUTPUT_DIR."
    exit()

file_path = os.path.abspath(sys.argv[1])
file_name = os.path.basename(os.path.normpath(sys.argv[1]))
input_file = open(file_path)

graph_dir = os.path.abspath(sys.argv[2])

data = pandas.read_csv(input_file)

grouped_profile_data = data.groupby('Profile')

# With runtime
figure = grouped_profile_data.mean().plot(kind='bar', stacked=True).get_figure()
figure.savefig(os.path.join(graph_dir, "{}_runtime_comparison.pdf".format(file_name)))

# Without runtime
grouped_profile_data_without_runtime = data.drop('Runtime', 1).groupby('Profile')
figure = grouped_profile_data_without_runtime.mean().plot(kind='bar', stacked=True).get_figure()
figure.savefig(os.path.join(graph_dir, "{}_parloop_comparison.pdf".format(file_name)))
