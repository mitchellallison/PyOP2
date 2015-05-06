#!/usr/bin/env python
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

# With runtime
grouped_profile_data = data.groupby('Profile')
runtime_figure = grouped_profile_data.mean().plot(kind='barh', stacked=True).get_figure()
runtime_figure.set_size_inches(20, 10)
runtime_figure.savefig(os.path.join(graph_dir, "{}_runtime_comparison.pdf".format(file_name)))
