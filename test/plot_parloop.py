#!/usr/bin/env python
import sys
import os

import matplotlib.pyplot as plt
import pandas

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

if len(sys.argv) != 3:
    print "Usage: ./ploy.py FILE_PATH OUTPUT_DIR."
    exit()

file_path = os.path.abspath(sys.argv[1])
file_name = os.path.basename(os.path.normpath(sys.argv[1]))
input_file = open(file_path)

graph_dir = os.path.abspath(sys.argv[2])

data = pandas.read_csv(input_file)

# Without runtime
parloop_profile_data = data.groupby('Profile')
plot = parloop_profile_data['ParLoop kernel'].mean().plot(kind='barh', stacked=False)
plot.set_xlabel('Time (Seconds)')
plot.set_ylabel('Backend')
title = plot.set_title('Time spent in parallel loop while performing an extruded RHS assembly for a given backend', fontsize=14, fontweight='bold')
title.set_y(1.04)
figure = plot.get_figure()
figure.set_size_inches(20, 10)
figure.savefig(os.path.join(graph_dir, "{}_parloop_comparison.pdf".format(file_name)))
