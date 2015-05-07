#!/usr/bin/env python
import sys
import os

from matplotlib import rc
import pandas

if len(sys.argv) != 3:
    print "Usage: ./ploy.py FILE_PATH OUTPUT_DIR."
    exit()

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

file_path = os.path.abspath(sys.argv[1])
file_name = os.path.basename(os.path.normpath(sys.argv[1]))
input_file = open(file_path)

graph_dir = os.path.abspath(sys.argv[2])

data = pandas.read_csv(input_file)

# With runtime
grouped_profile_data = data.groupby('Profile')
plot = grouped_profile_data.mean().plot(kind='barh', stacked=True)
plot.set_xlabel('Time (Seconds)')
plot.set_ylabel('Backend')
title = plot.set_title('Total runtime of the solution of an extruded RHS assembly problem for given backends', fontsize=14, fontweight='bold')
title.set_y(1.04)
figure = plot.get_figure()
figure.set_size_inches(20, 10)
figure.savefig(os.path.join(graph_dir, "{}_runtime_comparison.pdf".format(file_name)))
