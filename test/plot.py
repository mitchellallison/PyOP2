#!/usr/bin/python
import sys
import os

import matplotlib.pyplot as plt
import pandas

if len(sys.argv) != 2:
    print "Usage: ./ploy.py FILE_PATH."
    exit()

file_name = os.path.abspath(sys.argv[1])
file = open(file_name)

data = pandas.read_csv(file)

grouped_profile_data = data.groupby('Profile')

parloop_data = grouped_profile_data['ParLoop kernel']

from IPython import embed; embed()

parloop_data.mean().plot()

sys.exit()
