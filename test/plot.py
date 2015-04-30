#!/usr/bin/python
import sys
import os
import csv

if len(sys.argv) != 2:
    print "Usage: ./ploy.py FILE_PATH."
    exit()

file_name = os.path.abspath(sys.argv[1])
file = open(file_name)

headers = []
profile_types = set()
data = {}
iterations = 0

for iteration, row in enumerate(csv.reader(file)):
    row = map(lambda string: string.strip(), row)
    if iteration == 0:
        headers = row[1:]
    else:
        profile_types.add(row[0])
        for j, datapoint in enumerate(headers):
            if (row[0], datapoint) in data:
                data[row[0], datapoint].append(float(row[j + 1]))
            else:
                data[row[0], datapoint] = [float(row[j + 1])]
            iterations += 1

iterations /= (len(headers) * len(profile_types))

averages = {}

for profile_type in profile_types:
    for datapoint in headers:
        sum = 0
        for i in range(iterations):
            sum += data[profile_type, datapoint][i]
        averages[profile_type, datapoint] = sum / iterations

print averages
