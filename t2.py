import math
import csv
import random

# inputs
to_normalize = 1        # boolean option to normalize numbers
repetitions = 10        # numbers of times each test will be repeated

# function that calculate euclidean distance for two instances
def euclidean_d(point1x, point1y, point2x, point2y):
    distance = 0
    distance = math.sqrt( abs(point1x - point2x)**2 + abs(point1y - point2y)**2 )
    return distance

def euclidean_d2(point1, point2):
    distance = 0
    distance = math.sqrt( abs(point1[0] - point2[0])**2 + abs(point1[1] - point2[1])**2 )
    return distance

# open TXT file
instances = []
with open('test_instances.txt') as txt_file: # 5000 instances
    csv_reader = csv.reader(txt_file, delimiter=' ')
    for row in csv_reader:
        while len(row) > 2: # remove blank parameters
            row.pop(row.index(''))
        instances.append(list(row))

# converts instances to int
for row in instances:
    row[0] = int(row[0])
    row[1] = int(row[1])

print(instances[0])
print(instances[4999])
print(euclidean_d2(instances[0], instances[4999]))
