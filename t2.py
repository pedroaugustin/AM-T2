import math
import csv
import random

# function that calculate euclidean distance for two instances
def euclidean_d(point1x, point1y, point2x, point2y):
    distance = 0
    distance = math.sqrt( abs(point1x - point2x)**2 + abs(point1y - point2y)**2 )
    return distance

def euclidean_d2(point1, point2):
    distance = 0
    distance = math.sqrt( abs(point1[0] - point2[0])**2 + abs(point1[1] - point2[1])**2 )
    return distance

# open TXT file and adjust the dataset
def dataset():
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
    
    return instances

# normalizes the dataset
def normalize(data_list, printer):
    max_attr = list(data_list[0])
    min_attr = list(data_list[0])

    for row in data_list:
        for col in range(2):
            if row[col] > max_attr[col]:
                max_attr[col] = row[col]
            if row[col] < min_attr[col]:
                min_attr[col] = row[col]
    for row in data_list:
        for col in range(2):
            row[col] = (row[col] - min_attr[col]) / (max_attr[col] - min_attr[col])
    
    if(printer == 1):
        print("  maximo [0]: ", end='')
        print(max_attr[0], end='')
        print(", minimo [0]: ", end='')
        print(min_attr[0])
        print("  maximo [1]: ", end='')
        print(max_attr[1], end='')
        print(", minimo [1]: ", end='')
        print(min_attr[1])
    
    return data_list

instances = dataset()
print(instances[0])
print(instances[4999])
print(euclidean_d2(instances[0], instances[4999]))

instances = normalize(instances, 1)
print(instances[0])
print(instances[4999])
print(euclidean_d2(instances[0], instances[4999]))
