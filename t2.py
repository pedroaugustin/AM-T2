import math
import csv
import random

# inputs
k_clusters = 15

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

# normalizes the dataset (not necessary in the first case)
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

# RNG
def rng(max_rand):
    value = 0

    random.seed()
    value = int(random.random() * max_rand)

    return value

# generate k_clusters random centroids
def centroids():
    centroids = []
    
    for count in range(k_clusters):
        centroid = []
        centroid.append(rng(100000))
        centroid.append(rng(100000))
        centroids.append(list(centroid))
    
    return centroids



