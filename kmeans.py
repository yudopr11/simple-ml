from collections import defaultdict
from random import uniform
from math import sqrt
import numpy as np

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.

    Compute the center for each of the assigned groups.

    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = float('inf')  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Use euclidian distance to calculate distance of two points
    a and b.
    """
    dimensions = len(a)
    
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    """
    The K-Means algorithm follows a straightforward process:
    1. Initialization: Choose the number of clusters (K) and 
       randomly initialize K data points as the initial centroids.
    2. Assignment: Assign each data point to the nearest centroid 
       based on a distance metric (typically Euclidean distance). 
       This step forms the clusters.
    3. Update: Recalculate the centroids as the mean of all data 
       points belonging to each cluster.
    4. Iteration: Repeat steps 2 and 3 until convergence or until 
       a maximum number of iterations is reached. Convergence occurs 
       when the centroids no longer change significantly or when the 
       predefined number of iterations is reached.
    5. Result: After convergence, the final output is a set of optimized 
       clusters, with each data point belonging to a specific cluster.
    """
    max_iter = 10000
    iter = 1
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments or iter == max_iter:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        iter += 1
    return list(zip(assignments, dataset)), new_centers


points = [
    [1, 2], [2, 1], [3, 1],
    [5, 4], [5, 5], [6, 5],
    [10, 8], [7, 9], [11, 5],
    [14, 9], [14, 14], [9, 12]
    ]

fit, cntr = k_means(points, 2)

colors = ['green', 'red']

# Plotting the points
for i in range(len(points)):
    plt.plot(points[i][0], points[i][1], color=colors[fit[i][0]], marker='o', markersize=10)

# Plotting the centroids
for i in range(len(cntr)):
    plt.plot(cntr[i][0], cntr[i][1], color=colors[i], marker='x', markersize=10)

plt.show()
