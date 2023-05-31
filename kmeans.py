from collections import defaultdict
from random import uniform
from math import sqrt
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centers = []

    def point_avg(self, points):
        dimensions = len(points[0])
        new_center = []

        for dimension in range(dimensions):
            dim_sum = 0
            for p in points:
                dim_sum += p[dimension]

            new_center.append(dim_sum / float(len(points)))

        return new_center

    def update_centers(self, data_set, assignments):
        new_means = defaultdict(list)
        centers = []
        for assignment, point in zip(assignments, data_set):
            new_means[assignment].append(point)

        for points in new_means.values():
            centers.append(self.point_avg(points))

        self.centers = centers

    def assign_points(self, data_points):
        assignments = []
        for point in data_points:
            shortest = float('inf')
            shortest_index = 0
            for i in range(len(self.centers)):
                val = self.distance(point, self.centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            assignments.append(shortest_index)
        return assignments

    def distance(self, a, b):
        dimensions = len(a)
        _sum = 0
        for dimension in range(dimensions):
            difference_sq = (a[dimension] - b[dimension]) ** 2
            _sum += difference_sq
        return sqrt(_sum)

    def generate_k(self, data_set):
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

        for _k in range(self.k):
            rand_point = []
            for i in range(dimensions):
                min_val = min_max['min_%d' % i]
                max_val = min_max['max_%d' % i]
                rand_point.append(uniform(min_val, max_val))

            centers.append(rand_point)

        self.centers = centers

    def fit(self, dataset):
        self.generate_k(dataset)

        assignments = self.assign_points(dataset)
        old_assignments = None

        while assignments != old_assignments:
            self.update_centers(dataset, assignments)
            old_assignments = assignments
            assignments = self.assign_points(dataset)

        return list(zip(assignments, dataset)), self.centers


# Example usage
points = [
    [1, 2], [2, 1], [3, 1],
    [5, 4], [5, 5], [6, 5],
    [10, 8], [7, 9], [11, 5],
    [14, 9], [14, 14], [9, 12]
]

kmeans = KMeans(k=2)
fit, centers = kmeans.fit(points)

colors = ['green', 'red']

# Plotting the points
for i in range(len(points)):
    plt.plot(points[i][0], points[i][1], color=colors[fit[i][0]], marker='o', markersize=10)

# Plotting the centroids
for i in range(len(centers)):
    plt.plot(centers[i][0], centers[i][1], color=colors[i], marker='x', markersize=10)

plt.show()
