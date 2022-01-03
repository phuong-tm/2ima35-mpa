import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from sklearn.cluster import kmeans_plusplus


class Coreset:
    def __init__(self, k, eps, a, grid=True):
        self.k = k
        self.eps = eps
        self.a = a
        self.grid = grid

    def compute_coreset(self, points, centers, sample_size=10):
        coreset = np.array([[0, 0, 0]])

        D = [np.sqrt(np.power(points[:, 0:2] - center, 2).sum(axis=1)) for center in centers]
        cost = np.sum([point[2] * np.sqrt(np.sum(np.power(point[0:2] - centers, 2))) for point in points])
        levels = int(np.ceil(np.log(len(points)) * np.log(self.a * np.log(len(points)))))
        radius = np.sqrt(cost / (self.a * np.log(len(points)) * len(points)))

        last_radius = 0
        current_radius = radius
        for i in range(levels):
            if len(points) == 0:
                print("Stopping at level", i, "out of", levels)
                break

            for c in range(len(centers)):
                index = np.where((last_radius < D[c]) & (D[c] < current_radius))
                annulus_subset = points[index[0]]
                combined_sub = 0

                # grid sampling
                if self.grid:
                    side_length = (self.eps * current_radius) / np.sqrt(2)
                    grid_length = np.ceil((current_radius * 2) / side_length)

                    x_range = np.arange(centers[c][0] - (side_length * grid_length) / 2.0,
                                        #np.ceil(centers[c][0] + (side_length * grid_length) / 2.0), side_length)
                                        centers[c][0] + 1.0 + (side_length * (grid_length + 1)) / 2.0, side_length)
                    y_range = np.arange(centers[c][1] - (side_length * grid_length) / 2.0,
                                        #np.ceil(centers[c][1] + (side_length * grid_length) / 2.0), side_length)
                                        centers[c][1] + (side_length * (grid_length + 1)) / 2.0, side_length)
                    #if len(x_range) < grid_length - 1 or len(y_range) < grid_length - 1:
                    #    print(grid_length, side_length, centers[c], x_range, y_range)

                    for x_i in range(len(x_range) - 1):
                        for y_i in range(len(y_range) - 1):
                            subset = annulus_subset[
                                (annulus_subset[:, 0] > x_range[x_i]) & (annulus_subset[:, 0] < x_range[x_i + 1]) & (
                                        annulus_subset[:, 1] > y_range[y_i]) & (
                                            annulus_subset[:, 1] < y_range[y_i + 1])]
                            combined_sub += len(subset)
                            if len(subset) != 0:
                                sample_id = np.random.choice(subset.shape[0], 1, replace=False)
                                coreset_point = subset[sample_id]
                                coreset_point[0][2] = np.sum(subset[:, 2])
                                coreset = np.append(coreset, coreset_point, axis=0)

                # annulus sampling
                else:
                    sample_size_current = annulus_subset.shape[0] if sample_size > annulus_subset.shape[
                        0] else sample_size
                    if sample_size_current == 0:
                        continue
                    index_subset = np.random.choice(annulus_subset.shape[0], int(sample_size_current), replace=False)
                    subset = points[index_subset]
                    subset[:, 2] = np.sum(annulus_subset[:, 2]) / sample_size_current
                    coreset = np.append(coreset, subset, axis=0)

                # remove points from annulus subset in overall points set as they shouldn't be considered in other balls
                points = np.delete(points, index[0], axis=0)
                for c_id in range(len(centers)):
                    D[c_id] = np.delete(D[c_id], index[0], axis=0)

            last_radius = current_radius
            current_radius = np.power(2, i + 1) * radius

        return coreset[1:]

    def compute(self, points, sample_size=10):
        points = points.astype(np.float32)  # ensure type is large enough as all points will be squared
        unweighted_points = points[:, 0:2]

        # compute centers according to kmeans++
        centers = kmeans_plusplus(unweighted_points, self.k)[0]

        # Comupte coreset
        coreset = self.compute_coreset(points, centers, sample_size)

        # plt.scatter(coreset[:, 0], coreset[:, 1])
        # plt.scatter(centers[:, 0], centers[:, 1])
        # plt.show()

        return coreset

    def mpc_compute(self, data, machines, seed, sample_size=10):
        random.seed(seed)

        # split points
        point_sets = []
        points = data.copy()
        random.shuffle(points)
        for i in range(len(points)):
            if i < machines:
                point_sets.append([np.array(points[i], dtype=np.float32)])
            else:
                point_sets[i % machines] = np.append(point_sets[i % machines], [points[i]], axis=0)

        x = [np.sum(c[:, 2]) for c in point_sets]
        print(sum(x), x)

        spark: SparkSession = SparkSession.builder \
            .master("local[" + str(machines) + "]") \
            .appName("SparkDB") \
            .getOrCreate()

        iterations = 0
        while True:
            iterations += 1
            coreset_sets = spark.sparkContext.parallelize(point_sets).map(lambda x: self.compute(x, sample_size)).collect()

            if len(coreset_sets) == 1:
                print("Finished mpc coreset computation in", iterations, "rounds\nFinal coreset contains",
                      len(coreset_sets[0]), "points")
                spark.stop()  # close session
                return coreset_sets[0]

            x = [np.sum(c[:, 2]) for c in coreset_sets]
            print(sum(x), x)

            point_sets = []
            for i in range(math.floor(len(coreset_sets) / 2)):
                # Combine two adjacent core
                point_sets.append(coreset_sets[i * 2])
                point_sets[i] = np.append(point_sets[i], coreset_sets[i * 2 + 1], axis=0)

            # If we have an odd amount of coreset sets, we union the last three
            if len(coreset_sets) % 2 != 0:
                point_sets[-1] = np.append(point_sets[-1], coreset_sets[-1], axis=0)
