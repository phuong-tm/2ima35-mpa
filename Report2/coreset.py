import numpy as np
import logging, sys
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import kmeans_plusplus
from progress.bar import Bar



class Coreset:
    def __init__(self, data, k, eps, a, grid=True):
        self.X_w = data.astype(np.int64)
        self.X = data[:,0:2].astype(np.int64)
        self.k = k
        self.eps = eps
        self.a = a
        self.grid = grid

    def _compute_eps_cover(self, centers):
        coreset = np.array([[0, 0, 0]])
        #fig, ax = plt.subplots()

        sample_size = 5  # TODO
        points = self.X_w  # np.append(self.X_w, np.ones(len(self.X_w)).reshape(len(self.X_w), 1), axis=1)   # append weights of one
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

                # grid sampling
                if self.grid:
                    side_length = (self.eps * current_radius) / np.sqrt(2)
                    grid_length = np.ceil((current_radius * 2) / side_length)

                    x_range = np.arange(centers[c][0] - side_length * grid_length / 2,
                                        centers[c][0] + side_length * grid_length / 2, side_length)
                    y_range = np.arange(centers[c][1] - side_length * grid_length / 2,
                                        centers[c][1] + side_length * grid_length / 2, side_length)

                    for x_i in range(len(x_range) - 1):
                        for y_i in range(len(y_range) - 1):
                            subset = annulus_subset[(annulus_subset[:, 0] > x_range[x_i]) & (annulus_subset[:, 0] < x_range[x_i + 1]) & (
                                        annulus_subset[:, 1] > y_range[y_i]) & (annulus_subset[:, 1] < y_range[y_i + 1])]
                            if len(subset) != 0:
                                sample_id = np.random.choice(subset.shape[0], 1, replace=False)
                                coreset_point = subset[sample_id]
                                coreset_point[0][2] = np.sum(subset[:,2])
                                coreset = np.append(coreset, coreset_point, axis=0)
                                for point in subset:
                                    index = np.where((points[:,0] == point[0]) & (points[:,1] == point[1]))[0]
                                    points = np.delete(points, index, axis=0)
                                    for c_id in range(len(centers)):
                                        D[c_id] = np.delete(D[c_id], index, axis=0)

                # annulus sampling
                else:
                    sample_size_current = annulus_subset.shape[0] if sample_size > annulus_subset.shape[
                        0] else sample_size
                    index_subset = np.random.choice(annulus_subset.shape[0], int(sample_size_current), replace=False)
                    coreset = np.append(coreset, points[index_subset], axis=0)
                    points = np.delete(points, index_subset, axis=0)
                    for c_id in range(len(centers)):
                        D[c_id] = np.delete(D[c_id], index_subset, axis=0)

            last_radius = current_radius
            current_radius = np.power(2, i+1) * radius

        return coreset[1:]

    def compute(self):
        # Compute polynomial approximation a*OPT -> centers
        centers = kmeans_plusplus(self.X, self.k)[0]

        # Comupte eps ball cover for each center in centers
        coreset = self._compute_eps_cover(centers)

        #plt.scatter(coreset[:, 0], coreset[:, 1])
        #plt.scatter(centers[:, 0], centers[:, 1])
        #plt.show()

        print("Number of coreset points", len(coreset))
        return coreset