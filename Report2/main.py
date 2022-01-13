import coreset as agc
import data_generation as generator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

def compute_cost(cluster_centroids_1, cluster_centroids_2):
    """
    Compute distance matrix between original kmeans centroids and computed centroid
    """ 
    #compute Euclidean distance matrix between original kmeans centroids and computed centroid
    dist_matrix = distance_matrix(cluster_centroids_1,cluster_centroids_2)
    # sum all distances
    sum_cost = np.sum(dist_matrix)
    avg_cost = sum_cost/len(cluster_centroids_1)
    return avg_cost

def compute_cost2(cluster_centroids_1, cluster_centroids_2):
    """
    Compute euclidean distance from original kmeans centroids and computed centroid
    """
    return np.linalg.norm(cluster_centroids_1 - cluster_centroids_2)

# Output data format: Grid or Ball sampling, sklearn dataset type, number of points in the dataset, seed used,
#                      sample size for Ball sampling, total runtime, array of runtime per round/iterations,
#                      number of points in the resulting coreset, quality of solution
output_data = pd.DataFrame([], columns=["SamplingMethod", "DatasetType", "NrOfPoints", "Seed", "SampleSize", "TotalTime", "RoundTimes", "CoresetSize", "AvgQuality", "NormQuality"])
row_counter = 0  # keep track of row in dataframe

epsilon = 1/4
a = 0.5
machines = 8
seeds = [100, 101, 102, 105, 106]
kmeans_max_iter = 50

datasets = generator.load_datasets()
for seed in seeds:
    print("seed", seed)
    for datasetType, (dataset, label) in datasets:
        print(datasetType, len(dataset))
        figure1, axis1 = plt.subplots()
        figure2, axis2 = plt.subplots()
        axis2.scatter(dataset[:,0], dataset[:,1], s=0.5, c='blue')

        # assuming data only contains x and y, add weights of all ones
        dataset = np.append(dataset, np.ones(len(dataset)).reshape(len(dataset), 1), axis=1)
        
        centers = len(np.unique(label))  # number of centers/clusters for each dataset

        # kmeans
        kmeans = KMeans(n_clusters=centers, init='k-means++', random_state=seed, max_iter=kmeans_max_iter)

        # run kmeans on original/complete data
        kmeans_cluster = kmeans.fit(X=dataset[:, 0:2], sample_weight=dataset[:, 2])
        kmeans_centroids = kmeans_cluster.cluster_centers_
        axis1.scatter(kmeans_cluster.cluster_centers_[:, 0], kmeans_cluster.cluster_centers_[:, 1], marker='x', c='tab:red')
        axis2.scatter(kmeans_cluster.cluster_centers_[:, 0], kmeans_cluster.cluster_centers_[:, 1], marker='x', c='tab:red')

        # Run grid sampling
        geo = agc.Coreset(centers, epsilon, a, True)
        grid_coreset, total_time, round_times = geo.mpc_compute(dataset, machines, seed)
        grid_cluster = kmeans.fit(X=grid_coreset[:, 0:2], sample_weight=grid_coreset[:, 2])
        grid_centroids = grid_cluster.cluster_centers_
        axis1.scatter(grid_centroids[:, 0], grid_centroids[:, 1], marker='s', c='tab:blue')
        axis2.scatter(grid_centroids[:, 0], grid_centroids[:, 1], marker='s', c='tab:blue')

        # compute cost against original kmeans centroid position
        quality1 = compute_cost(kmeans_centroids, grid_centroids)
        quality2 = compute_cost2(kmeans_centroids, grid_centroids)

        # log results
        output_data.loc[row_counter] = ["Grid", datasetType, len(dataset), seed, 0, total_time, round_times, len(grid_coreset),
                                        quality1, quality2]
        output_data.to_csv("output_data.csv", index=False)
        row_counter += 1
        
        sample_sizes = list(np.array([0.001, 0.01, 0.1])*len(dataset)) #sample sizes vary from 0.1%, 1% and 10% of NrOfPoints
        colors = ['tab:orange', 'tab:green', 'tab:purple']
        for i in range(len(sample_sizes)):
            # Run ball sampling
            geo = agc.Coreset(centers, epsilon, a, False)
            ball_coreset, total_time, round_times = geo.mpc_compute(dataset, machines, seed)
            ball_cluster = kmeans.fit(X=ball_coreset[:, 0:2], sample_weight=ball_coreset[:, 2])
            ball_centroids = ball_cluster.cluster_centers_
            axis1.scatter(ball_centroids[:, 0], ball_centroids[:, 1], c=colors[i])
            axis2.scatter(ball_centroids[:, 0], ball_centroids[:, 1], c=colors[i])

            # compute cost against original kmeans centroid position
            quality1 = compute_cost(kmeans_centroids, ball_centroids)
            quality2 = compute_cost2(kmeans_centroids, ball_centroids)
            output_data.loc[row_counter] = ["Ball", datasetType, len(dataset), seed, sample_sizes[i], total_time, round_times, len(ball_coreset), quality1, quality2]
            output_data.to_csv("output_data.csv", index=False)
            row_counter += 1

        axis1.legend(labels=['kmeans', 'grid', 'ball 0.1%', 'ball 1%', 'ball 10%'])
        axis2.legend(labels=['data', 'kmeans', 'grid', 'ball 0.1%', 'ball 1%', 'ball 10%'], loc="best")
        axis1.set_title(datasetType + " dataset of " + str(len(dataset)) + " point")
        axis2.set_title(datasetType + " dataset of " + str(len(dataset)) + " point")
        figure1.savefig("images/" + datasetType + "_" + str(len(dataset)) + "_" + str(seed) + "_1.png")
        figure2.savefig("images/" + datasetType + "_" + str(len(dataset)) + "_" + str(seed) + "_2.png")
        figure1.clf()
        figure2.clf()
