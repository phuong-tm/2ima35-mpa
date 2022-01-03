import common.input as input
import coreset as agc
import common.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

# Output data format: Grid or Ball sampling, sklearn dataset type, number of points in the dataset, seed used,
#                      sample size for Ball sampling, total runtime, array of runtime per round/iterations,
#                      number of points in the resulting coreset, quality of solution
output_data = pd.DataFrame([], columns=["SamplingMethod", "DatasetType", "NrOfPoints", "Seed", "SampleSize", "TotalTime", "RoundTimes", "CoresetSize", "Quality"])
row_counter = 0  # keep track of row in dataframe

epsilon = 1/2
a = 0.5
machines = 8
seeds = [] # [100, 101, 102, 103, 104]
sample_sizes = []  # TODO: to be determined

# TODO: dataset generation
datasets = [] # or set of filenames


for seed in seeds:
    for dataset in datasets:
        # assuming data only contains x and y, add weights of all ones
        dataset = np.append(dataset, np.ones(len(dataset)).reshape(len(dataset), 1), axis=1)
        centers = 0  # TODO: obtain number of centers/clusters for each dataset

        # kmeans
        kmeans = KMeans(n_clusters=centers, init='k-means++', random_state=seed, max_iter=50)  # TODO: max_iter?

        # run kmeans on original/complete data
        kmeans_cluster = kmeans.fit(X=dataset[:, 0:2], sample_weight=dataset[:, 2])

        # Run grid sampling
        geo = agc.Coreset(centers, epsilon, a, True)
        grid_coreset, total_time, round_times = geo.mpc_compute(dataset, machines, seed)
        grid_cluster = kmeans.fit(X=grid_coreset[:, 0:2], sample_weight=grid_coreset[:, 2])
        # TODO: - Compare quality against original kmeans centroid position
        output_data.loc[row_counter] = ["Grid", "dataset type", "n", seed, 0, total_time, round_times, len(grid_coreset),
                                        "quality"]  # TODO: Fill dataset type, n and quality variables
        output_data.to_csv("output_data.csv", index=False)
        row_counter += 1

        for sample_size in sample_sizes:
            # Run ball sampling
            geo = agc.Coreset(centers, epsilon, a, False)
            ball_coreset, total_time, round_times = geo.mpc_compute(dataset, machines, seed)
            ball_cluster = kmeans.fit(X=ball_coreset[:, 0:2], sample_weight=ball_coreset[:, 2])
            # TODO: - Compare quality against original kmeans centroid position
            output_data.loc[row_counter] = ["Grid", "dataset type", "n", seed, 0, total_time, round_times, len(ball_coreset), "quality"]  # TODO: Fill dataset type, n and quality variables
            output_data.to_csv("output_data.csv", index=False)
            row_counter += 1

#plt.scatter(kmeans_cluster.cluster_centers_[:,0], kmeans_cluster.cluster_centers_[:, 1], s=0.5)
#plt.scatter(grid_cluster.cluster_centers_[:,0], grid_cluster.cluster_centers_[:, 1], s=0.5)
#plt.scatter(ball_cluster.cluster_centers_[:, 0], ball_cluster.cluster_centers_[:, 1], s=0.5)
#plt.show()

'''
@utils.timeit
def test_no_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(X=data)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    cost = utils.cost_function(data, kmeans.labels_, kmeans.cluster_centers_)
    return cost

@utils.timeit
def test_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(X=coreset)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    cost = utils.cost_function(data, kmeans.predict(X=data), kmeans.cluster_centers_)
    return cost


cost = test_no_coreset()
cost_cs = test_coreset()
cost_opt = utils.cost_function(data, opt, centers)

print("cost no coreset ", cost)
print("cost coreset ", cost_cs)
print("coreset improvment: {:.1%} ".format(np.abs(cost-cost_cs)/cost))
plt.show()
'''
