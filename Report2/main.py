import common.input as input
import coreset as agc
import common.utils as utils
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

data = input.parse_txt("dataset/s-set/s1.txt")
data = np.append(data, np.ones(len(data)).reshape(len(data), 1), axis=1)
opt = input.parse_txt("dataset/s-set/s1-label.pa")
centers = input.parse_txt("dataset/s-set/s1-cb.txt")

#Computing geometric decomposition coreset
geo = agc.Coreset(15, 0.50, 0.5, True)
coreset = geo.mpc_compute(data, 8, 0)
print(len(coreset))
print(coreset)
print(np.sum(coreset[:,2]))

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


#cost = test_no_coreset()
#cost_cs = test_coreset()
#cost_opt = utils.cost_function(data, opt, centers)

#print("cost no coreset ", cost)
#print("cost coreset ", cost_cs)
#print("coreset improvment: {:.1%} ".format(np.abs(cost-cost_cs)/cost))
plt.show()
