import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
def generating_datasets(n_samples=1500, random_state=170):
    """
    Generating sklearn datasets.
    
    Returns: list of datasets
    """
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    #gaussian quantile
    gaussian_quantile = datasets.make_gaussian_quantiles(n_samples=n_samples, random_state=random_state)
    data_sets = [noisy_circles, noisy_moons, blobs, no_structure, aniso, varied, gaussian_quantile]
    return data_sets

def dataset_to_txt():
    data_sets = generating_datasets()
    for i in range(5, len(data_sets)+5):
        data = data_sets[i-5]
        with open("dataset/s-set/s"+str(i)+".txt", 'w') as f:
            csv.writer(f, delimiter=' ').writerows(data[0].tolist())
        if (i-5)!=3:
            with open("dataset/s-set/s"+str(i)+"-label.pa", 'wb') as f:
                np.savetxt(f, data[1])

dataset_to_txt()