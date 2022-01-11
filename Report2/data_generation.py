import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets

np.random.seed(0)

def generating_datasets(n_samples=1500, random_state=170):
    """
    Generating sklearn datasets.
    
    Returns: list of datasets
    """
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    #no_structure = np.random.rand(n_samples, 2), None

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
    data_sets = [("Noisy circle", noisy_circles),
                 ("Noisy moon", noisy_moons),
                 ("Blobs", blobs),
                 ("Aniso", aniso),
                 ("varied", varied),
                 ("Gaussian quantile", gaussian_quantile)]
    return data_sets

def load_datasets(sample_sizes = [1500, 10000, 50000]):
    """
    Load all datasets.
    Return a list of tuples (datasetType, dataset, label)
    """
    data_sets = []
    for size in sample_sizes:
        data_sets.extend(generating_datasets(n_samples=size))
    return data_sets