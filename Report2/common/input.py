import numpy as np
import csv

def parse_txt(path) -> np.ndarray:
    data = np.loadtxt(path, dtype = np.float64)
    return data