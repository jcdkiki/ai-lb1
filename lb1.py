import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import itertools
import math

DEFAULT_RANDOMNESS = 10
DEFAULT_N_POINTS = 64

def rand_vector(r):
    return np.array([
        random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r),
        random.uniform(-r, r), random.uniform(-r, r)]
    )

def generate_line_point(x):
    v0 = np.array([43, -14, 17, 2, 50])
    v1 = np.array([15, -19, 2, -1, 1])
    return v1*x*100000 + v0

def generate_plane_point(x, y):
    v0 = np.array([432, 234, 23, 515, -100])
    v1 = np.array([-10, 40, 20, 10, 80])
    v2 = np.array([40, 20, 10, 10, 10])
    return v0 + x*v1 + y*v2

def generate_random_point():
    return rand_vector(400) + np.array([100, 23, -50, 1, -100])

def process_dataset(data, n_components=2):
    data_scaled = StandardScaler().fit_transform(data)
    return PCA(n_components).fit_transform(data_scaled)

def clusterize(data, k=3):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--randomness", type=int, default=DEFAULT_RANDOMNESS)
    parser.add_argument("-s", "--seed", type=int, default=time.time())
    parser.add_argument("-n", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--line-scale", type=float, default=1)
    parser.add_argument("-i", "--inverse", action='store-true')

    args = parser.parse_args()

    random.seed(args.seed)
    datasets = {}
    
    points = []
    for x in range(math.ceil(math.sqrt(args.n))):
        for y in range(math.ceil(math.sqrt(args.n))):
            points.append(generate_plane_point(x, y))
    datasets["plane"] = np.array(points[0:args.n])

    datasets["line"] = np.array([generate_line_point(x * args.line_scale) for x in range(args.n)])

    datasets["random"] = np.array([generate_random_point() for _ in range(args.n)])

    for name, data in datasets.items():
        for i, j in itertools.product(range(args.n), range(5)):
            data[i][j] += random.uniform(-1, 1) * args.randomness
        result = process_dataset(data)
        print(result)

        plt.cla()
        
        k = 3
        labels, centroids = clusterize(result)
        for i in range(k):
            plt.plot(result[labels == i, 0], result[labels == i, 1], 'o')

        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')

        plt.savefig(f"plots/{name}.png")

if __name__ == "__main__":
    main()
