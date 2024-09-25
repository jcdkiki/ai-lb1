import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import integrate
import itertools
import math
import os
import shutil

DEFAULT_RANDOMNESS = 10
DEFAULT_N_POINTS = 64
DEFAULT_LINE_SCALE = 1
DEFAULT_N_CLUSTERS = 3

def rand_vector(r):
    return np.array([
        random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r),
        random.uniform(-r, r), random.uniform(-r, r)], dtype='float64'
    )

def generate_line_point(x):
    v0 = np.array([43, -14, 17, 2, 50])
    v1 = np.array([15, -19, 2, -1, 1])
    return v1*x + v0

def generate_plane_point(x, y):
    v0 = np.array([432, 234, 23, 515, -100])
    v1 = np.array([-10, 40, 20, 10, 80])
    v2 = np.array([40, 20, 10, 10, 10])
    return v0 + x*v1 + y*v2

def generate_random_point():
    return rand_vector(400) + np.array([100, 23, -50, 1, -100])

def generate_polynomial_point(x):
    return np.array([10*x, 13*x*x, 54*x, 5*x + 30*x*x, 90*x*x*x*x]) + np.array([432, 54, 12321, 76, -1000])

def clusterize(data, k, seed):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--randomness", type=int, default=DEFAULT_RANDOMNESS)
    parser.add_argument("-s", "--seed", type=int, default=int(time.time()))
    parser.add_argument("--line-scale", type=float, default=DEFAULT_LINE_SCALE)
    parser.add_argument("-c", "--clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("-S", "--cluster-seed", type=int, default=int(time.time()))

    args = parser.parse_args()

    random.seed(args.seed)
    datasets = {}

    points = []
    for x in range(8):
        for y in range(8):
            points.append(generate_plane_point(x, y))
    datasets["plane"] = np.array(points, dtype='float64')

    datasets["line"] = np.array([generate_line_point(x * args.line_scale) for x in range(50)], dtype='float64')

    datasets["random"] = np.array([generate_random_point() for _ in range(50)], dtype='float64')

    datasets["polynomial"] = np.array([generate_polynomial_point((x - 20) * args.line_scale) for x in range(50)], dtype='float64')

    shutil.rmtree("plots", ignore_errors=True)
    os.makedirs("plots", exist_ok=True)

    for name, data in datasets.items():
        for i in (range(len(data))):
            data[i] += rand_vector(args.randomness)
        
        data_scaled = StandardScaler().fit_transform(data)
        result = PCA(2).fit_transform(data_scaled)

        print(result)

        summary = {}

        for k in range(1, args.clusters + 1):
            plt.cla()
            labels, centroids, inertia = clusterize(result, k, args.cluster_seed)
            for i in range(k):
                plt.plot(result[labels == i, 0], result[labels == i, 1], 'o')

            summary[k] = inertia

            plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
            plt.savefig(f"plots/{name}-{k}.png")
        
        plt.cla()
        plt.plot(list(summary.keys()), list(summary.values()))
        plt.savefig(f"plots/{name}-summary.png")


if __name__ == "__main__":
    main()
