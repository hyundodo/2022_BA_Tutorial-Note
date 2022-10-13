import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.graph import graph_shortest_path
from sklearn.datasets import load_iris
from sklearn import manifold, datasets


def distance_mat(X, n_neighbors=6):
    dis_matrix = np.zeros((len(X), len(X)))
    # compute Euclidean Distance
    for i in range(len(X)):
        for j in range(len(X)):
            dis_matrix[i, j] = np.sqrt(np.sum((X[i, :] - X[j, :])**2))
            
    # Keep 'n' nearest neighbors, others set to 0
    neighbors = np.zeros_like(dis_matrix)
    sort_distances = np.argsort(dis_matrix, axis=1)[:, 1:n_neighbors+1]
    for i in range(len(sort_distances)):
        for j in sort_distances:
            neighbors[i,j] = dis_matrix[i,j]
    return neighbors


def mds(distance_matrix, dim):
    # step2: extract the coordinates that preserve the distance information
    # compute inner product matrix B 
    n = len(distance_matrix)
    H = np.identity(n) - (1/n)*(np.ones(n))
    B = -0.5 * np.matrix(H) * np.matrix(distance_matrix*distance_matrix) * np.matrix(H)
    
    eigen_value, eigen_vector = np.linalg.eigh(B)
    
    # sort in descending order
    idx = np.argsort(eigen_value)[::-1]
    eigen_value = eigen_value[idx]
    eigen_vector = eigen_vector[:, idx]
    
    # compute positive semi-definite and of rank p
    idx_pos = np.where(eigen_value > 0)
    L = np.diag(np.sqrt(eigen_value[idx_pos]))
    V = np.squeeze(eigen_vector[:, idx_pos])
    # the coordinate matrix X
    X = np.matrix(V)*np.matrix(L)
    
    coordinate = []
    for i in range(dim):
        coord_value = []
        for j in range(len(distance_matrix)):
            coord_value.append(X[j, i])
        coordinate.append(coord_value)
    
    return coordinate


def isomap(data, n_components=2, n_neighbors=6):
    # Compute distance matrix
    neighbors = distance_mat(data, n_neighbors)

    # Compute shortest paths from distance matrix
    graph = graph_shortest_path(neighbors, directed=False)
    graph = -0.5 * (graph ** 2)
    
    # Return the MDS projection
    result_mds = mds(graph, n_components)

    return result_mds

def visualize(Y, label, dim=2):
    # express 2 dimension
    if dim == 2:
        fig, ax = plt.subplots()
        ax.scatter(Y[0], Y[1], c=label, cmap=plt.cm.Spectral)
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=label, cmap=plt.cm.Spectral)
    
# demo
def main():
    input_x, color = datasets.make_swiss_roll(n_samples=1500)
    visualize(input_x, color, dim=3)
    print('compute distance Matrix...')
    print('compute MDS projection with shortest graph...')
    Y = isomap(input_x)
    
    print('Visualize')
    #visualize(Y, input_y, input_name)
    visualize(Y, color, dim=2)
    
if __name__ == '__main__':
    main()