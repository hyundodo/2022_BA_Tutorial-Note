import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


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
    return neighbors, sort_distances


def LLE(data, neighbors_idx, n_components=2):
    n = data.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        # the neighbors matrix
        k_indexes = neighbors_idx[i, :]
        neighbors = data[k_indexes, :] - data[i, :]
        
        # compute the corresponding gram matrix
        gram_inv = np.linalg.pinv(np.dot(neighbors, neighbors.T))

        # setting the weight val
        lambda_par = 2/np.sum(gram_inv)
        w[i, k_indexes] = lambda_par*np.sum(gram_inv, axis=1)/2
        
    m = np.subtract(np.eye(n), w)
    values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
    Y = u[:, 1:n_components+1]
    
    return Y


def visualize(Y, color):
    # express 2 dimension
    fig, ax = plt.subplots()
    ax.scatter(Y[:,0], Y[:,1], c=color, cmap=plt.cm.Spectral)
    plt.show()
    
    
# demo with swiss roll dataset
def main():
    # import swiss roll dataset
    input_x, color = datasets.make_swiss_roll(n_samples=1500)
    # visualize original data
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(input_x[:, 0], input_x[:, 1], input_x[:, 2], c=color, cmap=plt.cm.Spectral)
    
    # compute the nearest neighbors
    print('comput neighbor matrix...')
    neibor, neibor_idx = distance_mat(input_x, n_neighbors=10)
    
    # compute lle
    print('compute LLE...')
    Y = LLE(input_x, neibor_idx, n_components=2)
    
    # visualize lle
    print('visualize...')
    visualize(Y, color)
    

if __name__ == '__main__':
    main()
    