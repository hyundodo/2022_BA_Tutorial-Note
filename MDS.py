import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

       
def distance_matrix(input):
    # step1: construct distance matrix
    # generate matrix
    dis_matrix = np.zeros((len(input), len(input)))
    # compute Euclidean Distance
    for i in range(len(input)):
        for j in range(len(input)):
            dis_matrix[i, j] = np.sqrt(np.sum((input[i, :] - input[j, :])**2))
    
    return dis_matrix
    
def mds(distance_matrix, dim=2):
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


def visualize(Y, y_value, label):
    # express 2 dimension
    Y_df = pd.DataFrame([Y[0] , Y[1] , y_value]).transpose()
    Y_df.columns = ['x1' , 'x2' , 'class']
    groups = Y_df.groupby('class')
    
    fig, ax = plt.subplots()
    for name, group in groups: 
        name = label[int(name)]
        ax.plot(group.x1, group.x2, marker = 'o', linestyle = '' , label = name)
    ax.legend(loc='upper right')
    plt.show()

# demo
def main():
    data = load_iris()
    input_x = data.data
    input_y = data.target
    input_name = data.target_names
    
    print('Data Info') 
    print(f'Number of X: {len(input_x)}')
    print(f'Number of y: {len(input_y)}')
    
    print('compute distance Matrix')
    dis_mat = distance_matrix(input_x)
    print('compute MDS')
    X = mds(dis_mat)
    
    print('Visualize')
    visualize(X, input_y, input_name)
    
if __name__ == '__main__':
    main()