import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# find solution(eigenvectors, eigenvalues)
def get_eigen(data):
    # data centering
    X = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    print('Shape of X: {}'.format(X.shape))
    # Covariance
    cov_X = ((X - np.mean(X, axis=0)).T.dot(X - np.mean(X, axis=0))) / (X.shape[0])
    print('Shape covariance matrix: {}'.format(cov_X.shape))
    eigvalues, eigvectors = np.linalg.eig(cov_X)

    return X, eigvalues, eigvectors

def express_var(eigvalues, eigvectors=None, n_eig=10):
    for i in range(n_eig):
        exp_var = np.sum(eigvalues[:i+1])*100 / np.sum(eigvalues)
        print(f'{i+1} Eigenvectors expresses {exp_var}% variance')
    
# find the base set of bases
def projection(X, eigvalues, eigvectors, n_components=2):
    eig = [(np.abs(eigvalues[i]), eigvectors[:, i]) for i in range(len(eigvalues))]
    eig.sort(key = lambda x: x[0], reverse=True)
    projection = np.hstack((eig[i][1].reshape(len(eigvalues),1)) for i in range(n_components))
    Y = X.dot(projection)
    
    return Y

def visualize(Y, y_value, label):
    Y_df = pd.DataFrame([Y[:,0] , Y[:,1] , y_value]).transpose()
    Y_df.columns = ['x1' , 'x2' , 'class']
    groups = Y_df.groupby('class')
    
    fig, ax = plt.subplots()
    for name, group in groups: 
        name = label[int(name)]
        ax.plot(group.x1, group.x2, marker = 'o', linestyle = '' , label = name)
    ax.legend(loc='upper right')
    #plt.figure(figsize=(30,30))
    plt.show()
    
# demo with iris dataset
def main():
    data = load_iris()
    input_x = data.data
    input_y = data.target
    input_name = data.target_names
    
    X, eigvals, eigvecs = get_eigen(input_x)
    print('-'*50)
    express_var(eigvals)
    Y = projection(X, eigvals, eigvecs)    #n_component=2
    visualize(Y, input_y, input_name)
    
if __name__ == '__main__':
    main()