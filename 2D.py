# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
import matplotlib.pyplot as plt
from prob2utils import train_model, get_err
import pandas as pd
def main():
    movies = pd.read_csv('data/movies.txt', encoding='latin_1', header=None, delimiter='\t').values

    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with M: ", M, " users, N: ", N, " movies.")
    K = 20

    reg = 0.1
    eta = 0.03 # learning rate
    E_in = []
    E_out = []

    # Use to compute Ein and Eout
    U, V, E_in = train_model(M, N, K, eta, reg, Y_train)
    E_out = get_err(U, V, Y_test)
    print(E_out)

    # Apply SVD to V
    A, s, B = np.linalg.svd(V)
    # Use first 2 columns of A
    A2 = A[:, :2]
    U_projected = np.dot(A2.T, U.T)
    V_projected = np.dot(A2.T, V).T
    print(A.shape)
    print(U_projected.shape)
    print(V_projected.shape)
    X = V_projected[:, 0]
    Y = V_projected[:, 1]


    # 2 (a) any 10 movies
    for i in range(10):
        plt.scatter(X[i], Y[i], marker='x')
        plt.text(X[i]+.005, Y[i]+.005, movies[i][1])
    plt.show()

if __name__ == "__main__":
    main()
