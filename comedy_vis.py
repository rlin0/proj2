import numpy as np
import matplotlib.pyplot as plt
from basicvis import visualize
import pandas as pd
from biasedutils import train_model, get_err

def main():
    movies = pd.read_csv('data/movies.txt', delimiter="\t", header=None, encoding="latin_1", usecols = (0,1,7)).values
    data = np.loadtxt('data/data.txt')
    mov_ids = data[:, 1]
    ratings = data[:, 2]


    train = np.loadtxt('data/train.txt').astype(int)
    test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(train[:,0]), max(test[:,0])).astype(int) # users
    N = max(max(train[:,1]), max(test[:,1])).astype(int) # movies
    print("Factorizing with M: ", M, " users, N: ", N, " movies.")

    Y_train = []
    Y_test = []

    for x in train:
        id = x[1] - 1
        if movies[id][2] == 1: # check if comedy
            Y_train.append(x)

    for x in test:
        id = x[1] - 1
        if movies[id][2] == 1: # check if comedy
            Y_test.append(x)

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    K = 20
    reg = 0.1
    eta = 0.03 # learning rate
    E_in = []
    E_out = []

    # Use to compute Ein and Eout
    U, V, a, b, E_in = train_model(M, N, K, eta, reg, Y_train)
    E_out = get_err(U, V, a, b, Y_test)
    print('E_out (MSE): ', E_out)


    # Apply SVD to V
    A, s, B = np.linalg.svd(V)
    # Use first 2 columns of A
    A2 = A[:, :2]
    U_projected = np.dot(A2.T, U.T)
    V_projected = np.dot(A2.T, V).T
    X = V_projected[:, 0]
    Y = V_projected[:, 1]


    # Plot stuff
    comedy_ids = [x[0] for x in movies if x[2] == 1]
    plot_x = []
    plot_y = []
    plot_ratings = []
    plot_names = []
    for id in comedy_ids:
        if id > len(X):
            continue
        plot_x.append(X[id-1])
        plot_y.append(Y[id-1])
        plot_ratings.append(ratings[id-1])
        plot_names.append(movies[id-1][1])
    plt.scatter(plot_x, plot_y, marker='*', c=plot_ratings, cmap='viridis')
    #plt.text(plot_x+.005, plot_y+.005, plot_names)
    plt.title('Comedy Movies')
    plt.savefig('ComedyMovieSpectrum.png')
    plt.close()

if __name__ == "__main__":
    main()
