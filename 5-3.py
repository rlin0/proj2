from surprise import Reader, Dataset
from surprise import SVD, accuracy
from surprise.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
from basicvis import visualize

def main():
    # Initialize  dataset (from old code)
    Y_train = np.loadtxt('data/train.txt').astype(int)
    Y_test = np.loadtxt('data/test.txt').astype(int)

    M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
    N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
    print("Factorizing with M: ", M, " users, N: ", N, " movies.")

    # Load data with Surprise
    reader = Reader(line_format='user item rating', sep='\t')
    Y_train = Dataset.load_from_file('data/train.txt', reader=reader)
    Y_test = Dataset.load_from_file('data/test.txt', reader=reader)

    trainset = Y_train.build_full_trainset()
    testset = Y_test.build_full_trainset().build_testset()


    K = 20
    reg = 0.1
    lr = 0.01


    # PART 5-3: INTRODUCE MEAN AND REGULARIZED BIAS TERMS
    # (based off of Step 1c in the guide)
    # Create model and fit it
    algo = SVD(n_factors=K, lr_all=lr, reg_all=reg, n_epochs=30, biased=True)
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Evaluate error using err function from problem set
    E_out = get_err(predictions)
    print('E_out (MSE): ', E_out)

    # Try GridSearchCV
    '''
    param_grid = {'n_epochs': [10, 15, 20, 25, 30],
                 'lr_all':   [0.002, 0.005, 0.01, 0.02, 0.03],
                 'reg_all':  [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(Y_train)

    print('Grid Search:')
    print(0.5 * gs.best_score['rmse'] ** 2)
    print(gs.best_params['rmse'])
    '''
    # Results: best params were n_epochs=30, reg=0.1, lr=0.01

    # Apply SVD to V
    V = algo.qi.T
    U = algo.pu
    A, s, B = np.linalg.svd(V)
    # Use first 2 columns of A
    A2 = A[:, :2]
    print(U.shape, V.shape, A2.shape)
    U_projected = np.dot(A2.T, U.T)
    V_projected = np.dot(A2.T, V).T
    X = V_projected[:, 0]
    Y = V_projected[:, 1]

    visualize(X, Y, '5-3')

# Computes mean squared error to match that used in prob2utils.py
def get_err(predictions):
    err = 0.5 * (accuracy.rmse(predictions) ** 2)
    return err

if __name__ == "__main__":
    main()
