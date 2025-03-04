
import numpy as np

# same data as in "Regression Shrinkage and Selection via the Lasso"
def lasso_linear(n = 20, d = 8, rho = 0.5, sigma=3.0):
    assert(d >= 8)
    prespecifiedBetaPart = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
    beta = np.zeros(d)
    beta[0:prespecifiedBetaPart.shape[0]] = prespecifiedBetaPart

    data_dim = beta.shape[0]
    assert(data_dim == d)
    covariance_matrix = np.zeros((data_dim, data_dim))
    for i in range(data_dim):
        for j in range(data_dim):
            covariance_matrix[i, j] = rho**(abs(i - j))

    X = np.random.multivariate_normal(np.zeros(data_dim), covariance_matrix, n)
    epsilon = np.random.randn(n)

    y = np.dot(X, beta) + sigma * epsilon
    gamma = np.where(beta != 0, 1, 0)
    return X, y, beta, gamma