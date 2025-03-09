
import numpy as np
import scipy.special

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


# bias = 1.0  # leads to roughly 58~70% true labels
def get_sample_data_logistic_regression(rng, data_size, data_dim, rho = 0.1, bias = 0.0):
    assert(data_dim > 5)

    beta = np.zeros(data_dim)
    beta[0] = 3.0
    beta[1] = 1.5
    beta[4] = 2.0
    
    X = getX(rng, data_size, data_dim, rho)
    
    logits = X @ beta + bias
    true_probs = scipy.special.expit(logits)

    y = scipy.stats.bernoulli(true_probs).rvs(random_state = rng)

    gamma = np.where(beta != 0, 1, 0)
    return X, y, beta, bias, gamma


def getX(rng, data_size, data_dim, rho):

    covariance_matrix = np.zeros((data_dim, data_dim))
    for i in range(data_dim):
        for j in range(data_dim):
            covariance_matrix[i, j] = rho**(abs(i - j))

    X = rng.multivariate_normal(mean = np.zeros(data_dim), cov = covariance_matrix, size = data_size)
    return X