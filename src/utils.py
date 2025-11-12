import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    J = (-1 / m) * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * (X.T @ (h - y))
        theta -= alpha * gradient
        cost = computeCost(X, y, theta)
        J_history.append(cost)
    return theta, J_history

def predict(X, theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

def plotData(X, y):
    pos = y == 1
    neg = y == 0
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], c='r', marker='+', label='Faulty')
    plt.scatter(X[neg, 0], X[neg, 1], c='g', marker='o', label='Functional')
    plt.legend()
    plt.grid(True)

def plotDecisionBoundary(X_orig, y, theta, mu, sigma):
    u = np.linspace(X_orig[:, 0].min() - 5, X_orig[:, 0].max() + 5, 100)
    v = np.linspace(X_orig[:, 1].min() - 5, X_orig[:, 1].max() + 5, 100)
    z = np.zeros((len(u), len(v)))
    
    for i in range(len(u)):
        for j in range(len(v)):
            u_norm = (u[i] - mu[0]) / sigma[0]
            v_norm = (v[j] - mu[1]) / sigma[1]
            features = np.array([1, u_norm, v_norm, u_norm * v_norm])
            z[i, j] = features @ theta
    
    u, v = np.meshgrid(u, v)
    plt.contour(u, v, z.T, levels=[0], colors='blue', linewidths=2)