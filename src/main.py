import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Load data
data = np.loadtxt('data/ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

# Normalize features
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

# Add polynomial features
X_poly = np.column_stack((
    np.ones(X.shape[0]),
    X_norm[:, 0],
    X_norm[:, 1], 
    X_norm[:, 0] * X_norm[:, 1]
))

# Train model
theta = np.zeros(X_poly.shape[1])
alpha = 0.01
num_iters = 2000

theta, J_history = gradientDescent(X_poly, y, theta, alpha, num_iters)

# Visualization
plotData(X, y)
plotDecisionBoundary(X, y, theta, mu, sigma)
plt.title("Decision Boundary and Training Data")
plt.xlabel("Vibration")
plt.ylabel("Rotation Unevenness")
plt.legend()
plt.grid(True)
plt.show()

# Prediction example
new_raw = np.array([65, 70])
new_norm = (new_raw - mu) / sigma
sample = np.array([1, new_norm[0], new_norm[1], new_norm[0] * new_norm[1]])

prob = sigmoid(sample @ theta)
classification = "FAULTY" if prob >= 0.5 else "FUNCTIONAL"

print(f"\nFailure probability: {prob:.4f}")
print(f"Conclusion: engine is {classification}")