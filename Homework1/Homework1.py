# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:39:11 2021

@author: yeesun
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path = r"C:\Users\yeesu\Documents\Yeesun\Career path\3. Machine Learning\CS 229\machine-learning-ex\ex1"
os.chdir(path)


# %% 1. Warm-up Exersise
A = np.eye(5)


# %% 2. Linear Regression with One Variable
def plotData(x, y):
    plt.figure()
    plt.title("Scatter plot of training data")
    plt.scatter(x, y, c='red', marker="+")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")

data = pd.read_csv('ex1data1.txt', header=None)
x, y = np.array(data[0]), np.array(data[1])
plotData(x, y)


# %% 3. Gradient Descent
m = len(x)
X = np.array([np.ones((1, m))[0], x])
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01


def computeCost(X, y, theta):
    m = X.shape[1]
    h_x = np.matmul(theta.T, X)
    J_theta = 1/(2 * m) * np.dot((h_x - y), (h_x - y).T).sum()
    return J_theta

def gradientDecent(X, y, theta, alpha, iterations):
    count = 1
    n = theta.shape[0]
    m = X.shape[1]
    h_x = np.matmul(theta.T, X)
    J_theta = computeCost(X, y, theta)
    J_theta_list = [J_theta]
    
    while count < iterations:
        for i in range(n):
            theta[i] -= alpha * sum((h_x[0] - y) * X[i]) / m
        J_theta = computeCost(X, y, theta)
        J_theta_list.append(J_theta)
        count += 1
        h_x = np.matmul(theta.T, X)
    return theta, J_theta_list

# Run gradient descent
theta, J_theta_list = gradientDecent(X, y, theta, alpha, iterations)
print("Theta computed from gradient descent: %f, %f" %(theta[0][0], theta[1][0]))

# Plot the linear fit
plotData(x, y)
plt.plot(X[1], np.matmul(theta.T, X)[0])
plt.legend(['Linear regression', 'Training data'])

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)[0]
print("For population = 35,000, we predict a profit of %f" %(predict1 * 10000))

predict2 = np.dot(np.array([1, 7]), theta)[0]
print("For population = 70,000, we predict a profit of %f" %(predict2 * 10000))


# Visualizing J(theta_0, theta_1) with Surface and Contour plots
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = computeCost(X, y, t)


def surf(X, Y, Z):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('J_theta')
    
def contour(X, Y, Z):
    plt.contourf(theta0_vals, theta1_vals, J_vals)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')

    
surf(theta0_vals, theta1_vals, J_vals)
contour(theta0_vals, theta1_vals, J_vals)


# %% Optional Execise - 1. Gradient Descent for Multi-variable model

data1 = pd.read_csv('ex1data2.txt', header=None)
X, y = np.array(data1[[0, 1]]), np.array(data1[2])
m = len(y)


#  Feature Normalization
def featureNormalize(X, div):
    # div - 'range', 'std'
    mu_list = []
    scale_list = []
    X_norm = [np.ones(X.shape[0])]
    for i in range(X.shape[1]):
        feature = X[:, i]
        mu = feature.mean()
        if div == 'range':
            std = feature.max() - feature.min()
        else:
            std = feature.std()
        x_norm = (feature - mu) / std
        
        mu_list.append(mu)
        scale_list.append(std)
        X_norm.append(x_norm)
    return np.array(X_norm), mu_list, scale_list
        
X_norm, mu, scale =  featureNormalize(X, 'std')


# Run gradient descent
alpha = 0.1
num_iters = 400
theta = np.zeros((3, 1))
theta, _ = gradientDecent(X_norm, y, theta, alpha, num_iters)
print("Theta computed from gradient descent: \n%f\n%f\n%f" %(theta[0], theta[1], theta[2]))


# Predict values for 1650 sq-ft, 3 br house
price = np.dot(np.array([1, (1650 - mu[0]) / scale[0],
                         (3.5 - mu[1]) / scale[1]]), theta)[0]
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f" %price)


# %% Optional Execise - 2. Selecting Learning Rates

alpha_list = [1, 0.3, 0.1, 0.03, 0.01]
num_iters = 400

for alpha in alpha_list:
    theta = np.zeros((3, 1))
    theta, J_theta_list = gradientDecent(X_norm, y, theta, alpha, num_iters)
    plt.plot(J_theta_list)
plt.legend(alpha_list)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')


# %% Optional Execise - 3. Normal Equations

def normalEqn(X, y):
    theta = np.matmul(np.matmul(np.linalg.inv((np.matmul(X.T, X))), X.T), y)
    return theta

theta_est = normalEqn(X_norm.T, y)

print("Theta computed from normal equation: \n%f\n%f\n%f" %(theta_est[0], theta_est[1], theta_est[2]))


