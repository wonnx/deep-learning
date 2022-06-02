import math
import numpy as np
import random

################################################################
m = 10000
n = 1000
# W1 = np.random.uniform(-0.5, 0.5, (2,1))
# W2 = np.random.uniform(-0.5, 0.5, (1,2))
# b1 = np.random.uniform(-0.5, 0.5, (2,1))
# b2 = np.random.uniform(-0.5, 0.5, (1,1))
W1 = np.array([[-1.0], [-1.0]])
W2 = np.array([[1., -1.]])
b1 = np.array([[90.], [270.]])
b2 = np.array([[0.]])
k = 50000
alpha = 0.001
#################################################################

def generate_dataset(size):
    x_train = list(); y_train = list()
    for i in range(size):
        degree_value = random.uniform(0, 360)
        cosine_value = math.cos(math.radians(degree_value))
        x_train.append(degree_value)
        if cosine_value > 0: y_train.append(1)
        else: y_train.append(0)
    x_train = np.array(x_train).reshape(1, len(x_train))
    y_train = np.array(y_train).reshape(1, len(y_train))
    return x_train, y_train

def logistic_regression(X, Y):
    global W1, W2, b1, b2
    Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    A1 = np.clip(A1, 1e-12, 1 - 1e-12)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    A2 = np.clip(A2, 1e-12, 1 - 1e-12)

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / m 
    db1 = np.sum(dZ1, axis = 1, keepdims = True)

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

def testing_accuracy_and_cost(size, X, Y):
    accuracy = 0
    Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    A1 = np.clip(A1, 1e-12, 1 - 1e-12)
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    A2 = np.clip(A2, 1e-12, 1 - 1e-12)
    for i in range (size):
        if A2[0,i] > 0.5 and Y[0,i] == 1: accuracy += 1
        elif A2[0,i] < 0.5 and Y[0,i] == 0: accuracy += 1
    accuracy = accuracy / size * 100
    cost = -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return accuracy, cost

if __name__ == '__main__':
    x_train, y_train = generate_dataset(m)
    x_test, y_test = generate_dataset(n)
    for i in range(1,k+1):
        logistic_regression(x_train, y_train)
        if i % 500 == 0: print(W1, b1, W2, b2)
    print(testing_accuracy_and_cost(m, x_train, y_train))
    print(testing_accuracy_and_cost(n, x_test, y_test))