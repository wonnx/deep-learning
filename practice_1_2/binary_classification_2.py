import math
import numpy as np
import random

################################################################
m = 10000
n = 1000
w = np.array([[0.2]]) # shape(n, 1)
b = 0
k = 300000
alpha = 0.001
################################################################

def generate_dataset(size):
    x_train = list(); y_train = list()
    for i in range(size):
        degree_value = random.uniform(0, 360)
        sine_value = math.sin(math.radians(degree_value))
        x_train.append(degree_value)
        if sine_value > 0: y_train.append(1)
        else: y_train.append(0)
    x_train = np.array(x_train).reshape(1, len(x_train))
    y_train = np.array(y_train).reshape(1, len(y_train))
    return x_train, y_train

def logistic_regression(X, Y):
    global w, b
    Z = np.dot(w.T, X) + b
    A = 1 / (1 + np.exp(-Z))
    A = np.clip(A, 1e-12, 1 - 1e-12)
    dZ = A - Y
    db = np.sum(dZ) / m
    dw = np.dot(X, dZ.T) / m
    w = w - alpha * dw
    b = b - alpha * db

def testing_accuracy_and_cost(size, X, Y):
    accuracy = 0
    Z = np.dot(w.T, X) + b
    A = 1 / (1 + np.exp(-Z))
    A = np.clip(A, 1e-12, 1 - 1e-12)
    for i in range (size):
        if A[0,i] > 0.5 and Y[0,i] == 1: accuracy += 1
        elif A[0,i] < 0.5 and Y[0,i] == 0: accuracy += 1
    accuracy = accuracy / size * 100
    cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return accuracy, cost

if __name__ == '__main__':
    x_train, y_train = generate_dataset(m)
    x_test, y_test = generate_dataset(n)
    for i in range(1,k+1): 
        logistic_regression(x_train, y_train)
        if i % 500 == 0: print(w, b)
    print(testing_accuracy_and_cost(m, x_train, y_train))
    print(testing_accuracy_and_cost(n, x_test, y_test))