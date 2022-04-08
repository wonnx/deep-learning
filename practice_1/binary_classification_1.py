import numpy as np
from random import random

################################################################
m = 10000
n = 1000
w = np.array([0.5, 0.5])
b = 0.5
k = 5000
alpha = 0.01
################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_dataset(m, n):
    x1_train = list(); x2_train = list(); y_train = list()
    for i in range(m):
        x1_train.append(np.random.uniform(-10, 10))
        x2_train.append(np.random.uniform(-10, 10))
        if x1_train[-1] + x2_train[-1] > 0: y_train.append(1)
        else: y_train.append(0)

    x1_test = list(); x2_test = list(); y_test = list()
    for i in range(n):
        x1_test.append(np.random.uniform(-10, 10))
        x2_test.append(np.random.uniform(-10, 10))
        if x1_test[-1] + x2_test[-1] > 0: y_test.append(1)
        else: y_test.append(0)

    return x1_train, x2_train, y_train, x1_test, x2_test, y_test

def logistic_regression(x1_train, x2_train, y_train):
    global w,  b
    j = 0; dw1 = 0; dw2 = 0; db = 0
    for i in range(m):
        z = np.dot(w.T, np.array([x1_train[i], x2_train[i]])) + b
        a = sigmoid(z)
        if a < 1e-12: a = 1e-12
        elif a > 1 - 1e-12: a = 1 - 1e-12
        j += -(y_train[i] * np.log(a) + (1-y_train[i]) * np.log(1-a))
        dz = a - y_train[i]
        dw1 += x1_train[i] * dz
        dw2 += x2_train[i] * dz
        db += dz
    j = j/m; dw1 = dw1/m; dw2 = dw2/m; db = db/m
    w = w - alpha * np.array([dw1, dw2])
    b = b - alpha * db
    pass

def testing_accuracy_and_cost(x1_samples, x2_samples, y_samples, samples):
    count = 0; cost = 0
    for i in range (samples):
        z = np.dot(w.T, np.array([x1_samples[i], x2_samples[i]])) + b
        a = sigmoid(z)
        if a < 1e-12: a = 1e-12
        elif a > 1 - 1e-12: a = 1 - 1e-12
        if a > 0.5 and y_samples[i] == 1: count += 1
        elif a < 0.5 and y_samples[i] == 0: count += 1
        cost += -(y_samples[i] * np.log(a) + (1-y_samples[i]) * np.log(1-a))
    return count, cost

if __name__ == '__main__':
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = generate_dataset(m, n)
    for i in range(k):
        logistic_regression(x1_train, x2_train, y_train)
        if i % 500 == 0:
            print(w)
            print(b)
    train_set_accuracy, train_set_cost = testing_accuracy_and_cost(x1_train, x2_train, y_train, m)
    test_set_accuracy, test_set_cost = testing_accuracy_and_cost(x1_test, x2_test, y_test, n)
    print(train_set_accuracy / m * 100)
    print(train_set_cost)
    print(test_set_accuracy / n * 100)
    print(test_set_cost)