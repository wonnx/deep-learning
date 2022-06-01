import math
import numpy as np
import random

################################################################
m = 10000
n = 1000
w1 = np.array([[-0.5]]) # shape(n, 1)
w2 = np.array([[0.5]]) # shape(n, 1)
b1 = np.array([[1.]])
b2 = np.array([[0.5]])
k = 40000
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
    global w1, w2, b1, b2
    Z1 = np.dot(w1.T, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    A1 = np.clip(A1, 1e-12, 1 - 1e-12)
    Z2 = np.dot(w2.T, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    A2 = np.clip(A2, 1e-12, 1 - 1e-12)

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(w2.T, dZ2) * (A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / m 
    db1 = np.sum(dZ1, axis = 1, keepdims = True)

    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1

def testing_accuracy_and_cost(size, X, Y):
    accuracy = 0
    Z1 = np.dot(w1.T, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    A1 = np.clip(A1, 1e-12, 1 - 1e-12)
    Z2 = np.dot(w2.T, A1) + b2
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
        if i % 500 == 0: print(w1, b1, w2, b2)
    print(testing_accuracy_and_cost(m, x_train, y_train))
    print(testing_accuracy_and_cost(n, x_test, y_test))