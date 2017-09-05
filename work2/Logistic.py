import random
import numpy as np
import pandas as pd

''' Function for splitting the data to test and training sets '''
def test_train_split(data, test_size):
    length = len(data)
    tst_size = int(length*test_size)
    trn_size = length - tst_size
    tst = random.sample(range(0,length), tst_size)
    trn = list(set([i for i in range(0,length)])-set(tst))
    test = data.loc[tst]
    train = data.loc[trn]
    return test,train

''' Sigmoid function '''
def sigmoid(val):
    return 1/(1+np.exp(-val))

''' Function to find the log likelihood '''
def log_Likelihood(features, target, weights):
    scores = np.dot(features, weights)
    log_l = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return log_l

'''Logistic function 
considering number of iterations as 10000
            alpha or learning rate as 0.3
Using gradient descent method '''

def logistic_regression(features, target,add_intercept = False):
    num_steps = 10000
    learning_rate = 0.3
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        weights = np.zeros(features.shape[1])
        
        for step in range(num_steps):
            scores = np.dot(features, weights)
            predictions = sigmoid(scores)

            error = target - predictions
            gradient = np.dot(features.T, error)
            weights += learning_rate * gradient
    return weights
    
#......................main code starts from here .......................
file = open('data/admission.txt','r')
data = pd.read_table(file,',')

train, test = test_train_split(data, test_size = 0.25)
train_len = train.shape[0]
test_len = test.shape[0]

y_train = train.admit.values
x_train = train[['gre','gpa','rank']]
x_train = x_train.as_matrix()
y_train = y_train.reshape(train_len,1)

y_test = test.admit.values
x_test = test[['gre','gpa','rank']]
x_test = x_test.as_matrix()
y_test = y_test.reshape(test_len,1)

logistic_regression(x_train,y_train,add_intercept=True)
