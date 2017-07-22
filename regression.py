import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

#reading the file using pandas
file = open('data/brain-body-mass.txt','r')
data = pd.read_table(file,'\s+')

#splitting data :traing and testing data in 80:20 ratio
train, test = train_test_split(data, test_size = 0.8)
train_len = train.shape[0]
test_len = test.shape[0]

#generating train_set values
x_train = train.Brain.values
y_train = train.Body.values
x_train = x_train.reshape(train_len,1)
y_train = y_train.reshape(train_len,1)

#generating test_set values
x_test = test.Brain.values
y_test = test.Body.values
x_test = x_test.reshape(test_len,1)
y_test = y_test.reshape(test_len,1)

#fitting Linear Regression
regression = linear_model.LinearRegression(normalize=1)
regression.fit(x_train,y_train)

print('Coefficient of Regression :', regression.coef_)
print("Mean squared error: %.2f" % np.mean((regression.predict(x_test) - y_test) ** 2))
print('Variance score: %.2f' % regression.score(x_test, y_test))

#plotting the Graph
plt.title('Regressed Values vs Real Values')
plt.xlabel('Brain Mass')
plt.ylabel('Body Mass')
plt.scatter(x_test,y_test,color='black')
plt.scatter(x_test,regression.predict(x_test),color='red')
plt.plot(x_test, regression.predict(x_test), color='green', linewidth=3)
plt.show()
