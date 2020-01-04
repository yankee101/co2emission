import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import pandas as pd
df = pd.read_csv("FuelConsumption.csv")
df.head()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
from sklearn import linear_model
rgr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
rgr.fit(x_train,y_train)
print("the coefficient: ",rgr.coef_)
print("the intercept: ",rgr.intercept_)
x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_test_hat = rgr.predict(x_test)
print ("Residual sum of errors: %.2f" % np.mean((y_test_hat - y_test) ** 2))
print('Variance score: %.2f' % rgr.score(x_test, y_test))
from sklearn.metrics import r2_score
print("r2 score: %.2f" % r2_score(y_test_hat,y_test))
plt.scatter(y_test_hat,y_test, color='blue')

