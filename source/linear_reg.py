import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#HOW TO ADD ALL THESE LIBRARIES?????????????    

disch_data = pd.read_csv('G_disch_20160728.csv', header = [10,11,12]) #COL 12 WHICH IS THE RESULTS IS ALSO A FUNCTION OF ITS PREVIOUS VALUES.
gage_data = pd.read_csv('G_gage_20160728.csv', header = [10,11,12]) 

disch_res = pd.read_csv('G_disch_20160728.csv', header = [12])

data = disch_data + gage_data #HOW TO COMBINE BOTH THE DATA SETS?????

# Split the data into training/testing sets
data_train = data[:-20]
data_test = data[-20:]

# Split the targets into training/testing sets
res_train = disch_res[:-20]
res_test = disch_res[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data_train, res_train)

# Make predictions using the testing set
res_pred = regr.predict(data_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(res_test, res_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(res_test, res_pred))


# Plot outputs
plt.scatter(data_test, res_test,  color='black')
plt.plot(data_test, res_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()