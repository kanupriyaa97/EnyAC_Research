import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


#Assuming that dependant variable (disch row 12) is not dependant on its previous values.
disch_data = pd.read_csv('G_disch_20160728.csv')[8:11] 
gage_data = pd.read_csv('G_gage_20160728.csv')[8:11]

gage_res = pd.read_csv('G_gage_20160728.csv')[10:11]

data = pd.concat([disch_data, gage_data]) 

# Split the data into training/testing sets
data_train = clean_dataset(data.ix[:,:-100].transpose())
data_test = clean_dataset(data.ix[:,-100:-4].transpose())

#print(data_train.ix[:5,:])
#print(data_test)

# Split the targets into training/testing sets
res_train = clean_dataset(gage_res.ix[:, 3:-97].transpose())
res_test = clean_dataset(gage_res.ix[:, -97:].transpose())

#print(res_train)
#print(res_test)

#print(data.ix[:,0:5])
#print(type(data.ix[:,0:5]))
#print(data_train)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(data_train, res_train)
LinearRegression(copy_X = True, fit_intercept = True, normalize = False)
# Make predictions using the testing set
res_pred = regr.predict(data_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(res_test, res_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(res_test, res_pred))

print(data_test.values[:,0:1].shape)
print(res_test.shape)

# Plot outputs
plt.scatter(data_test.values[:,0:1], res_test,  color = "m", marker = "o", 
    s = 2)
plt.show()

plt.scatter(data_test.values[:,1:2], res_test,  color = "m", marker = "o", 
    s = 2)
plt.show()

plt.scatter(data_test.values[:,2:3], res_test,  color = "m", marker = "o", 
    s = 2)
plt.show()

plt.scatter(data_test.values[:,3:4], res_test,  color = "m", marker = "o", 
    s = 2)
plt.show()

plt.scatter(data_test.values[:,4:5], res_test,  color = "m", marker = "o", 
    s = 2)
plt.show()

plt.scatter(data_test.values[:,5:6], res_test,  color = "m", marker = "o", 
    s = 2)
plt.show()

plt.plot(data_test, res_test, color='blue', linewidth=0.5)
plt.plot(data_test, res_pred, color='red', linewidth=0.5)
plt.plot(res_pred, res_test, color='yellow', linewidth=0.5)

#plt.xticks(())
#plt.yticks(())

plt.show()