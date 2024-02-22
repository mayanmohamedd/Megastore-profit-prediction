from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

# reading data and printing shape
df = pd.read_csv("megastore-tas-test-regression.CSV")
print("shape = ", df.shape)

# drop duplicates and check null values
df.drop_duplicates()
print("shape after removing duplicates", df.shape)
print("null values are", "\n", df.isnull().sum())

df.dropna(how='any', inplace=True)
print("shape after removing null", df.shape)

# extract month and year from order date
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Order month"] = df["Order Date"].dt.month
df["Order year"] = df["Order Date"].dt.year

df["Ship Date"] = pd.to_datetime(df["Ship Date"])
df["ship month"] = df["Ship Date"].dt.month
df["ship year"] = df["Ship Date"].dt.year

print(df.head)

# split the data before preprocessing
X = df.drop(["Profit", "Ship Date", "Order Date", "Country"], axis=1)
Y = df["Profit"]
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

cols = ["Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment",
        "City", "State", "Region", "Product ID", "CategoryTree", "Product Name"]


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


X_train = Feature_Encoder(x_train, cols)
X_test = Feature_Encoder(x_test, cols)


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

dataset = pd.DataFrame(X_train)
dataset['Profit'] = y_train

# Handling outliers using IQR
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) |
                    (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]

# visualize correlation by heatmap
correlation = dataset.corr()
plt.figure(figsize=(22, 22))
g = sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()
print(correlation)

# selecting the most important features (5 features)
cor = abs(correlation["Profit"])
select_feat = cor[cor > 0.02]
print("........................................................")
print("features are", "\n", select_feat)
# x_test = x_test[x_test[10], x_test[11], x_test[17], x_test[19]]
print(x_test.columns)
# -----------------------------------------Multiple Linear Regression---------------------------------------------

# Dictionary that will hold all the MSE values of the three models to get the minimum of them
MSE = {}


# Loading the model
lreg_loaded = joblib.load('Multiple_Linear_Regression')

# Generate Prediction on test set
lreg_y_pred = lreg_loaded.predict(X_test)

# calculating Mean Squared Error (mse)
meanSquaredError = np.mean((lreg_y_pred - y_test)**2)
MSE['Multiple Regression'] = meanSquaredError

# Putting together the coefficient and their corresponding variable names
lreg_coefficient = pd.DataFrame()
lreg_coefficient["Columns"] = pd.DataFrame(X_train).columns
lreg_coefficient['Coefficient Estimate'] = pd.Series(lreg_loaded.coef_)
print(lreg_coefficient)

# plotting the coefficient score
fig, ax = plt.subplots(figsize=(20, 20))

color = ['tab:gray', 'tab:blue', 'tab:orange',
         'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
         'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
         'tab:orange', 'tab:green', 'tab:red', 'tab:blue',
         'tab:olive', 'tab:cyan', 'tab:pink', 'tab:brown',
         'tab:gray', 'tab:blue', 'tab:orange', 'tab:green']
ax.bar(lreg_coefficient["Columns"],
       lreg_coefficient['Coefficient Estimate'], color=color)
plt.xticks(rotation=90)
plt.show()

# create a 2x3 subplot grid
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# iterate over the columns of X and create a separate plot for each feature
for i in range(5):
    # get the current subplot axis
    row = i // 3
    col = i % 3
    ax = axs[row, col]

    # plot the test data and the regression line
    ax.scatter(X_test[:, i], y_test)
    ax.plot(X_test[:, i], lreg_loaded.predict(X_test), color='red')

    # add labels and title to the plot
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel('Y')
    ax.set_title(f'Linear Regression Line for Feature {i}')

# adjust the spacing between the subplots
plt.tight_layout()

# display the plot
plt.show()

# calculate the R-squared score of the predictions
r2_lasso_multiple_linear_regrission = r2_score(y_test, lreg_y_pred)

# print the R-squared score
print("R-squared score for multiple linear regression: %.2f" %
      r2_lasso_multiple_linear_regrission)
# ------------------------------------------------------------------------------------------------------------------

# --------------------------------------------Polynomial Regression------------------------------------------------

# # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)


# Loading the model
Poly_loaded = joblib.load('Polynomial_Regression')

# predicting on training data-set
y_train_predicted = Poly_loaded.predict(X_train_poly)
ypred = Poly_loaded.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = Poly_loaded.predict(poly_features.fit_transform(X_test))

mse_polynomial = metrics.mean_squared_error(y_test, prediction)

MSE["Polynomial Regression"] = mse_polynomial


# sort the test data by the feature that you want to plot
sort_idx = X_test[:, 0].argsort()
X_test_sort = X_test[sort_idx]
ypred_sort = ypred[sort_idx]

# create a 2x3 subplot grid
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# iterate over the columns of X and create a separate plot for each feature
for i in range(5):
    # sort the training data by the current feature
    sort_idx = X_test[:, i].argsort()
    X_test_sort = X_test[sort_idx]
    ypred_sort = ypred[sort_idx]

    # get the current subplot axis
    row = i // 3
    col = i % 3
    ax = axs[row, col]

    # plot the test data and the regression line
    ax.scatter(X_test[:, i], y_test)
    ax.plot(X_test_sort[:, i], ypred_sort, color='red')

    # add labels and title to the plot
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel('Y')
    ax.set_title(f'Polynomial Regression Line for Feature {i}')


# adjust the spacing between the subplots
plt.tight_layout()

# display the plot
plt.show()

# calculate the R-squared score of the predictions
r2_lasso_multiple_polynomial_regrission = r2_score(y_test, ypred)

# print the R-squared score
print("R-squared score for polynomial linear regression: %.2f" %
      r2_lasso_multiple_polynomial_regrission)

# ------------------------------------------------------------------------------------------------------------------

# --------------------------------------------Lasso Regression------------------------------------------------

# Loading the model
lasso_loaded = joblib.load('Lasso_Regression')

# predict the test data
y_pred = lasso_loaded.predict(X_test)


# create a 2x3 subplot grid
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# flatten the axs array for easy indexing
axs = axs.flatten()

# assuming X_test has 5 features
for i in range(5):
    # plot the scatter plot of the test data for the ith feature
    axs[i].scatter(X_test[:, i], y_test, color='blue')

    # fit the Lasso Regression model on the ith feature
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train[:, i].reshape(-1, 1), y_train)

    # predict the values for the test data using the fitted model
    y_pred = lasso.predict(X_test[:, i].reshape(-1, 1))

    # plot the regression line for the ith feature
    axs[i].plot(X_test[:, i], y_pred, color='red')

    # set the plot title and labels for the ith feature
    axs[i].set_title(
        "Lasso Regression with alpha=0.1 for feature {}".format(i+1))
    axs[i].set_xlabel("X{}".format(i+1))
    axs[i].set_ylabel("y")

# remove the unused subplot
for i in range(5, 6):
    fig.delaxes(axs[i])

# adjust the spacing between the subplots
fig.tight_layout()

# show the plot
plt.show()


# calculate the mean squared error of the predictions
mse_lasso = mean_squared_error(y_test, y_pred)
MSE["Lasso Regression"] = mse_lasso

# calculate the R-squared score of the predictions
r2_lasso_lasso_regression = r2_score(y_test, y_pred)

# print the R-squared score
print("R-squared score for Lasso regression: %.2f" % r2_lasso_lasso_regression)


# ------------------------------------------------------------------------------------------------------------------

# --------------------------------------------Elastic Regression------------------------------------------------

# Loading the model
e_net_loaded = joblib.load('Elastic_Regression')

# calculate the prediction and mean square error
y_pred_elastic = e_net_loaded.predict(x_test)
mse_Elastic = np.mean((y_pred_elastic - y_test)**2)
MSE["Elastic Regression"] = mse_Elastic

# create a 2x3 subplot grid
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# iterate over the columns of X and create a separate plot for each feature
for i in range(5):
    # get the current subplot axis
    row = i // 3
    col = i % 3
    ax = axs[row, col]

    # fit the Elastic Net model to the current feature
    e_net = ElasticNet(alpha=0.1)
    e_net.fit(x_train.values[:, i:i+1], y_train)

    # calculate the prediction
    y_pred = e_net.predict(x_test.values[:, i:i+1])

    # plot the test data and the regression line
    ax.scatter(x_test.values[:, i], y_test)
    ax.plot(x_test.values[:, i], y_pred, color='red')

    # add labels and title to the plot
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel('Y')
    ax.set_title(f'Elastic Regression for Feature {i}')

# adjust the spacing between the subplots
plt.tight_layout()

# display the plot
plt.show()
# e_net_coeff = pd.DataFrame()
# e_net_coeff["Columns"] = pd.DataFrame(x_train).columns
# e_net_coeff['Coefficient Estimate'] = pd.Series(e_net.coef_)
# e_net_coeff

# calculate the R-squared score of the predictions
r2_lasso_Elastic_regrission = r2_score(y_test, y_pred_elastic)

# print the R-squared score
print("R-squared score for Elastic regression: %.2f" %
      r2_lasso_Elastic_regrission)


# ------------------------------------------------------------------------------------------------------------------

# --------------------------------------------Ridge Regression------------------------------------------------


# Loading the model
ridgeR_loaded = joblib.load('Ridge_Regression')

y_pred_ridge = ridgeR_loaded.predict(x_test)

# calculate mean square error
mean_squared_error_ridge = np.mean((y_pred_ridge - y_test)**2)

MSE["Ridge Regression"] = mean_squared_error_ridge

# create a 2x3 subplot grid
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# iterate over the columns of X and create a separate plot for each feature
for i in range(5):
    # get the current subplot axis
    row = i // 3
    col = i % 3
    ax = axs[row, col]

    # fit the Elastic Net model to the current feature
    ridge = Ridge(alpha=1)
    ridge.fit(x_train.values[:, i:i+1], y_train)

    # calculate the prediction
    y_pred = ridge.predict(x_test.values[:, i:i+1])

    # plot the test data and the regression line
    ax.scatter(x_test.values[:, i], y_test)
    ax.plot(x_test.values[:, i], y_pred, color='red')

    # add labels and title to the plot
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel('Y')
    ax.set_title(f'Ridge Regression for Feature {i}')

# adjust the spacing between the subplots
plt.tight_layout()

# display the plot
plt.show()

# calculate the R-squared score of the predictions
r2_lasso_ridge_regrission = r2_score(y_test, y_pred_ridge)

# print the R-squared score
print("R-squared score for ridge regression: %.2f" % r2_lasso_ridge_regrission)


for key, value in MSE.items():
    print('MSE of ' + key + ' equal '+str(value))

# Getting the least MSE of all models

Least_MSE = min(MSE.values())

for key, value in MSE.items():
    if value == Least_MSE:
        print('Least MSE is of ' + key + ' and equal '+str(value))
