from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import mstats
import numpy as np
import joblib
import pandas as pd
import preprocessor as preprocessor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

# reading data and printing shape
df = pd.read_csv(
    "megastore-tas-test-classification.csv")
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
X = df.drop(["ReturnCategory", "Ship Date", "Order Date", "Country"], axis=1)
Y = df["ReturnCategory"]

cols = ["Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment",
        "City", "State", "Region", "Product ID", "CategoryTree", "Product Name"]


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)


# Get the numerical columns in the dataframe
numerical_cols = df.select_dtypes(include='number').columns


# Specify the columns for which to replace outliers
outliers_cols = ['Sales', 'Quantity', 'Discount']

# Loop over each column and replace outliers with the median value
for col in outliers_cols:
    # Calculate IQR for the column
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the upper and lower bounds for identifying outliers
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Count the number of outliers in the column
    cnt = 0
    for i in range(len(df)):
        if df[col].iloc[i] > upper_bound:
            cnt += 1
        if df[col].iloc[i] < lower_bound:
            cnt += 1

    # Replace the outlier values with the median value
    median_val = df[col].median()
    df[col] = np.where(df[col] > upper_bound, median_val, df[col])
    df[col] = np.where(df[col] < lower_bound, median_val, df[col])


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

# feature selection using chi2
selector = SelectKBest(chi2, k=5)
selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

dataset = pd.DataFrame(X_train)
dataset['ReturnCategory'] = y_train

print(".................................................Logistic_regression................................................")


# Loading the model
lr_default_loaded = joblib.load('logistic_regression_model_default')

y_pred_default = lr_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# logistic regression model with the best hyperparameters

# Loading the model
lr_best_loaded = joblib.load('logistic_regression_model_best')

y_pred_best = lr_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of logistic regression with default hyperparameters: ",
      accuracy_default*100, "%")
print("Accuracy of logistic with best hyperparameters: ", accuracy_best*100, "%")


print("---------------------------------------------------------------KNN--------------------------------------------------")

# Loading the model
KNN_default_loaded = joblib.load('KNN_model_default')

y_pred_default = KNN_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# KNN model with the best hyperparameters

# Loading the model
KNN_best_loaded = joblib.load('KNN_model_best')

y_pred_best = KNN_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of KNN with default hyperparameters: ", accuracy_default*100, "%")
print("Accuracy of KNN with best hyperparameters: ", accuracy_best*100, "%")

print("-------------------------------------------------------Random Forest--------------------------------------------------------")


# Loading the model
RF_default_loaded = joblib.load('Random_Forest_model_default')

y_pred_default = RF_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# random forest model with the best hyperparameters

# Loading the model
RF_best_loaded = joblib.load('Random_Forest_model_best')

y_pred_best = RF_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of random forest with default  hyperparameters: ",
      accuracy_default*100, "%")
print("Accuracy of random forest with best  hyperparameters: ",
      accuracy_best*100, "%")


print("---------------------------------------------------------Decision Tree------------------------------------------------------")


# Loading the model
DT_default_loaded = joblib.load('Decision_tree_model_default')

y_pred_default = DT_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# decision tree model with the best hyperparameters

# Loading the model
DT_best_loaded = joblib.load('Decision_tree_model_best')

y_pred_best = DT_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of decision tree with default hyperparameters: ",
      accuracy_default*100, "%")
print("Accuracy of decision tree with best hyperparameters: ", accuracy_best*100, "%")

print("----------------------------------------------------------SVM-------------------------------------------------------")

# SVM model with the default hyperparameters

# Loading the model
SVM_default_loaded = joblib.load('SVM_model_default')

y_pred_default = SVM_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# SVM model with the best hyperparameters

# Loading the model
SVM_best_loaded = joblib.load('SVM_model_best')

y_pred_best = SVM_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of svm with default hyperparameters: ", accuracy_default*100, "%")
print("Accuracy of svm with best hyperparameters: ", accuracy_best*100, "%")
