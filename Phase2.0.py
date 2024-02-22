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
from imblearn.over_sampling import SMOTE

# reading data and printing shape
df = pd.read_csv(
    "megastore-classification-dataset.csv")
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
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

cols = ["Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment",
        "City", "State", "Region", "Product ID", "CategoryTree", "Product Name"]


# Import libraries


# Get the numerical columns in the dataframe
# numerical_cols = df.select_dtypes(include='number').columns
outliers_cols = ['Sales', 'Quantity', 'Discount']

# Create a boxplot for each numerical column
for col in outliers_cols:
    plt.boxplot(df[col])
    plt.title(col)
    plt.show()


# Loop over each column and calculate IQR and identify outliers

# Specify the columns for which to replace outliers

# numerical_cols = df.select_dtypes(include='number').columns
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

    # Print the results for the column
    print("Column:", col)
    print("Upper bound:", upper_bound)
    print("Lower bound:", lower_bound)
    print("Number of outliers:", cnt)

    # Create a boxplot of the column after outlier removal
    plt.boxplot(df[col])
    plt.title("Boxplot of {} after outlier removal".format(col))
    plt.show()


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

smote = SMOTE(random_state=42)

smote.fit_resample(x_train, y_train)

print("..........................................................................................")

# Set up the hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Create an instance of logistic regression
logreg = LogisticRegression()

# Create a grid search object
grid_search = GridSearchCV(logreg, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters for logisitc regression: ", grid_search.best_params_)
print("Best score for logistic regression: ", grid_search.best_score_)

# Print the scores for all hyperparameter combinations
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, params in zip(means, stds, params):
    print("Accuracy: %0.3f (+/- %0.3f) for %r" % (mean, std * 2, params))


# Train a logistic regression model with the default hyperparameters
default_model = LogisticRegression()
default_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(default_model, 'logistic_regression_model_default')

# Loading the model
lr_default_loaded = joblib.load('logistic_regression_model_default')

y_pred_default = lr_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Train a logistic regression model with the best hyperparameters
best_model = LogisticRegression(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(best_model, 'logistic_regression_model_best')

# Loading the model
lr_best_loaded = joblib.load('logistic_regression_model_best')

y_pred_best = lr_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of logistic regression with default hyperparameters: ",
      accuracy_default*100, "%")
print("Accuracy of logistic with best hyperparameters: ", accuracy_best*100, "%")


# Set up the hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Create an instance of KNN classifier
knn = KNeighborsClassifier()

# Create a grid search object
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)
print("-----------------------------------------------------------------------------------------------------------------")

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters for KNN: ", grid_search.best_params_)
print("Best score for KNN: ", grid_search.best_score_)

# Print the scores for all hyperparameter combinations
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, params in zip(means, stds, params):
    print("Accuracy: %0.3f (+/- %0.3f) for %r" % (mean, std * 2, params))


# Train a KNN model with the default hyperparameters
default_model = KNeighborsClassifier()
default_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(default_model, 'KNN_model_default')

# Loading the model
KNN_default_loaded = joblib.load('KNN_model_default')

y_pred_default = KNN_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Train a KNN model with the best hyperparameters
best_model = KNeighborsClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(best_model, 'KNN_model_best')

# Loading the model
KNN_best_loaded = joblib.load('KNN_model_best')

y_pred_best = KNN_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of KNN with default hyperparameters: ", accuracy_default*100, "%")
print("Accuracy of KNN with best hyperparameters: ", accuracy_best*100, "%")


# Set up the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# Create an instance of Random Forest classifier
rf = RandomForestClassifier()

# Create a grid search object
grid_search = GridSearchCV(rf, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)
print("-----------------------------------------------------------------------------------------------------------------")

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters for random forest: ", grid_search.best_params_)
print("Best score for random forest: ", grid_search.best_score_)

# Print the scores for all hyperparameter combinations
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, params in zip(means, stds, params):
    print("Accuracy: %0.3f (+/- %0.3f) for %r" % (mean, std * 2, params))


# Train a random forest model with the default hyperparameters
default_model = RandomForestClassifier()
default_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(default_model, 'Random_Forest_model_default')

# Loading the model
RF_default_loaded = joblib.load('Random_Forest_model_default')

y_pred_default = RF_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Train a random forest model with the best hyperparameters
best_model = RandomForestClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(best_model, 'Random_Forest_model_best')

# Loading the model
RF_best_loaded = joblib.load('Random_Forest_model_best')

y_pred_best = RF_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of random forest with default  hyperparameters: ",
      accuracy_default*100, "%")
print("Accuracy of random forest with best  hyperparameters: ",
      accuracy_best*100, "%")


# Set up the hyperparameter grid
param_grid = {
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Create an instance of Decision Tree classifier
dt = DecisionTreeClassifier()

# Create a grid search object
grid_search = GridSearchCV(dt, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)
print("-----------------------------------------------------------------------------------------------------------------")

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters for dt: ", grid_search.best_params_)
print("Best score for dt: ", grid_search.best_score_)

# Print the scores for all hyperparameter combinations
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, params in zip(means, stds, params):
    print("Accuracy: %0.3f (+/- %0.3f) for %r" % (mean, std * 2, params))


# Train a decision tree model with the default hyperparameters
default_model = DecisionTreeClassifier()
default_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(default_model, 'Decision_tree_model_default')

# Loading the model
DT_default_loaded = joblib.load('Decision_tree_model_default')

y_pred_default = DT_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Train a decision tree model with the best hyperparameters
best_model = DecisionTreeClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(best_model, 'Decision_tree_model_best')

# Loading the model
DT_best_loaded = joblib.load('Decision_tree_model_best')

y_pred_best = DT_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of decision tree with default hyperparameters: ",
      accuracy_default*100, "%")
print("Accuracy of decision tree with best hyperparameters: ", accuracy_best*100, "%")


# Set up the hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01, 0.001]
}

# Create an instance of SVM classifier
svm = SVC(kernel='rbf')

# Create a grid search object
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)
print("-----------------------------------------------------------------------------------------------------------------")

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters for svm: ", grid_search.best_params_)
print("Best score for svm: ", grid_search.best_score_)

# Print the scores for all hyperparameter combinations
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, params in zip(means, stds, params):
    print("Accuracy: %0.3f (+/- %0.3f) for %r" % (mean, std * 2, params))


# Train an SVM model with the default hyperparameters
default_model = SVC(kernel='rbf')
default_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(default_model, 'SVM_model_default')

# Loading the model
SVM_default_loaded = joblib.load('SVM_model_default')

y_pred_default = SVM_default_loaded.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Train an SVM model with the best hyperparameters
best_model = SVC(kernel='rbf', **grid_search.best_params_)
best_model.fit(X_train, y_train)

# Saving model into a file
joblib.dump(best_model, 'SVM_model_best')

# Loading the model
SVM_best_loaded = joblib.load('SVM_model_best')

y_pred_best = SVM_best_loaded.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the accuracies of the default and best models
print("Accuracy of svm with default hyperparameters: ", accuracy_default*100, "%")
print("Accuracy of svm with best hyperparameters: ", accuracy_best*100, "%")
