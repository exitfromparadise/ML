"""
based on https://realpython.com/knn-python/
"""

"""
knn is a supervised machine learning algorithm very well suited for not too complex problems
and its easy to understand
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_attribute(df, attr, nbins):
    df[attr].hist(bins=nbins)
    plt.savefig("plot_%s.png" % attr)


url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/abalone/abalone.data"
)
abalone = pd.read_csv(url, header=None)

print(abalone.head())

abalone.columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]

abalone = abalone.drop("Sex", axis=1)
plot_attribute(abalone, "Rings", 15)

# get correlation matrix
correlation_matrx = abalone.corr()

print(correlation_matrx["Rings"])

X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)


from sklearn.neighbors import KNeighborsRegressor
n_neighbors = 3
knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)

knn_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

# train data eval
# rmse - average error of the predicted age
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print("rmse train: ", rmse)

# test data eval
test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print("rmse test: ", rmse)

# the larger the difference between rmse in train and test, its a clear sign of overfitting

import seaborn as sns
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
color = test_preds #y_test
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=color, s=50, cmap=cmap
)
f.colorbar(points)
plt.savefig("knn_fit.png")

# improve knn performance using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print("gridsearch.best_params_: ", gridsearch.best_params_)

# train again with optimal value
train_preds_grid = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print("test_rmse: ", test_rmse)



# adding weights for neighbors
parameters = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"],
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print("gridsearch.best_params_: ", gridsearch.best_params_)
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print("test_rmse weighted: ", test_rmse)

# use bagging to further improve
# bagging is an ensemble method
# fits a large number of ML models with slight variations in each fit -> less affected by fluctuations

best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]
bagged_knn = KNeighborsRegressor(
    n_neighbors=best_k, weights=best_weights
)
from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)

bagging_model.fit(X_train, y_train)
test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print("test_rmse bagged: ", test_rmse)
