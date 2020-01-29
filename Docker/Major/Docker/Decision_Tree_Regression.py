from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from Test_Train import TestTrainSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import os
import numpy as np
import math
import joblib
import pickle
import sys


def DecisionTreeRegressionTrain(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    regressor = DecisionTreeRegressor(criterion=hyperparameters['criterion'],
                                      splitter=hyperparameters['splitter'],
                                      max_depth=hyperparameters['max_depth'],
                                      min_samples_split=hyperparameters['min_samples_split'],
                                      min_samples_leaf=hyperparameters['min_samples_leaf'],
                                      min_weight_fraction_leaf=hyperparameters['min_weight_fraction_leaf'],
                                      max_features=hyperparameters['max_features'],
                                      random_state=hyperparameters['random_state'],
                                      max_leaf_nodes=hyperparameters['max_leaf_nodes'],
                                      min_impurity_decrease=hyperparameters['min_impurity_decrease'],
                                      min_impurity_split=hyperparameters['min_impurity_split'],
                                      presort=hyperparameters['presort'])

    regressor.fit(X_train, y_train)
    if not os.path.exists("/app/user_{}/trained_model".format(user)):
        os.makedirs("/app/user_{}/trained_model".format(user))
    download_link = "/app/user_{0}/trained_model/{1}".format(user, 'regressor.pkl')
    joblib.dump(regressor, download_link)
    y_pred = regressor.predict(X_test)
    result = mean_squared_error(y_test, y_pred)
    result = math.sqrt(result)
    result = round(result, 2)
    download_link = "media/user_{0}/trained_model/{1}".format(user, 'regressor.pkl')

    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({'result': result, 'download_link': download_link}, f)


def DecisionTreeRegressionValidate(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']
    cv = hyperparameters['cv']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    regressor = DecisionTreeRegressor(criterion=hyperparameters['criterion'],
                                      splitter=hyperparameters['splitter'],
                                      max_depth=hyperparameters['max_depth'],
                                      min_samples_split=hyperparameters['min_samples_split'],
                                      min_samples_leaf=hyperparameters['min_samples_leaf'],
                                      min_weight_fraction_leaf=hyperparameters['min_weight_fraction_leaf'],
                                      max_features=hyperparameters['max_features'],
                                      random_state=hyperparameters['random_state'],
                                      max_leaf_nodes=hyperparameters['max_leaf_nodes'],
                                      min_impurity_decrease=hyperparameters['min_impurity_decrease'],
                                      min_impurity_split=hyperparameters['min_impurity_split'],
                                      presort=hyperparameters['presort'])

    scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_score = np.sqrt(-scores)
    rmse_score = np.round(rmse_score, 3)
    mean = np.round(scores.mean(), 3)
    std = np.round(scores.std(), 3)
    scores = np.round(scores, 3)
    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({'rmse_score': rmse_score, 'mean': mean, 'std': std, 'scores': scores}, f)


def DecisionTreeRegressionRandomSearch(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
    rand_criterion = ["mse", 'mae']
    rand_splitter = ["best", "random"]
    rand_max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    rand_max_depth.append(None)
    rand_min_samples_split = [int(x) for x in np.linspace(2, 20, num=10)]
    rand_min_samples_leaf = [int(x) for x in np.linspace(2, 20, num=10)]
    rand_max_features = ["auto", "sqrt", "log2", None]
    rand_max_leaf_nodes = [int(x) for x in np.linspace(2, 20, num=10)]
    rand_max_leaf_nodes.append(None)
    rand_presort = [True, False]
    regressor = DecisionTreeRegressor()
    hyperparameters = dict(criterion=rand_criterion,
                           splitter=rand_splitter,
                           max_depth=rand_max_depth,
                           min_samples_split=rand_min_samples_split,
                           min_samples_leaf=rand_min_samples_leaf,
                           max_features=rand_max_features,
                           max_leaf_nodes=rand_max_leaf_nodes,
                           presort=rand_presort)

    clf = RandomizedSearchCV(regressor, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
                             n_jobs=1)
    best_model = clf.fit(X, y)
    parameters = best_model.best_estimator_.get_params()
    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({"parameters": parameters}, f)


argumentList = sys.argv
User = argumentList[1]
method = argumentList[2]
if method == "1":
    DecisionTreeRegressionTrain(User)
elif method == "2":
    DecisionTreeRegressionValidate(User)
else:
    DecisionTreeRegressionRandomSearch(User)