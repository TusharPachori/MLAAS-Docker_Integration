import os
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.svm import SVR
from Test_Train import TestTrainSplit
import numpy as np
import joblib
import pickle
import sys


def SupportVectorRegressionTrain(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
    regressor = SVR(C=hyperparameters['C'],
                    kernel=hyperparameters['kernel'],
                    degree=hyperparameters['degree'],
                    gamma=hyperparameters['gamma'],
                    coef0=hyperparameters['coef0'],
                    shrinking=hyperparameters['shrinking'],
                    tol=hyperparameters['tol'],
                    epsilon=hyperparameters['epsilon'],
                    cache_size=hyperparameters['cache_size'],
                    verbose=hyperparameters['verbose'],
                    max_iter=hyperparameters['max_iter'])

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



def SupportVectorRegressionValidation(user):
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
    regressor = SVR(C=hyperparameters['C'],
                    kernel=hyperparameters['kernel'],
                    degree=hyperparameters['degree'],
                    gamma=hyperparameters['gamma'],
                    coef0=hyperparameters['coef0'],
                    shrinking=hyperparameters['shrinking'],
                    tol=hyperparameters['tol'],
                    epsilon=hyperparameters['epsilon'],
                    cache_size=hyperparameters['cache_size'],
                    verbose=hyperparameters['verbose'],
                    max_iter=hyperparameters['max_iter'])
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



def SupportVectorRegressionRandomSearch(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
    rand_C = [float(x) for x in np.linspace(1.0, 10.0, num=10)]
    rand_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    rand_degree = [int(x) for x in np.linspace(2.0, 5.0, num=3)]
    rand_shrinking = [True, False]
    regressor = SVR()
    hyperparameters = dict(C=rand_C,
                           kernel=rand_kernel,
                           degree=rand_degree,
                           shrinking=rand_shrinking)

    clf = RandomizedSearchCV(regressor, hyperparameters, random_state=1, n_iter=100,
                             cv=5, verbose=0, n_jobs=1)
    best_model = clf.fit(X, y)
    parameters = best_model.best_estimator_.get_params()
    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({'parameters': parameters}, f)


argumentList = sys.argv
User = argumentList[1]
method = argumentList[2]
if method == "1":
    SupportVectorRegressionTrain(User)
elif method == "2":
    SupportVectorRegressionValidation(User)
else:
    SupportVectorRegressionRandomSearch(User)