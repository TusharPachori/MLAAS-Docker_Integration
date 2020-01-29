from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import numpy as np
from Test_Train import TestTrainSplit
import joblib
import os
import pickle
import sys


def LinearClassifiersLogisticRegressionTrain(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    classifier = LogisticRegression(penalty=hyperparameters["penalty"],
                                    dual=hyperparameters["dual"],
                                    tol=hyperparameters["tol"],
                                    C=hyperparameters["C"],
                                    fit_intercept=hyperparameters["fit_intercept"],
                                    intercept_scaling=hyperparameters["intercept_scaling"],
                                    class_weight=hyperparameters["class_weight"],
                                    random_state=hyperparameters["random_state"],
                                    solver=hyperparameters["solver"],
                                    max_iter=hyperparameters["max_iter"],
                                    multi_class=hyperparameters["multi_class"],
                                    verbose=hyperparameters["verbose"],
                                    warm_start=hyperparameters["warm_start"],
                                    n_jobs=hyperparameters["n_jobs"],
                                    l1_ratio=hyperparameters["l1_ratio"])

    classifier.fit(X_train, y_train)
    if not os.path.exists("/app/user_{}/trained_model".format(user)):
        os.makedirs("/app/user_{}/trained_model".format(user))
    download_link = "/app/user_{0}/trained_model/{1}".format(user, 'classifier.pkl')
    joblib.dump(classifier, download_link)
    y_pred = classifier.predict(X_test)
    result = accuracy_score(y_test, y_pred)
    download_link = "media/user_{0}/trained_model/{1}".format(user, 'classifier.pkl')

    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({'result': result, 'download_link': download_link}, f)


def LinearClassifiersLogisticRegressionValidate(user):
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
    classifier = LogisticRegression(penalty=hyperparameters["penalty"],
                                    dual=hyperparameters["dual"],
                                    tol=hyperparameters["tol"],
                                    C=hyperparameters["C"],
                                    fit_intercept=hyperparameters["fit_intercept"],
                                    intercept_scaling=hyperparameters["intercept_scaling"],
                                    class_weight=hyperparameters["class_weight"],
                                    random_state=hyperparameters["random_state"],
                                    solver=hyperparameters["solver"],
                                    max_iter=hyperparameters["max_iter"],
                                    multi_class=hyperparameters["multi_class"],
                                    verbose=hyperparameters["verbose"],
                                    warm_start=hyperparameters["warm_start"],
                                    n_jobs=hyperparameters["n_jobs"],
                                    l1_ratio=hyperparameters["l1_ratio"])
    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    rmse_score = np.sqrt(scores)
    rmse_score = np.round(rmse_score, 3)
    mean = np.round(scores.mean(), 3)
    std = np.round(scores.std(), 3)
    scores = np.round(scores, 3)
    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({'rmse_score': rmse_score, 'mean': mean, 'std': std, 'scores': scores}, f)


def LinearClassifiersLogisticRegressionRandomSearch(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    rand_penalty = ['l2']
    rand_C = [float(x) for x in np.linspace(1.0, 10.0, num=10)]
    rand_fit_intercept = [True, False]
    rand_intercept_scaling = [float(x) for x in np.linspace(1.0, 10.0, num=10)]
    rand_intercept_scaling.append(None)
    rand_solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
    rand_max_iter = [int(x) for x in np.linspace(100, 350, num=6)]
    rand_multi_class = ['ovr', 'auto']
    rand_warm_start = [True, False]
    clf = LogisticRegression()
    hyperparameters = dict(penalty=rand_penalty,
                           C=rand_C,
                           fit_intercept=rand_fit_intercept,
                           intercept_scaling=rand_intercept_scaling,
                           solver=rand_solver,
                           max_iter=rand_max_iter,
                           multi_class=rand_multi_class,
                           warm_start=rand_warm_start, )

    clf = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
                             n_jobs=1)
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
    LinearClassifiersLogisticRegressionTrain(User)
elif method == "2":
    LinearClassifiersLogisticRegressionValidate(User)
else:
    LinearClassifiersLogisticRegressionRandomSearch(User)