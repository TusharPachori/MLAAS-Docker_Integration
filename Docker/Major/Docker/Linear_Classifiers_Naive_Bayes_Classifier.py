from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from Test_Train import TestTrainSplit
from sklearn.metrics import accuracy_score
import joblib
import pickle
import numpy as np
import os
import sys


def LinearClassifiersNaiveBayesClassifierTrain(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    classifier = GaussianNB(priors=hyperparameters["priors"], var_smoothing=hyperparameters["var_smoothing"])
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


def LinearClassifiersNaiveBayesClassifierValidate(user):
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

    classifier = GaussianNB(priors=hyperparameters["priors"], var_smoothing=hyperparameters["var_smoothing"])
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


argumentList = sys.argv
User = argumentList[1]
method = argumentList[2]
if method == "1":
    LinearClassifiersNaiveBayesClassifierTrain(User)
elif method == "2":
    LinearClassifiersNaiveBayesClassifierValidate(User)

