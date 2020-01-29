from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from Test_Train import TestTrainSplit
import joblib
import pickle
import numpy as np
import os
import sys



def NearestNeighborTrain(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
    classifier = KNeighborsClassifier(n_neighbors=hyperparameters["n_neighbors"],
                                      weights=hyperparameters["weights"],
                                      algorithm=hyperparameters["algorithm"],
                                      leaf_size=hyperparameters["leaf_size"],
                                      p=hyperparameters["p"],
                                      metric=hyperparameters["metric"],
                                      metric_params=hyperparameters["metric_params"],
                                      n_jobs=hyperparameters["n_jobs"])
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



def NearestNeighborValidate(user):
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

    classifier = KNeighborsClassifier(n_neighbors=hyperparameters["n_neighbors"],
                                      weights=hyperparameters["weights"],
                                      algorithm=hyperparameters["algorithm"],
                                      leaf_size=hyperparameters["leaf_size"],
                                      p=hyperparameters["p"],
                                      metric=hyperparameters["metric"],
                                      metric_params=hyperparameters["metric_params"],
                                      n_jobs=hyperparameters["n_jobs"])

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



def NearestNeighborRandomSearch(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    rand_n_neighbors = [int(x) for x in np.linspace(1, 10, num=10)]
    rand_weights = ["uniform", "distance"]
    rand_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    rand_leaf_size = [int(x) for x in np.linspace(20, 50, num=4)]
    rand_p = [int(x) for x in np.linspace(1, 5, num=5)]

    clf = KNeighborsClassifier()
    hyperparameters = dict(n_neighbors=rand_n_neighbors,
                           weights=rand_weights,
                           algorithm=rand_algorithm,
                           leaf_size=rand_leaf_size,
                           p=rand_p)

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
    NearestNeighborTrain(User)
elif method == "2":
    NearestNeighborValidate(User)
else:
    NearestNeighborRandomSearch(User)