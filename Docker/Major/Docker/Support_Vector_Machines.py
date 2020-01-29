import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from Test_Train import TestTrainSplit
import joblib
import numpy as np
import pickle
import sys



def SupportVectorMachinesTrain(user):
    with open('/app/user_{}/Variable/hyperparameters.pkl'.format(user), 'rb') as f:
        hyperparameters = pickle.load(f)

    file_name = hyperparameters['filename']
    user = hyperparameters['user']
    my_file = "/app/user_{0}/processed_csv/{1}".format(user, file_name)
    features_list = hyperparameters["featureList"]
    label = hyperparameters['label']
    ratio = hyperparameters['ratio']

    X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

    classifier = SVC(C=hyperparameters['C'],
                     kernel=hyperparameters['kernel'],
                     degree=hyperparameters['degree'],
                     gamma=hyperparameters['gamma'],
                     coef0=hyperparameters['coef0'],
                     shrinking=hyperparameters['shrinking'],
                     probability=hyperparameters['probability'],
                     tol=hyperparameters['tol'],
                     cache_size=hyperparameters['cache_size'],
                     class_weight=hyperparameters['class_weight'],
                     verbose=hyperparameters['verbose'],
                     max_iter=hyperparameters['max_iter'],
                     decision_function_shape=hyperparameters['decision_function_shape'],
                     random_state=hyperparameters['random_state'])
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


def SupportVectorMachinesValidate(user):
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
    classifier = SVC(C=hyperparameters['C'],
                     kernel=hyperparameters['kernel'],
                     degree=hyperparameters['degree'],
                     gamma=hyperparameters['gamma'],
                     coef0=hyperparameters['coef0'],
                     shrinking=hyperparameters['shrinking'],
                     probability=hyperparameters['probability'],
                     tol=hyperparameters['tol'],
                     cache_size=hyperparameters['cache_size'],
                     class_weight=hyperparameters['class_weight'],
                     verbose=hyperparameters['verbose'],
                     max_iter=hyperparameters['max_iter'],
                     decision_function_shape=hyperparameters['decision_function_shape'],
                     random_state=hyperparameters['random_state'])

    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    rmse_score = np.round(np.sqrt(scores), 3)
    mean = np.round(scores.mean(), 3)
    std = np.round(scores.std(), 3)
    scores = np.round(scores, 3)
    if not os.path.exists("/app/user_{}/Variable".format(user)):
        os.makedirs("/app/user_{}/Variable".format(user))
    with open('/app/user_{}/Variable/result.pkl'.format(user), 'wb') as f:
        pickle.dump({'rmse_score': rmse_score, 'mean': mean, 'std': std, 'scores': scores}, f)



def SupportVectorMachinesRandomSearch(user):
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
    rand_probability = [True, False]
    rand_decision_function_shape = ['ovo', 'ovr']
    regressor = SVC()
    hyperparameters = dict(C=rand_C,
                           kernel=rand_kernel,
                           degree=rand_degree,
                           shrinking=rand_shrinking,
                           probability=rand_probability,
                           decision_function_shape=rand_decision_function_shape, )
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
    SupportVectorMachinesTrain(User)
elif method == "2":
    SupportVectorMachinesValidate(User)
else:
    SupportVectorMachinesRandomSearch(User)