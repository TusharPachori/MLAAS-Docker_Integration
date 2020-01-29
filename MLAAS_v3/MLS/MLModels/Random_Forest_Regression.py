from django.shortcuts import render
import pickle
import os
import docker


client = docker.from_env()


def Random_Forest_Regression(request):
    if request.method == 'POST':
        try:
            if request.POST['submit'] != "RandomSearch":
                file_name = request.POST['filename']
                user = request.user
                user_name = str(user)
                features = request.POST.getlist('features')
                features_list = []
                for feature in features:
                    feature = feature[1:-1]
                    feature = feature.strip().split(", ")
                    for i in feature:
                        features_list.append(i[1:-1])
                label = request.POST['label']
                ratio = request.POST['ratio']
                cv = int(request.POST['cv'])

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "VALIDATE":
                    n_estimators = int(request.POST['n_estimators'])
                    criterion = request.POST['criterion']
                    max_depth = None if request.POST['max_depth'] == 'None' else request.POST['max_depth']
                    if max_depth is not None:
                        max_depth = int(request.POST['max_depth_values'])
                    min_samples_split = int(request.POST['min_samples_split'])
                    min_samples_leaf = int(request.POST['min_samples_leaf'])
                    min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
                    max_features = request.POST['max_features']
                    if max_features == "Int":
                        max_features = int(request.POST['max_features_integer'])
                    max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST['max_leaf_nodes']
                    if max_leaf_nodes is not None:
                        max_leaf_nodes = int(request.POST['max_leaf_nodes_value'])
                    min_impurity_decrease = float(request.POST['min_impurity_decrease'])
                    min_impurity_split = float(request.POST['min_impurity_split'])
                    bootstrap = True if request.POST['bootstrap'] == "True" else False
                    oob_score = True if request.POST['oob_score'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == 'None' else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(request.POST['n_jobs_value'])
                    random_state = None if request.POST['n_jobs'] == 'None' else request.POST['random_state']
                    if random_state is not None:
                        random_state = int(request.POST['random_state_value'])
                    verbose = int(request.POST['verbose'])
                    warm_start = True if request.POST['warm_start'] == "True" else False
                else:
                    n_estimators = int(request.POST['n_estimators'])
                    criterion = request.POST['criterion']
                    max_depth = None if request.POST['max_depth'] == 'None' else request.POST['max_depth']
                    if max_depth is not None:
                        max_depth = int(max_depth)
                    min_samples_split = int(request.POST['min_samples_split'])
                    min_samples_leaf = int(request.POST['min_samples_leaf'])
                    min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
                    max_features = request.POST['max_features']
                    if max_features != "auto" and max_features != "sqrt" and max_features != "log2" and max_features is not None:
                        max_features = int(max_features)
                    max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST[
                        'max_leaf_nodes']
                    if max_leaf_nodes is not None:
                        max_leaf_nodes = int(max_leaf_nodes)
                    min_impurity_decrease = float(request.POST['min_impurity_decrease'])
                    min_impurity_split = None if request.POST['min_impurity_split'] == 'None' else request.POST[
                        'min_impurity_split']
                    if min_impurity_split is not None:
                        min_impurity_split = float(min_impurity_split)
                    bootstrap = True if request.POST['bootstrap'] == "True" else False
                    oob_score = True if request.POST['oob_score'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == 'None' else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(n_jobs)
                    random_state = None if request.POST['n_jobs'] == 'None' else request.POST['random_state']
                    if random_state is not None:
                        random_state = int(random_state)
                    verbose = int(request.POST['verbose'])
                    warm_start = True if request.POST['warm_start'] == "True" else False

                hyperparameters = {"filename": file_name,"user": user_name,"label": label,"ratio": ratio,"cv": cv,
                                   "featureList": features_list,"n_estimators": n_estimators,"criterion": criterion,
                                   "max_depth": max_depth, "min_samples_split": min_samples_split,
                                   "min_samples_leaf": min_samples_leaf,
                                   "min_weight_fraction_leaf": min_weight_fraction_leaf, "max_features": max_features,
                                   "max_leaf_nodes": max_leaf_nodes, "min_impurity_decrease": min_impurity_decrease,
                                   "min_impurity_split": min_impurity_split, "bootstrap": bootstrap,
                                   "oob_score": oob_score, "n_jobs": n_jobs, "random_state": random_state,
                                   "verbose": verbose, "warm_start": warm_start}

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
                    if not os.path.exists("media/user_{}/Variable".format(user)):
                        os.makedirs("media/user_{}/Variable".format(user))
                    with open('media/user_{}/Variable/hyperparameters.pkl'.format(user), 'wb') as f:
                        pickle.dump(hyperparameters, f)
                    source_dir = "/Users/tusharpachori/PycharmProjects/Major1/project_v3/MLAAS_v3/media/user_{}".format(
                        user)
                    dest_dir = "/app/user_{}/".format(user)
                    container = client.containers.create("mlaas",
                                                         volumes={source_dir: {'bind': dest_dir, 'mode': 'rw'}},
                                                         tty=True, stdin_open=True, auto_remove=False)
                    container.start()
                    log = container.exec_run("python3 Major/Docker/Random_Forest_Regression.py {} 1".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    result, download_link = [results['result'], results['download_link']]
                    return render(request, 'MLS/result.html', {"model": "Random_Forest_Regression",
                                                               "metrics": "ROOT MEAN SQUARE ROOT",
                                                               "result": result,
                                                               "link": download_link})

                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
                    if not os.path.exists("media/user_{}/Variable".format(user)):
                        os.makedirs("media/user_{}/Variable".format(user))
                    with open('media/user_{}/Variable/hyperparameters.pkl'.format(user), 'wb') as f:
                        pickle.dump(hyperparameters, f)
                    source_dir = "/Users/tusharpachori/PycharmProjects/Major1/project_v3/MLAAS_v3/media/user_{}".format(
                        user)
                    dest_dir = "/app/user_{}/".format(user)
                    container = client.containers.create("mlaas",
                                                         volumes={source_dir: {'bind': dest_dir, 'mode': 'rw'}},
                                                         tty=True, stdin_open=True, auto_remove=False)
                    container.start()
                    log = container.exec_run("python3 Major/Docker/Random_Forest_Regression.py {} 2".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    rmse_score, mean, std, scores = [results["rmse_score"], results["mean"], results["std"],
                                                     results["scores"]]

                    return render(request, 'MLS/validate.html', {"model": "Random_Forest_Regression",
                                                                 "scoring": "neg_mean_squared_error",
                                                                 "scores": scores,'mean': mean,
                                                                 'std': std,'rmse': rmse_score,'cv': range(cv),
                                                                 'cv_list': range(1, cv+1)})

            elif request.POST['submit'] == "RandomSearch":
                file_name = request.POST['filename']
                features = request.POST.getlist('features')
                user = request.user
                user_name = str(user)
                features_list = []
                for feature in features:
                    feature = feature[1:-1]
                    feature = feature.strip().split(", ")
                    for i in feature:
                        features_list.append(i[1:-1])
                label = request.POST['label']
                ratio = request.POST['ratio']

                hyperparameters = {"filename": file_name, "user": user_name, "label": label, "ratio": ratio,
                                   "featureList": features_list, }

                if not os.path.exists("media/user_{}/Variable".format(user)):
                    os.makedirs("media/user_{}/Variable".format(user))
                with open('media/user_{}/Variable/hyperparameters.pkl'.format(user), 'wb') as f:
                    pickle.dump(hyperparameters, f)
                source_dir = "/Users/tusharpachori/PycharmProjects/Major1/project_v3/MLAAS_v3/media/user_{}".format(
                    user)
                dest_dir = "/app/user_{}/".format(user)
                container = client.containers.create("mlaas",
                                                     volumes={source_dir: {'bind': dest_dir, 'mode': 'rw'}},
                                                     tty=True, stdin_open=True, auto_remove=False)
                container.start()
                log = container.exec_run("python3 Major/Docker/Random_Forest_Regression.py {} 3".format(user),
                                         stderr=True, stdout=True, stream=True)
                for line in log[1]:
                    line = line.decode()
                    files = line.strip().split("\n")
                    for file in files:
                        print(file)
                print()
                container.stop()
                container.remove()
                with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                    results = pickle.load(f)
                parameters = results["parameters"]

                return render(request, 'MLS/RandomSearch.html', {"model": "Random_Forest_Regression",
                                                                 "parameters": parameters, "features": features_list,
                                                                 "label": label, "filename": file_name})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})