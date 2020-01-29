from django.shortcuts import render
import os
import pickle
import docker


client = docker.from_env()


def Decision_Tree_Regression(request):
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
                    criterion = request.POST['criterion']
                    splitter = request.POST['splitter']
                    max_depth = None if request.POST['max_depth'] == "None" else request.POST['max_depth']
                    if max_depth is not None:
                        max_depth = float(request.POST['max_depth_value'])
                    min_samples_split = int(request.POST['min_samples_split'])
                    min_samples_leaf = int(request.POST['min_samples_leaf'])
                    min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
                    max_features = request.POST['max_features']
                    if max_features == "Int":
                        max_features = int(request.POST['max_features_integer'])
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']
                    if random_state is not None:
                        random_state = int(request.POST['random_state_value'])
                    max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST['max_leaf_nodes']
                    if max_leaf_nodes is not None:
                        max_leaf_nodes = int(request.POST['max_leaf_nodes_value'])
                    min_impurity_decrease = float(request.POST['min_impurity_decrease'])
                    min_impurity_split = float(request.POST['min_impurity_split'])
                    presort = True if request.POST['presort'] == "True" else False

                else:
                    criterion = request.POST['criterion']
                    splitter = request.POST['splitter']
                    max_depth = None if request.POST['max_depth'] == "None" else request.POST['max_depth']
                    if max_depth is not None:
                        max_depth = int(max_depth)
                    min_samples_split = int(request.POST['min_samples_split'])
                    min_samples_leaf = int(request.POST['min_samples_leaf'])
                    min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
                    max_features = None if request.POST['max_features'] == "None" else request.POST['max_features']
                    if max_features != "auto" and max_features != "sqrt" and max_features != "log2" and max_features is not None:
                        max_features = int(max_features)
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']
                    if random_state is not None:
                        random_state = int(random_state)
                    max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST[
                        'max_leaf_nodes']
                    if max_leaf_nodes is not None:
                        max_leaf_nodes = int(max_leaf_nodes)
                    min_impurity_decrease = float(request.POST['min_impurity_decrease'])
                    min_impurity_split = None if request.POST['min_impurity_split'] == 'None' else request.POST['min_impurity_split']
                    if min_impurity_split is not None:
                        min_impurity_split = int(min_impurity_split)
                    presort = True if request.POST['presort'] == "True" else False

                hyperparameters = {"filename": file_name,
                                   "user": user_name,
                                   "label": label,
                                   "ratio": ratio,
                                   "cv": cv,
                                   "featureList": features_list,
                                   "criterion": criterion,
                                   "splitter": splitter,
                                   "max_depth": max_depth,
                                   "min_samples_split": min_samples_split,
                                   "min_samples_leaf": min_samples_leaf,
                                   "min_weight_fraction_leaf": min_weight_fraction_leaf,
                                   "max_features": max_features,
                                   "random_state": random_state,
                                   "max_leaf_nodes": max_leaf_nodes,
                                   "min_impurity_decrease": min_impurity_decrease,
                                   "min_impurity_split" : min_impurity_split,
                                   "presort": presort}

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
                    if not os.path.exists("media/user_{}/Variable".format(user)):
                        os.makedirs("media/user_{}/Variable".format(user))
                    with open('media/user_{}/Variable/hyperparameters.pkl'.format(user), 'wb') as f:
                        pickle.dump(hyperparameters, f)
                    source_dir = "/Users/tusharpachori/PycharmProjects/Major1/project_v3/MLAAS_v3/media/user_{}".format(user)
                    dest_dir = "/app/user_{}/".format(user)
                    container = client.containers.create("mlaas",volumes={source_dir: {'bind': dest_dir, 'mode': 'rw'}},
                                                         tty=True, stdin_open=True, auto_remove=False)
                    container.start()
                    log = container.exec_run("python3 Major/Docker/Decision_Tree_Regression.py {} 1".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    result, download_link = [results['result'], results['download_link']]

                    return render(request, 'MLS/result.html', {"model": "Decision_Tree_Regression",
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
                    log = container.exec_run("python3 Major/Docker/Decision_Tree_Regression.py {} 2".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    rmse_score, mean, std, scores = [results["rmse_score"], results["mean"], results["std"], results["scores"]]

                    return render(request, 'MLS/validate.html', {"model": "Decision_Tree_Regression",
                                                                 "scoring": "neg_mean_squared_error",
                                                                 "scores": scores,
                                                                 'mean': mean,
                                                                 'std': std,
                                                                 'rmse': rmse_score,'cv': range(cv),
                                                                 'cv_list': range(1, cv + 1)})

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

                hyperparameters = {"filename": file_name,
                                   "user": user_name,
                                   "label": label,
                                   "ratio": ratio,
                                   "featureList": features_list,}

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
                log = container.exec_run("python3 Major/Docker/Decision_Tree_Regression.py {} 3".format(user),
                                         stderr=True, stdout=True, stream=True)
                container.stop()
                container.remove()
                with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                    results = pickle.load(f)
                parameters = results["parameters"]
                print(parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Decision_Tree_Regression",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})