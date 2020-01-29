from django.shortcuts import render
import pickle
import os
import docker


client = docker.from_env()


def Nearest_Neighbor(request):
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
                    n_neighbors = int(request.POST['n_neighbors'])
                    weights = request.POST['weights']
                    algorithm = request.POST['algorithm']
                    leaf_size = int(request.POST['leaf_size'])
                    p = int(request.POST['p'])
                    metric = request.POST['metric']
                    metric_params = None if request.POST['metric_params'] == "None" else request.POST['metric_params']
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(request.POST['n_jobs_value'])
                else:
                    n_neighbors = int(request.POST['n_neighbors'])
                    weights = request.POST['weights']
                    algorithm = request.POST['algorithm']
                    leaf_size = int(request.POST['leaf_size'])
                    p = int(request.POST['p'])
                    metric = request.POST['metric']
                    metric_params = None if request.POST['metric_params'] == "None" else request.POST['metric_params']
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(n_jobs)

                hyperparameters = {"filename": file_name, "user": user_name, "label": label, "ratio": ratio, "cv": cv,
                                   "featureList": features_list, "n_neighbors": n_neighbors, "weights": weights,
                                   "algorithm": algorithm, "leaf_size": leaf_size, "p": p, "metric":metric,
                                   "metric_params": metric_params, "n_jobs": n_jobs}

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
                    log = container.exec_run("python3 Major/Docker/Nearest_Neighbor.py {} 1".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    result, download_link = [results['result'], results['download_link']]

                    return render(request, 'MLS/result.html', {"model": "Nearest_Neighbor",
                                                               "metrics": "Accuracy Score",
                                                               "result": result*100,
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
                    log = container.exec_run("python3 Major/Docker/Nearest_Neighbor.py {} 2".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    rmse_score, mean, std, scores = [results["rmse_score"], results["mean"], results["std"],
                                                     results["scores"]]

                    return render(request, 'MLS/validate.html', {"model": "Nearest_Neighbor",
                                                                 "scoring": "accuracy",
                                                                 "scores": scores,
                                                                 'mean': mean,
                                                                 'std': std,
                                                                 'rmse': rmse_score,
                                                                 'cv': range(cv),
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

                hyperparameters = {"filename": file_name,
                                   "user": user_name,
                                   "label": label,
                                   "ratio": ratio,
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
                log = container.exec_run("python3 Major/Docker/Nearest_Neighbor.py {} 3".format(user),
                                         stderr=True, stdout=True, stream=True)
                container.stop()
                container.remove()
                with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                    results = pickle.load(f)
                parameters = results["parameters"]
                print(parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Nearest_Neighbor",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})