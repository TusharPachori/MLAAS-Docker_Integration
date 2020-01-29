from django.shortcuts import render
import pickle
import os
import docker


client = docker.from_env()


def Simple_Linear_Regression(request):
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
                    fit_intercept = True if request.POST['fit_intercept'] == "True" else False
                    normalize = True if request.POST['normalize'] == "True" else False
                    copy_X = True if request.POST['copy_X'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                else:
                    fit_intercept = True if request.POST['fit_intercept'] == "True" else False
                    normalize = True if request.POST['normalize'] == "True" else False
                    copy_X = True if request.POST['copy_X'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(n_jobs)

                hyperparameters = {"filename": file_name,
                                   "user": user_name,
                                   "label": label,
                                   "ratio": ratio,
                                   "cv": cv,
                                   "featureList": features_list,
                                   "fit_intercept": fit_intercept,
                                   "normalize": normalize,
                                   "copy_X": copy_X,
                                   "n_jobs": n_jobs}

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
                    log = container.exec_run("python3 Major/Docker/Simple_Linear_Regression.py {} 1".format(user),
                                             stderr=True, stdout=True, stream=True)
                    # for line in log[1]:
                    #     line = line.decode()
                    #     files = line.strip().split("\n")
                    #     for file in files:
                    #         print(file)
                    # print()
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    result, download_link = [results['result'], results['download_link']]
                    return render(request, 'MLS/result.html', {"model": "Simple_Linear_Regression",
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
                    log = container.exec_run("python3 Major/Docker/Simple_Linear_Regression.py {} 2".format(user),
                                             stderr=True, stdout=True, stream=True)
                    # for line in log[1]:
                    #     line = line.decode()
                    #     files = line.strip().split("\n")
                    #     for file in files:
                    #         print(file)
                    # print()
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    rmse_score, mean, std, scores = [results["rmse_score"], results["mean"], results["std"],
                                                     results["scores"]]

                    return render(request, 'MLS/validate.html', {"model": "Simple_Linear_Regression",
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

                hyperparameters = {"filename": file_name,"user": user_name,"label": label,"ratio": ratio,
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
                log = container.exec_run("python3 Major/Docker/Simple_Linear_Regression.py {} 3".format(user),
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
                print(parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Simple_Linear_Regression",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})
