from django.shortcuts import render
import pickle
import os
import docker


client = docker.from_env()


def Support_Vector_Machines(request):
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
                    kernel = request.POST['kernel']
                    degree = int(request.POST['degree'])
                    gamma = request.POST['gamma']
                    coef0 = float(request.POST['coef0'])
                    tol = float(request.POST['tol'])
                    C = float(request.POST['C'])
                    shrinking = True if request.POST['shrinking'] == "True" else False
                    probability = True if request.POST['probability'] == "True" else False
                    cache_size = int(request.POST['cache_size'])
                    class_weight = None if request.POST['class_weight'] == 'None' else request.POST['class_weight']
                    verbose = True if request.POST['verbose'] == "True" else False
                    max_iter = int(request.POST['max_iter'])
                    decision_function_shape = request.POST['decision_function_shape']
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']
                else:
                    kernel = request.POST['kernel']
                    degree = int(request.POST['degree'])
                    gamma = request.POST['gamma']
                    coef0 = float(request.POST['coef0'])
                    tol = float(request.POST['tol'])
                    C = float(request.POST['C'])
                    shrinking = True if request.POST['shrinking'] == "True" else False
                    probability = True if request.POST['probability'] == "True" else False
                    cache_size = int(request.POST['cache_size'])
                    class_weight = None if request.POST['class_weight'] == 'None' else request.POST['class_weight']
                    verbose = True if request.POST['verbose'] == "True" else False
                    max_iter = int(request.POST['max_iter'])
                    decision_function_shape = request.POST['decision_function_shape']
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']

                hyperparameters = {"filename": file_name,"user": user_name,"label": label,"ratio": ratio,"cv": cv,
                                   "featureList": features_list,"C": C,"kernel": kernel,"degree": degree,
                                   "gamma": gamma, "coef0": coef0, "shrinking": shrinking, "probability": probability,
                                   "tol": tol, "cache_size": cache_size, "class_weight": class_weight,
                                   "verbose": verbose, "max_iter": max_iter,
                                   "decision_function_shape": decision_function_shape, "random_state": random_state}

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
                    log = container.exec_run("python3 Major/Docker/Support_Vector_Machines.py {} 1".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    result, download_link = [results['result'], results['download_link']]
                    return render(request, 'MLS/result.html', {"model": "Support_Vector_Machines",
                                                               "metrics": "Accuracy Score",
                                                               "result":result*100,
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
                    log = container.exec_run("python3 Major/Docker/Support_Vector_Machines.py {} 2".format(user),
                                             stderr=True, stdout=True, stream=True)
                    container.stop()
                    container.remove()
                    with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                        results = pickle.load(f)
                    rmse_score, mean, std, scores = [results["rmse_score"], results["mean"], results["std"],
                                                     results["scores"]]
                    return render(request, 'MLS/validate.html', {"model": "Support_Vector_Machines",
                                                                 "scoring": "accuracy",
                                                                 "scores": scores,'mean': mean,'std': std,
                                                                 'rmse': rmse_score, 'cv': range(cv),
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
                                   "featureList": features_list}
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
                log = container.exec_run("python3 Major/Docker/Support_Vector_Machines.py {} 3".format(user),
                                         stderr=True, stdout=True, stream=True)
                container.stop()
                container.remove()
                with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                    results = pickle.load(f)
                parameters = results["parameters"]
                print(parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Support_Vector_Machines",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})