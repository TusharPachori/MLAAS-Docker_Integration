from django.shortcuts import render
import pickle
import os
import docker


client = docker.from_env()



def Linear_Classifiers_Naive_Bayes_Classifier(request):
    if request.method == 'POST':
        try:
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

            priors = None if request.POST['priors']=="None" else request.POST['priors']
            var_smoothing= float(request.POST['var_smoothing'])

            hyperparameters = {"filename": file_name, "user": user_name, "label": label, "ratio": ratio, "cv": cv,
                               "featureList": features_list, "priors": priors, "var_smoothing": var_smoothing}


            if request.POST['submit'] == "TRAIN":
                if not os.path.exists("media/user_{}/Variable".format(user)):
                    os.makedirs("media/user_{}/Variable".format(user))
                with open('media/user_{}/Variable/hyperparameters.pkl'.format(user), 'wb') as f:
                    pickle.dump(hyperparameters, f)
                source_dir = "/Users/tusharpachori/PycharmProjects/Major1/project_v3/MLAAS_v3/media/user_{}".format(
                    user)
                dest_dir = "/app/user_{}/".format(user)
                container = client.containers.create("mlaas", volumes={source_dir: {'bind': dest_dir, 'mode': 'rw'}},
                                                     tty=True, stdin_open=True, auto_remove=False)
                container.start()
                log = container.exec_run("python3 Major/Docker/Linear_Classifiers_Naive_Bayes_Classifier.py {} 1".format(user),
                                         stderr=True, stdout=True, stream=True)
                container.stop()
                container.remove()
                with open('media/user_{}/Variable/result.pkl'.format(user), 'rb') as f:
                    results = pickle.load(f)
                result, download_link = [results['result'], results['download_link']]
                return render(request, 'MLS/result.html', {"model": "Linear_Classifiers_Naive_Bayes_Classifier",
                                                           "metrics": "Accuracy Score",
                                                           "result": result*100,
                                                           "link": download_link})

            elif request.POST['submit'] == "VALIDATE":
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
                log = container.exec_run("python3 Major/Docker/Linear_Classifiers_Naive_Bayes_Classifier.py {} 2".format(user),
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

                return render(request, 'MLS/validate.html', {"model": "Linear_Classifiers_Naive_Bayes_Classifier",
                                                             "scoring": "accuracy",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score,
                                                             'cv': range(cv),
                                                             'cv_list': range(1, cv+1)})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})