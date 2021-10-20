import sklearn.naive_bayes
import sklearn.ensemble
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle as pkl


def factory(dataset, model_type, load_trained=False):
    if not load_trained:
        if dataset == 'boston' and model_type == 1:
            classifier = sklearn.naive_bayes.GaussianNB()
        if dataset == 'online_news' and model_type == 2:
            classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
    else:
        if dataset == 'online_news' and model_type == 2:
            classifier = pkl.load( open('../models/userstudy/news_classifier.pkl', 'rb'))

    return classifier
