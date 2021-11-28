#from django.shortcuts import render
import json

import nimfa
from django.http import HttpResponse
import pickle
import numpy as np
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt

from nimfa import Bmf
# Create your views here.
#Paths to serialize model and dictionaires
model_path = "./recommendation_ms/data/model_fitted.sav"
user_index_path = "./recommendation_ms/data/user_index.sav"
index_user_path = "./recommendation_ms/data/index_user.sav"
video_index_path = "./recommendation_ms/data/video_index.sav"
index_video_path = "./recommendation_ms/data/index_video.sav"
n_features = 30

def index(request):
    return HttpResponse("Hello World xdxd this is recommendation system's index!")

def two_way_dictionaries(array):
    forward = {v: i for i, v in enumerate(np.unique(array))}
    backward = dict(map(reversed, forward.items()))
    return forward, backward

def load_files():
    model = load_model()
    user_index = load_user_index()
    index_user = load_index_user()
    video_index = load_video_index()
    index_video = load_index_video()
    return model, user_index, index_user, video_index, index_video


def load_model():
    model = pickle.load(open(model_path, 'rb'))
    return model

def load_index_video():
    index_video = pickle.load(open(index_video_path, 'rb'))
    return index_video

def load_video_index():
    video_index = pickle.load(open(video_index_path, 'rb'))
    return video_index

def load_index_user():
    index_user = pickle.load(open(index_user_path, 'rb'))
    return index_user

def load_user_index():
    user_index = pickle.load(open(user_index_path, 'rb'))
    return user_index

@csrf_exempt
def rate_video_like(request, user_id, video_id):
    model = load_model()
    user_index = load_user_index()
    video_index = load_video_index()
    user = model.W[user_index[user_id]]
    video = model.H[:, video_index[video_id]]
    response = HttpResponse()
    response.write({'like_rating': float(np.dot(user, video))})
    return HttpResponse(response)

@csrf_exempt
def rate_video_like_list(request):
    body = json.loads(request.body)
    model, user_index, index_user, video_index, index_video = load_files()
    user = model.W[user_index[body['user_id']]]
    videos = model.H[:, [video_index[v] for v in body['video_list']]]
    M = np.dot(user, videos)
    ans = {body['user_id']: [body['video_list'][x] for x in np.asarray(np.argsort(M)).squeeze()]}
    response = HttpResponse()
    response.write(ans)
    return response


def test_rate_all(request):
    model = load_model()
    index_user = load_index_user()
    index_video = load_index_video()
    M = np.dot(model.W, model.H)
    dicc = {int(index_user[u]): list(int(index_video[x]) for x in np.asarray(np.argsort(M[u, :])).squeeze()) for u in range(model.W.shape[0])}
    response = HttpResponse()
    response.write(dicc)
    return response



def train_model(request):
    try:
        data = np.genfromtxt('./recommendation_ms/data/ratings.csv', delimiter=',', skip_header=1)
        user_index, index_user = two_way_dictionaries(data[:, 0])
        video_index, index_video = two_way_dictionaries(data[:, 1])
        X = np.zeros((len(user_index),len(video_index)))

        for user_id, video_id, rating, _ in data:
            X[user_index[user_id], video_index[video_id]] = rating.astype(float)

        bmf_rating = nimfa.Bmf(X, rank=n_features)
        bmf_fit = bmf_rating()
        print("Model Fitted, RSS: ", bmf_fit.distance())

        with open(model_path, 'wb') as model_file:
            pickle.dump(bmf_rating, model_file)
        print("Model serialized in "+model_path)

        pickle.dump(index_user, open(index_user_path, 'wb'))
        pickle.dump(user_index, open(user_index_path, 'wb'))
        pickle.dump(index_video, open(index_video_path, 'wb'))
        pickle.dump(video_index, open(video_index_path, 'wb'))
        print("dictionaires serialized")

        return HttpResponse("Modelo entrenado y guardado satisfactoriamente UwU")
    except FileNotFoundError as e:
        return HttpResponse("Error al cargar datos : "+str(e), status=500)
    except Exception as e:
        print(e)
        return HttpResponse("Error al entrenar el modelo: "+str(e), status=500)
