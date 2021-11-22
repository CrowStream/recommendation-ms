#from django.shortcuts import render
import json

import nimfa
from django.http import HttpResponse
from .models import User, Video
import numpy as np
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt

from nimfa import Bmf
# Create your views here.

def index(request):
    return HttpResponse("Hello World xdxd this is index!")

def two_way_dictionaries(array):
    forward = {v: i for i, v in enumerate(np.unique(array))}
    backward = dict(map(reversed, forward.items()))
    return forward, backward

@csrf_exempt
def rate_video_like(request, user_id, video_id):
    user = User.objects.get(user_id=user_id)
    video = Video.objects.get(video_id=video_id)
    response = HttpResponse()
    response.write({'like_rating': np.dot(user.user_features_like, np.array(video.video_features_like).T)})
    return HttpResponse(response)

@csrf_exempt
def rate_video_like_list(request):
    body = json.loads(request.body)
    print(body["user_id"])
    user = User.objects.get(user_id=body["user_id"]).user_features_like
    videos = np.matrix([[video_id] + Video.objects.get(video_id=video_id).video_features_like for video_id in body['video_list']])
    M = np.dot(user, videos[:,1:].T)
    ans = {body['user_id']: list(np.asarray(videos[np.argsort(M), 0]).squeeze().astype(int))}
    response = HttpResponse()
    response.write(ans)
    return response


def test_rate_all(request):
    U = np.matrix([[user.user_id] + user.user_features_like for user in User.objects.all()])
    V = np.matrix([[video.video_id] + video.video_features_like for video in Video.objects.all()])
    M = np.dot(U[:, 1:], V[:, 1:].T)

    dicc = {U[u, 0].astype(int): list( V[x, 0].astype(int) for x in np.asarray(np.argsort(M[u, :])).squeeze()) for u in range(U.shape[0])}
    response = HttpResponse()
    response.write(dicc)
    return response

def train_model(request):
    path = Path(__file__).parent.absolute()
    print(path)
    try:
        data = np.genfromtxt('/home/sebasdeloco/Documentos/ArquiSoft/recommendation-ms/recommendation/recommendation_ms/data/ratings.csv', delimiter=',', skip_header=1)
        user_index, index_user = two_way_dictionaries(data[:, 0])
        video_index, index_video = two_way_dictionaries(data[:, 1])
        X = np.zeros((len(user_index),len(video_index)))

        for user_id, video_id, rating, _ in data:
            X[user_index[user_id], video_index[video_id]] = rating.astype(float)

        bmf_rating = nimfa.Bmf(X, rank=10)
        bmf_fit = bmf_rating()
        print("Model Fitted, RSS: ", bmf_fit.distance())

        for i, user in enumerate(bmf_rating.W):
            u = User()
            u.user_id = i
            u.user_features_like = np.asarray(user).squeeze().tolist()
            u.save()

        for i, video in enumerate(bmf_rating.H.T):
            v = Video()
            v.video_id = i
            v.video_features_like = np.asarray(video).squeeze().tolist()
            v.save()
        print("Users and videos saved")
        return HttpResponse("Modelo entrenado y guardado satisfactoriamente UwU")
    except FileNotFoundError:
        return HttpResponse("Error al cargar datos :c")
    except Exception as e:
        print(e)
        return HttpResponse("Error al entrenar el modelo: "+str(e), status=500)
