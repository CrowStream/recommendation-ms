#from django.shortcuts import render
import nimfa
from django.http import HttpResponse
from .models import User, Movie
import numpy as np
from pathlib import Path

from nimfa import Bmf
# Create your views here.

def index(request):
    return HttpResponse("Hello World xdxd this is index!")

def two_way_dictionaries(array):
    forward = {v: i for i, v in enumerate(np.unique(array))}
    backward = dict(map(reversed, forward.items()))
    return forward, backward

def train_model(request):
    path = Path(__file__).parent.absolute()
    print(path)
    try:
        data = np.genfromtxt('/home/sebasdeloco/Documentos/ArquiSoft/recommendation-ms/recommendation/recommendation_ms/data/ratings.csv', delimiter=',', skip_header=1)
        user_index, index_user = two_way_dictionaries(data[:, 0])
        movie_index, index_movie = two_way_dictionaries(data[:, 1])
        print("Hi")
        X = np.zeros((len(user_index),len(movie_index)))

        for user_id, movie_id, rating, _ in data:
            X[user_index[user_id], movie_index[movie_id]] = rating.astype(float)

        bmf_rating = nimfa.Bmf(X, rank=10)
        bmf_fit = bmf_rating()
        print("Model Fitted, RSS: ", bmf_fit.distance())

        for i, user in enumerate(bmf_rating.W):
            u = User()
            u.user_id = i
            u.user_features_like = np.asarray(user).squeeze().tolist()
            u.save()

        for i, movie in enumerate(bmf_rating.H.T):
            m = Movie()
            m.movie_id = i
            m.movie_features_like = np.asarray(movie).squeeze().tolist()
            m.save()
        print("Users and movies saved")
        return HttpResponse("Modelo entrenado y guardado satisfactoriamente UwU")
    except FileNotFoundError:
        return HttpResponse("Error al cargar datos")
    except Exception as e:
        print(e)
        return HttpResponse("Error al entrenar el modelo: "+str(e), status=500)
