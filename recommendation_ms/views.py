#from django.shortcuts import render
import json
import nimfa
from django.http import HttpResponse, JsonResponse
import pickle
import numpy as np
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt
from .models import Profile, Video
from nimfa import Bmf

# Create your views here.
#Paths to serialize model and dictionaires
path = "./recommendation_ms/data/"

n_features = 30

def index(request):
    return HttpResponse("Hello World xdxd this is recommendation system's index!")

def two_way_dictionaries(array):
    forward = {v: i for i, v in enumerate(np.unique(array))}
    backward = dict(map(reversed, forward.items()))
    return forward, backward

@csrf_exempt
def train_model(request):
    Profile.objects.all().delete()
    Video.objects.all().delete()
    body = json.loads(request.body)

    #Like
    bmf_like, bmf_like_fit, like_index_profile, like_index_video = process_model(body['like'])
    print('Like BMF fit, RSS: ',bmf_like_fit.distance())
    for profile_index, profile_features in enumerate(bmf_like.W):
        profile, _ = Profile.objects.get_or_create(profile_id=like_index_profile[profile_index])
        profile.profile_features_like = np.asarray(profile_features).squeeze().tolist()
        profile.save()
    for video_index, video_features in enumerate(bmf_like.H.T):
        video, _ = Video.objects.get_or_create(video_id=like_index_video[video_index])
        video.video_features_like = np.asarray(video_features).squeeze().tolist()
        video.save()

    # dislike
    bmf_dislike, bmf_dislike_fit, dislike_index_profile, dislike_index_video = process_model(body['dislike'])
    print('dislike BMF fit, RSS: ', bmf_dislike_fit.distance())
    for profile_index, profile_features in enumerate(bmf_dislike.W):
        profile, _ = Profile.objects.get_or_create(profile_id=dislike_index_profile[profile_index])
        profile.profile_features_dislike = np.asarray(profile_features).squeeze().tolist()
    for video_index, video_features in enumerate(bmf_dislike.H.T):
        profile.save()
        video, _ = Video.objects.get_or_create(video_id=dislike_index_video[video_index])
        video.video_features_dislike = np.asarray(video_features).squeeze().tolist()
        video.save()

    # click
    bmf_click, bmf_click_fit, click_index_profile, click_index_video = process_model(body['click'])
    print('click BMF fit, RSS: ', bmf_click_fit.distance())
    for profile_index, profile_features in enumerate(bmf_click.W):
        profile, _ = Profile.objects.get_or_create(profile_id=click_index_profile[profile_index])
        profile.profile_features_click = np.asarray(profile_features).squeeze().tolist()
        profile.save()
    for video_index, video_features in enumerate(bmf_click.H.T):
        video, _ = Video.objects.get_or_create(video_id=click_index_video[video_index])
        video.video_features_click = np.asarray(video_features).squeeze().tolist()
        video.save()

    # watch
    bmf_watch, bmf_watch_fit, watch_index_profile, watch_index_video = process_model(body['watch'])
    print('watch BMF fit, RSS: ', bmf_watch_fit.distance())
    for profile_index, profile_features in enumerate(bmf_watch.W):
        profile, _ = Profile.objects.get_or_create(profile_id=watch_index_profile[profile_index])
        profile.profile_features_watch = np.asarray(profile_features).squeeze().tolist()
        profile.save()
    for video_index, video_features in enumerate(bmf_watch.H.T):
        video, _ = Video.objects.get_or_create(video_id=watch_index_video[video_index])
        video.video_features_watch = np.asarray(video_features).squeeze().tolist()
        video.save()
    print("Models saved")
    return JsonResponse({'trained': True})

def process_model(event_list):
    data = np.matrix(event_list)
    profile_index, index_profile = two_way_dictionaries(data[:, 0].tolist())
    video_index, index_video = two_way_dictionaries(data[:, 1].tolist())
    X = np.zeros((len(profile_index), len(video_index)))
    for profile_id, video_id in event_list:
        X[profile_index[profile_id], video_index[video_id]] = 1
    bmf = nimfa.Bmf(X, rank=n_features)
    bmf_fit = bmf()
    return bmf, bmf_fit, index_profile,  index_video

@csrf_exempt
def rate_video_list(request):
    body = json.loads(request.body)
    profile = Profile.objects.get(profile_id=body['profile_id'])
    video_list = [Video.objects.get(video_id=video_id) for video_id in body['video_list']]
    n_videos = body['n_videos']
    like = np.matrix([video.video_features_like if video.video_features_like else np.zeros(n_features) for video in video_list])
    dislike = np.matrix([video.video_features_dislike if video.video_features_dislike else np.zeros(n_features) for video in video_list])
    click = np.matrix([video.video_features_click if video.video_features_click else np.zeros(n_features) for video in video_list])
    watch = np.matrix([video.video_features_watch if video.video_features_watch else np.zeros(n_features) for video in video_list])

    M_like = np.dot(profile.profile_features_like if profile.profile_features_like else np.zeros(n_features), like.T)
    M_dislike = np.dot(profile.profile_features_dislike if profile.profile_features_dislike else np.zeros(n_features), dislike.T)
    M_click = np.dot(profile.profile_features_click if profile.profile_features_click else np.zeros(n_features), click.T)
    M_watch = np.dot(profile.profile_features_watch if profile.profile_features_watch else np.zeros(n_features), watch.T)

    ans = [body['video_list'][x] for x in np.asarray(np.argsort(0.8 * M_like + 0.3 * M_click + 0.5 * M_watch - 0.8 * M_dislike)).squeeze()]
    #response = HttpResponse(ans)
    #response.write({"videoList": ans})
    return JsonResponse({"videos": ans[:n_videos]})
