import graphene
from graphene_django.types import DjangoObjectType
from .models import User, Video
import numpy as np
import nimfa
from graphene_django import DjangoListField
n_features = 30

class Query(graphene.ObjectType):
    rate_video_list = graphene.List(graphene.Int, user_id=graphene.Int(), video_list=graphene.List(graphene.Int))
    @staticmethod
    def resolve_rate_video_list(self, info, user_id, video_list):
        user = User.objects.get(user_id=user_id)
        object_video_list = [Video.objects.get(video_id=video_id) for video_id in video_list]
        like = np.matrix([video.video_features_like if video.video_features_like else np.zeros(n_features) for video in object_video_list])
        dislike = np.matrix([video.video_features_dislike if video.video_features_dislike else np.zeros(n_features) for video in object_video_list])
        click = np.matrix([video.video_features_click if video.video_features_click else np.zeros(n_features) for video in object_video_list])
        watch = np.matrix([video.video_features_watch if video.video_features_watch else np.zeros(n_features) for video in object_video_list])

        M_like = np.dot(user.user_features_like if user.user_features_like else np.zeros(n_features), like.T)
        M_dislike = np.dot(user.user_features_dislike if user.user_features_dislike else np.zeros(n_features),
                           dislike.T)
        M_click = np.dot(user.user_features_click if user.user_features_click else np.zeros(n_features), click.T)
        M_watch = np.dot(user.user_features_watch if user.user_features_watch else np.zeros(n_features), watch.T)

        #ans = {body['user_id']: }
        return [video_list[x] for x in np.asarray( np.argsort(0.8 * M_like + 0.3 * M_click + 0.5 * M_watch - 0.8 * M_dislike)).squeeze()]


class TrainModels(graphene.Mutation):
    class Arguments:
        like = graphene.List(graphene.List(graphene.Int))
        dislike = graphene.List(graphene.List(graphene.Int))
        click = graphene.List(graphene.List(graphene.Int))
        watch = graphene.List(graphene.List(graphene.Int))

    ok = graphene.Boolean()

    def mutate(self, info, like, dislike, click, watch):
        User.objects.all().delete()
        Video.objects.all().delete()

        # Like
        bmf_like, bmf_like_fit, like_index_user, like_index_video = process_model(like)
        print('Like BMF fit, RSS: ', bmf_like_fit.distance())
        for user_index, user_features in enumerate(bmf_like.W):
            user, _ = User.objects.get_or_create(user_id=like_index_user[user_index])
            user.user_features_like = np.asarray(user_features).squeeze().tolist()
            user.save()
        for video_index, video_features in enumerate(bmf_like.H.T):
            video, _ = Video.objects.get_or_create(video_id=like_index_video[video_index])
            video.video_features_like = np.asarray(video_features).squeeze().tolist()
            video.save()

        # dislike
        bmf_dislike, bmf_dislike_fit, dislike_index_user, dislike_index_video = process_model(dislike)
        print('dislike BMF fit, RSS: ', bmf_dislike_fit.distance())
        for user_index, user_features in enumerate(bmf_dislike.W):
            user, _ = User.objects.get_or_create(user_id=dislike_index_user[user_index])
            user.user_features_dislike = np.asarray(user_features).squeeze().tolist()
        for video_index, video_features in enumerate(bmf_dislike.H.T):
            user.save()
            video, _ = Video.objects.get_or_create(video_id=dislike_index_video[video_index])
            video.video_features_dislike = np.asarray(video_features).squeeze().tolist()
            video.save()

        # click
        bmf_click, bmf_click_fit, click_index_user, click_index_video = process_model(click)
        print('click BMF fit, RSS: ', bmf_click_fit.distance())
        for user_index, user_features in enumerate(bmf_click.W):
            user, _ = User.objects.get_or_create(user_id=click_index_user[user_index])
            user.user_features_click = np.asarray(user_features).squeeze().tolist()
            user.save()
        for video_index, video_features in enumerate(bmf_click.H.T):
            video, _ = Video.objects.get_or_create(video_id=click_index_video[video_index])
            video.video_features_click = np.asarray(video_features).squeeze().tolist()
            video.save()

        # watch
        bmf_watch, bmf_watch_fit, watch_index_user, watch_index_video = process_model(watch)
        print('watch BMF fit, RSS: ', bmf_watch_fit.distance())
        for user_index, user_features in enumerate(bmf_watch.W):
            user, _ = User.objects.get_or_create(user_id=watch_index_user[user_index])
            user.user_features_watch = np.asarray(user_features).squeeze().tolist()
            user.save()
        for video_index, video_features in enumerate(bmf_watch.H.T):
            video, _ = Video.objects.get_or_create(video_id=watch_index_video[video_index])
            video.video_features_watch = np.asarray(video_features).squeeze().tolist()
            video.save()
        print("Models saved")
        return TrainModels(ok=True)

class Mutation(graphene.ObjectType):
    train_model = TrainModels.Field()

def process_model(event_list):
    data = np.matrix(event_list)
    user_index, index_user = two_way_dictionaries(data[:, 0].tolist())
    video_index, index_video = two_way_dictionaries(data[:, 1].tolist())
    X = np.zeros((len(user_index), len(video_index)))
    for user_id, video_id in event_list:
        X[user_index[user_id], video_index[video_id]] = 1
    bmf = nimfa.Bmf(X, rank=n_features)
    bmf_fit = bmf()
    return bmf, bmf_fit, index_user,  index_video

def two_way_dictionaries(array):
    forward = {v: i for i, v in enumerate(np.unique(array))}
    backward = dict(map(reversed, forward.items()))
    return forward, backward

schema = graphene.Schema(query=Query, mutation=Mutation )
