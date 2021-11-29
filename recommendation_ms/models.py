from django.db import models
from django.contrib.postgres.fields import ArrayField
N_features = 30

# Create your models here.
class User(models.Model):
    user_id = models.IntegerField(primary_key=True)
    user_features_watch = ArrayField(models.FloatField(), size=N_features, null=True)
    user_features_click = ArrayField(models.FloatField(), size=N_features, null=True)
    user_features_like = ArrayField(models.FloatField(), size=N_features, null=True)
    user_features_dislike = ArrayField(models.FloatField(), size=N_features, null=True)

class Video(models.Model):
    video_id = models.IntegerField(primary_key=True)
    video_features_watch = ArrayField(models.FloatField(), size=N_features, null=True)
    video_features_click = ArrayField(models.FloatField(), size=N_features, null=True)
    video_features_like = ArrayField(models.FloatField(), size=N_features, null=True)
    video_features_dislike = ArrayField(models.FloatField(), size=N_features, null=True)