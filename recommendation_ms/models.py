from django.db import models
from django.contrib.postgres.fields import ArrayField
N_features = 30

# Create your models here.
class Profile(models.Model):
    profile_id = models.CharField(primary_key=True, max_length=40)
    profile_features_watch = ArrayField(models.FloatField(), size=N_features, null=True)
    profile_features_click = ArrayField(models.FloatField(), size=N_features, null=True)
    profile_features_like = ArrayField(models.FloatField(), size=N_features, null=True)
    profile_features_dislike = ArrayField(models.FloatField(), size=N_features, null=True)

class Video(models.Model):
    video_id = models.IntegerField(primary_key=True)
    video_features_watch = ArrayField(models.FloatField(), size=N_features, null=True)
    video_features_click = ArrayField(models.FloatField(), size=N_features, null=True)
    video_features_like = ArrayField(models.FloatField(), size=N_features, null=True)
    video_features_dislike = ArrayField(models.FloatField(), size=N_features, null=True)