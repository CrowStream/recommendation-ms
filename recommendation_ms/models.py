from django.db import models
from django.contrib.postgres.fields import ArrayField
N_features = 10

# Create your models here.
class User(models.Model):
    user_id = models.IntegerField(primary_key=True)
    user_features_watch = ArrayField(models.FloatField(), size=N_features, null=True)
    user_features_click = ArrayField(models.FloatField(), size=N_features, null=True)
    user_features_like = ArrayField(models.FloatField(), size=N_features, null=True)
    user_features_dislike = ArrayField(models.FloatField(), size=N_features, null=True)

class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    movie_features_watch = ArrayField(models.FloatField(), size=N_features, null=True)
    movie_features_click = ArrayField(models.FloatField(), size=N_features, null=True)
    movie_features_like = ArrayField(models.FloatField(), size=N_features, null=True)
    movie_features_dislike = ArrayField(models.FloatField(), size=N_features, null=True)