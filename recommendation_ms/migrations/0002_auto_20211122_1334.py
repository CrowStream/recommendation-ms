# Generated by Django 3.2.9 on 2021-11-22 13:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('recommendation_ms', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Movie',
            new_name='Video',
        ),
        migrations.RenameField(
            model_name='video',
            old_name='movie_features_click',
            new_name='video_features_click',
        ),
        migrations.RenameField(
            model_name='video',
            old_name='movie_features_dislike',
            new_name='video_features_dislike',
        ),
        migrations.RenameField(
            model_name='video',
            old_name='movie_features_like',
            new_name='video_features_like',
        ),
        migrations.RenameField(
            model_name='video',
            old_name='movie_features_watch',
            new_name='video_features_watch',
        ),
        migrations.RenameField(
            model_name='video',
            old_name='movie_id',
            new_name='video_id',
        ),
    ]
