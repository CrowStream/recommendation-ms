# Generated by Django 3.2.9 on 2021-12-04 17:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recommendation_ms', '0008_auto_20211204_1656'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='profile_id',
            field=models.CharField(max_length=40, primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='video',
            name='video_id',
            field=models.CharField(max_length=40, primary_key=True, serialize=False),
        ),
    ]
