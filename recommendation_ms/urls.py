from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train_model', views.train_model, name='train_model'),
    path('test_rate_all', views.test_rate_all, name='test_rate_all'),
    path('rate_video_like_list', views.rate_video_like_list, name='rate_video_like_list'),
    path('rate_video_like/<int:user_id>/<int:video_id>/', views.rate_video_like, name='rate_video_like'),
]