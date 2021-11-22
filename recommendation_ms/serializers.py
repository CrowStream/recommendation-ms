from rest_framework import serializers
from .models import User, Video

class UserSerializer(serializers.Serializer):
    user_id = serializers.IntegerField(read_only=True)
    user_feature