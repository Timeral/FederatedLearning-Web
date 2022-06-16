from django.contrib.auth.models import User
from django.utils import timezone
from NeuralModel.models import UserAdditionInf


def getUserAvatar(request):
    if request.user.is_authenticated:
        res = UserAdditionInf.objects.filter(username=request.user.username)
        return res[0].avatar
    else:
        return "/static/images/user/logo_default.jpg"
