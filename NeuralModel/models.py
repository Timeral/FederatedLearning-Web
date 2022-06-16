from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db.models.deletion import CASCADE

# Create your models here.
class UserAdditionInf(models.Model):
    """
    学生附加信息
    """
    username = models.CharField("用户名称", max_length=20, primary_key=True)
    avatar = models.CharField("头像位置", max_length=50, default="/static/images/user/logo_default.jpg")

    class Meta:
        db_table = "附加用户信息"
