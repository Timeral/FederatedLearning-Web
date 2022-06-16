from django.apps import AppConfig
from django.db.models.signals import post_migrate


def init_superuser():
    """
    初始化管理员账户
    """
    from django.contrib.auth.models import User
    if (User.objects.filter(username="admin").count() == 0):
        user1 = User.objects.create_superuser("admin", None, "1234567890")
        user1.save()


def init_user():
    """
    初始化用户
    """
    from django.contrib.auth.models import User
    try:
        if (User.objects.filter(username="190827010212").count() == 0):
            user = User.objects.create_user("190827010212", "timeral@qq.com", "1234567890")
            user.first_name="刘延亮"
            user.save()
        if (User.objects.filter(username="190809010326").count() == 0):
            user = User.objects.create_user("190809010326", "2139447940@qq.com", "1234567890")
            user.first_name="王帅斌"
            user.save()
        if (User.objects.filter(username="190809010409").count() == 0):
            user = User.objects.create_user("190809010409", "tigero97@qq.com", "1234567890")
            user.first_name="唐龙"
            user.save()
    except:
        print('Duplicate student, ignoring...')


def init_user_addinf():
    """
    初始化附加信息
    """
    from .models import UserAdditionInf
    try:
        if (UserAdditionInf.objects.filter(username="190827010212").count() == 0):
            inf = UserAdditionInf.objects.create(username="190827010212", avatar="/static/images/user/logo_190827010212.jpg")
            inf.save()
        if (UserAdditionInf.objects.filter(username="190809010326").count() == 0):
            inf = UserAdditionInf.objects.create(username="190809010326", avatar="/static/images/user/logo_190809010326.jpg")
            inf.save()
        if (UserAdditionInf.objects.filter(username="190809010409").count() == 0):
            inf = UserAdditionInf.objects.create(username="190809010409", avatar="/static/images/user/logo_190809010409.jpg")
            inf.save()
    except:
        print('Duplicate student, ignoring...')


def do_init_data(sender, **kwargs):
    """
    初始化django数据库
    """
    print('Initializing the superuser...')
    init_superuser()
    print('Initializing user...')
    init_user()
    print('Initializing addinf...')
    init_user_addinf()
    print('-->Done.')

class NeuralmodelConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'NeuralModel'
    def ready(self):
        post_migrate.connect(do_init_data, sender=self)
