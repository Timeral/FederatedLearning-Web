from django.http.response import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
# from NeuralModel.models import DailySignIn, StudentAddInf
from django.utils import timezone
from .db_funcs import getUserAvatar

def login_request(request):
    """
    /do_login
    用于提交登录请求；
    需要的表单方法:POST；
    需要的表单参数:
        username:注册账户名
        password:账户密码
    """
    # 若用户已经登录，也就不应该访问这个页面，直接跳转回首页
    if request.user.is_authenticated:
        return redirect("/index/")
    # 返回的表单所用的方法必须是POST，否则不响应
    if request.method == 'POST':
        # POST方法获取用户名和密码
        username = request.POST['username']
        password = request.POST['password']
        # 使用内置的方法验证用户，如果验证成功，该方法会返回对应的用户对象，否则返回None
        user = authenticate(request, username=username, password=password)
        if user is not None:
            # 用内置的方法将该用户设为登录状态
            login(request, user)
            return HttpResponseRedirect("/index/?firstlogin=1")
        else:
            # 验证不成功，则跳回登录页面，并展示错误
            return render(request, 'login.html', {"msg": "登录失败，请检查用户名和密码","avatar":getUserAvatar(request)})
    else:
        return render(request, 'login.html')


def register_request(request):
    """
    /do_register
    用于注册账户；
    需要的表单方法:POST；
    需要的表单参数:
        username:注册账户名
        password:账户密码
        name:学生昵称（名称）
        email:可选，邮件地址
    """
    if request.method == 'POST':
        # 获取所有注册需要的数据
        username = request.POST['username']
        shortname = request.POST['shortname']
        password = request.POST['password']
        name = request.POST['name']
        # 邮箱可选填，其他必填
        email = request.POST['email']
        # 查询数据库中是否存在这个用户
        user = authenticate(request, username=username, password=password)
        # 若用户不存在，则继续注册
        if user is None:
            if len(email) <= 1:
                email = None
            # 若用户
            user = User.objects.create_user(username, email, password)
            user.first_name=shortname
            user.save()
            return render(request, 'login.html', {"msg": "注册成功，您可以登录了", "Type": "inf","avatar":getUserAvatar(request)})
        else:
            return render(request, 'login.html', {"msg": "注册失败，用户已存在", "Type": "err", "avatar":getUserAvatar(request)})
    else:
        return render(request, 'login.html', {"avatar":getUserAvatar(request)})


@login_required
def logout_request(request):
    """
    /logout
    登出请求，直接访问logout即可登出，随后跳转
    """
    # 使用内置方法logout可直接登出
    logout(request)
    return render(request, 'login.html', {"msg": "您已成功登出", "Type": "inf"})


def password_change_request(request):
    """
    /do_password_change
    用于处理更改密码的申请；
    如果用户已登录，则无需输入用户名；
    需要的表单方法:POST；
    需要的表单参数:
        username:登录账户名
        old_password:旧登录密码
        new_password:新登录密码
    """
    if request.method == 'POST':
        username = None
        # 如果用户已登录，就直接获取学号，否则按照用户输入接收数据
        if request.user.is_authenticated:
            username = request.user.get_username()
        else:
            username = request.POST['username']
        old_password = request.POST['old_password']
        new_password = request.POST['new_password']
        # 根据用户名筛选出用户对象
        user = User.objects.get(username=username)

        if (user is None) or (user.check_password(old_password) is False):
            # 旧密码不吻合时，返回错误提示
            return render(request, 'password_change.html', {"msg": "无法更改，请检查用户名或密码","avatar":getUserAvatar(request)})
        else:
            # 否则设置新密码
            user.set_password(new_password)
            user.save()
            if request.user.is_authenticated:
                # 若用户已登录，则登出用户
                logout(request)
        return render(request, 'login.html', {"msg": "更新成功，请重新登录", "Type": "inf","avatar":getUserAvatar(request)})
    else:
        return render(request, 'index.html',{"avatar":getUserAvatar(request)})

