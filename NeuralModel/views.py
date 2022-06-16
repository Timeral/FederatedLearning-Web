from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .db_funcs import getUserAvatar

@login_required
def index(request):
    """
    /index
    首页
    表单方法:GET
    表单参数:
        firstlogin:是否为用户首次登录?是则展示欢迎信息
    """
    firstlogin = "0"
    print(getUserAvatar(request))
    try:
        firstlogin = request.GET.get('firstlogin')
    finally:
        if firstlogin == "1":
            return render(request, 'index.html', {"msg": "", "Type": "inf" ,"avatar":getUserAvatar(request)})
    return render(request, 'index.html',{'username':'',"avatar":getUserAvatar(request)})


def login_view(request):
    """
    /login
    渲染登录页面
    """
    return render(request, 'login.html',{"avatar":getUserAvatar(request)})


def password_change_view(request):
    """
    /password_change
    渲染更改密码的页面
    """
    return render(request, 'password_change.html',{"avatar":getUserAvatar(request)})


@login_required
def about(request):
    """
    /about
    渲染关于界面
    """
    return render(request, "about.html",{"avatar":getUserAvatar(request)})


@login_required
def trainCenter(request):
    """
    /trainCenter
    训练中心页面
    """
    return render(request, "trainCenter.html",{"avatar":getUserAvatar(request)})


@login_required
def predictData(request):
    """
    /predictData
    数据预测页面
    """
    return render(request, "predictData.html",{"avatar":getUserAvatar(request)})

