from django.urls import path
from NeuralModel import views,requests
 
urlpatterns = [
    path('login/', views.login_view),
    path('do_logout/', requests.logout_request),
    path('password_change/', views.password_change_view),
    path('do_login/', requests.login_request),
    path('do_password_change/', requests.password_change_request),
    path('do_register/', requests.register_request),
    # path('do_askforleave/', requests.leave_request),
    path('index/', views.index,name='index'),
    path('',views.index),
    path('about/', views.about, name='about'),
    path('trainCenter/', views.trainCenter, name='trainCenter'),
    path('predictData/', views.predictData, name='predictData'),
]
