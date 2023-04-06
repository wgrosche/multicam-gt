from django.urls import re_path as url
from django.urls import path

from . import views

urlpatterns = [
    url(r'^&MID=(?P<workerID>[A-Z0-9]+)$', views.index, name="index"),
    url(r'^$', views.requestID, name="requestID"),
    #url(r'^(?P<workerID>[A-Z0-9]+)/processInit$', views.processInit, name="processInit"),
    path('<str:dataset_name>/<str:workerID>/processInit/', views.processInit, name='processInit'),
    url(r'^(?P<workerID>[A-Z0-9]+)/processIndex$', views.processIndex, name="processIndex"),
    url(r'^(?P<workerID>[A-Z0-9]+)/processTuto$', views.processTuto, name="processTuto"),
    #url(r'^(?P<workerID>[A-Z0-9]+)$', views.dispatch, name="dispatch"),
    path('<str:dataset_name>/<str:workerID>/', views.dispatch, name="dispatch"),
    url(r'^(?P<workerID>[A-Z0-9]+)/index$', views.index, name="index"),
    url(r'^(?P<workerID>[A-Z0-9]+)/tuto$', views.tuto, name="tuto"),
    #url(r'^(<str:dataset_name>/?P<workerID>[A-Z0-9]+)/frame$', views.frame, name='frame'),
    path('<str:dataset_name>/<str:workerID>/frame', views.frame, name='frame'),
    url(r'^(?P<workerID>[A-Z0-9]+)/processFrame$', views.processFrame, name="processFrame"),
    url(r'^.*click',views.click,name="click"),
    url(r'^.*move',views.move,name="move"),
    url(r'^.*action$',views.action,name="action"),
    url(r'^.*changeframe$', views.changeframe, name='changeframe'),
    url(r'^.*save$',views.save,name='save'),
    url(r'^.*load$',views.load,name='load'),
    url(r'^.*loadprev$',views.load_previous,name='loadprev'),
    url(r'^.*processFinish$',views.processFinish,name='processFinish'),
    url(r'^.*tracklet$',views.tracklet,name='tracklet'),
    url(r'^.*changeid$',views.change_id,name='changeid'),
    url(r'^.*person$',views.person_action,name='personAction'),
    url(r'^.*timeview$',views.timeview,name='timeview'),
    url(r'^.*interpolate$',views.interpolate,name='interpolate'),
    url(r'^(?P<workerID>[A-Z0-9]+)/finish$',views.finish,name='finish'),
]
