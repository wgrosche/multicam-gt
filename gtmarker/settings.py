"""
Django settings for gtmarker project.

Generated by 'django-admin startproject' using Django 1.10.2.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.10/ref/settings/
"""

import os
from pathlib import Path
import numpy as np
import shutil
from gtm_hit.misc.wildtrack_calib import load_calibrations
from gtm_hit.misc.utils import read_calibs, get_frame_size
from gtm_hit.misc.scout_calib import load_scout_calib

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.10/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '%z1g%^3%nf-k3sf$i^qra_d*0m4745c57f&(su(2=&nuwt#=z1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

#ALLOWED_HOSTS = ['127.0.0.1']
ALLOWED_HOSTS = ['10.90.43.13', 'pedestriantag.epfl.ch','localhost','127.0.0.1', '0.0.0.0',"192.168.100.23", "iccvlabsrv15.iccluster.epfl.ch"]

# Application definition

INSTALLED_APPS = [
    'marker.apps.MarkerConfig',
    'gtm_hit.apps.Gtm_hitConfig',
    #'gtm_hit',
    'home',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'bootstrapform',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    'marker.middleware.RequireLoginMiddleware',
]
LOGIN_REQUIRED_URLS = (
    r'/marker/(.*)$',
)
LOGIN_REQUIRED_URLS_EXCEPTIONS = (
    r'/marker/login(.*)$',
    r'/marker/logout(.*)$',
)

LOGIN_URL = '/login/'

ROOT_URLCONF = 'gtmarker.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'gtmarker.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.10/ref/settings/#databases

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
#     }
# }
#
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'scout',
        'USER': 'scout',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '',
    }
}
DATA_UPLOAD_MAX_NUMBER_FIELDS = None
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': 'mydatabase', # This is where you put the name of the db file. 
#                  # If one doesn't exist, it will be created at migration time.
#     }
# }
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.postgresql_psycopg2',
#         'NAME': 'pedestriantag',
#         'USER': 'pedestriantag',
#         'PASSWORD': 'lAzyLift96',
#         'HOST': 'localhost',
#         'PORT': '',
#     }
# }

# Password validation
# https://docs.djangoproject.com/en/1.10/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/1.10/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.10/howto/static-files/

print(BASE_DIR) 
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join('gtm_hit/static')

#STATICFILES_DIRS = (os.path.join(BASE_DIR, 'gtm_hit'),)

# Additional locations of static files
#STATICFILES_DIRS = [
#location of your application, should not be public web accessible 
# os.path.join(BASE_DIR, '/home/static/'),
# os.path.join(BASE_DIR, '/gtm_him/static/'),
# os.path.join(BASE_DIR, '/marker/static/')
#]

SAVES = '/labels/'

# Constants
#
# f_rect = open('./marker/static/marker/cst.txt', 'r')
# lines = f_rect.readlines()
# f_rect.close()
# NB_WIDTH = int(lines[2].split()[1])
# NB_HEIGHT = int(lines[3].split()[1])
# NB_RECT = NB_WIDTH * NB_HEIGHT
# MAN_RAY = float(lines[4].split()[1])
# MAN_HEIGHT = float(lines[5].split()[1])
# REDUCTION = float(lines[6].split()[1])
# NB_CAMS = int(lines[9].split()[1])

DELTA_SEARCH = 5

#TEMPLATES[0]['OPTIONS']['context_processors'].append("marker.context_processors.rectangles_processor")

# try:
#     rectangles_file = './marker/static/marker/rectangles.pom'#480x1440.pom'
#     f_rect = open(rectangles_file, 'r')
#     lines = f_rect.readlines()
#     f_rect.close()
#     if lines[0].split()[0] != "WIDTH":
#         messagebox.showerror("Error","Incorrect file header")
#     else:
#         NB_WIDTH = int(lines[2].split()[1])
#         NB_HEIGHT = int(lines[3].split()[1])
#         NB_RECT = NB_WIDTH * NB_HEIGHT
#         MAN_RAY = float(lines[4].split()[1])
#         MAN_HEIGHT = float(lines[5].split()[1])
#         REDUCTION = float(lines[6].split()[1])
#         NB_CAMS = int(lines[9].split()[1])
#         incr = 0
#         test = []
#         FIND_RECT = [[{} for _ in range(2913)] for _ in range(NB_CAMS)]
#         RECT = [{} for _ in range(NB_CAMS)]
#         for line in lines[10:]:
#             l = line.split()
#             cam = int(l[1])
#             id_rect = int(l[2])
#             if l[3] != "notvisible":
#                 a, b, c, d = l[3:]
#                 a = int(a)
#                 b = int(b)
#                 c = int(c)
#                 d = int(d)
#                 ratio = 180/(d-b)
#                 if d < 5000:
#                     if abs(c - a) < abs(d - b):
#                         RECT[cam][id_rect] = (a, b, c, d,ratio)
#                         FIND_RECT[cam][d][(a + c) // 2] = id_rect
#         # NB_CAMS = 4
# except FileNotFoundError:
#         print("Error: Rectangle file not found")



# DSETNAME = "rayon4"
# CAMS = ["cam1","cam2","cam3","cam4"]
# FRAME_SIZES = get_frame_size(DSETNAME, CAMS, STARTFRAME)
# CALIBS = read_calibs(Path("./gtm_hit/static/gtm_hit/dset/"+DSETNAME+"/calibrations/full_calibration.json"), CAMS)
# NB_CAMS = len(CAMS)

# need to: establish symlinked folders for get frame size etc
DSETNAME = "SCOUT"
WORKER_ID = 'TEST'
DSETPATH = Path("./gtm_hit/static/gtm_hit/dset/") / DSETNAME
SYMLINK_DEST_FRAMES = DSETPATH / "frames"
# SYMLINK_SOURCE_FRAMES = Path('/cvlabscratch/home/engilber/datasets/SCOUT/collect_30_05_2024/sync_frame_seq_1')
CALIBPATH = DSETPATH / "calibrations"
# CALIB_SRC = Path("/cvlabscratch/home/engilber/datasets/SCOUT/collect_30_05_2024/sync_frame_seq_1/calibrations/calibrations")
FPS = 1 # framerate of input video (note, assumes 10fps base)
NUM_FRAMES = 100#12000
FRAME_START = 0
FRAME_END = FRAME_START + NUM_FRAMES
HEIGHT = 1.8
RADIUS = 0.5 #person radius
FLAT_GROUND = True # Whether or not to use the mesh for dataset generation and annotation
FRAME_SKIP = int(float(10 / FPS))
TIMEWINDOW = 10 * FRAME_SKIP # cropped frames loaded when selecting a bounding box (on either side)

VALIDATIONCODES = []
STARTFRAME = 2
NBFRAMES = NUM_FRAMES + 10
LASTLOADED = 0
INCREMENT = FRAME_SKIP
UNLABELED = list(range(0,NBFRAMES,INCREMENT))

STEPL = 0.02
MOVE_STEP = 0.02 #same as stepl vidis ovoDA
SIZE_CHANGE_STEP=0.03
# NOTE: run data creation with full cameras before bed!
CAMS = [Path(cam).name.replace('_0.json', '') for cam in CALIBPATH.iterdir()]#["cam1","cam2","cam3","cam4","cam5","cam6","cam7","cam8"]
FRAME_SIZES = get_frame_size(DSETNAME, CAMS, STARTFRAME)
#CALIBS = read_calibs(Path("./gtm_hit/static/gtm_hit/dset/"+DSETNAME+"/calibrations/full_calibration.json"), CAMS)
NB_CAMS = len(CAMS)
CALIBS= load_scout_calib(CALIBPATH, cameras=CAMS)
ROTATION_THETA = np.pi/24
UNDISTORTED_FRAMES=False

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# MESHPATH = Path("/cvlabscratch/home/engilber/datasets/SCOUT/collect_30_05_2024/scene_dense_textured_cleanup.ply")

MESHPATH = Path("/cvlabdata2/home/grosche/dev/calibration") \
    / "scene_dense_texturet_decimate_1_manual_cleanup.ply"
import trimesh
try:
    import trimesh.ray.ray_pyembree
except:
    print("It's going to be slow")
MESH = trimesh.load(MESHPATH)

import json
from shapely.geometry import Polygon
from gtm_hit.misc.geometry import get_polygon_from_points_3d



ROIjson = json.load(open('/cvlabdata2/home/grosche/dev/calibration/ROI_annotated_polygon.json'))

ROI = {}
for cam_name, polygon in ROIjson['points_3d'].items():
    ROI[cam_name] = get_polygon_from_points_3d(polygon)