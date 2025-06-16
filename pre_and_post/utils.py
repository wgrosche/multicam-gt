"""
Helper functions for the post processing stage.

Includes:

- extract frame data
"""
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
django.setup()

# from ..gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset, Annotation2DView, View
import argparse
import json
from datetime import datetime


def get_frame_data(annotation:Annotation, view:View):
    view_id = view.view_id
    person_id = annotation.person.person_id
    x1, y1, x2, y2 = annotation.twod_views.all()[view_id].x1, annotation.twod_views.all()[view_id].y1, annotation.twod_views.all()[view_id].x2, annotation.twod_views.all()[view_id].y2
    xw, yw, zw = annotation.Xw, annotation.Yw, annotation.Zw
    frame_id = annotation.frame.frame_id
    # if [x1, x2, y1, y2] == [-1, -1, -1, -1]:
    #     return None
    # view = annotation.twod_views.all()[0].view.view_id
    return view.view_id, frame_id, person_id, [x1, y1, x2, y2], [xw, yw, zw]


def extract_frame_data(frame_id:int, annotation:Annotation, view:View, worker:Worker):
    """
    
    """
    pass