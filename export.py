import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
django.setup()

from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset, Annotation2DView, View
import argparse
import json
from datetime import datetime


from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person, Dataset, Annotation2DView, View
import argparse
import json

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


def get_projection_from_nearest_valid_camera():
    """
    Uses calib to find distance between camera and point for each view.
    """

    for view in View.objects.all():
        view_id = view.view_id
        camera_name = settings.CAMS[view_id]
        calib = settings.CALIBS[camera_name]

        


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SCOUT')
    parser.add_argument('--worker', type=str, default='HIGHRESMESH')
    # parser.add_argument('--output', type=str, required=True, help='Output file path')

    args = parser.parse_args()
    print('Exporting data for dataset: {} and worker: {}'.format(args.dataset, args.worker))

    frames = MultiViewFrame.objects.filter(dataset__name=args.dataset, worker__workerID=args.worker, frame_id__in=range(0, 2000))
    annotations = Annotation.objects.filter(person__dataset__name=args.dataset, person__worker__workerID=args.worker, frame__in=frames)


    export_dict = {}
    for view in View.objects.all():
        print('Exporting data for view: {}'.format(view.view_id))
        view_data = Annotation2DView.objects.filter(annotation__in=annotations, view=view).values('annotation__person__person_id', 'x1', 'y1', 'x2', 'y2', 'annotation__Xw', 'annotation__Yw', 'annotation__Zw', 'annotation__frame__frame_id')
        for view_data_entry in view_data:
            view_id = view.view_id
            person_id = view_data_entry['annotation__person__person_id']
            x1, y1, x2, y2 = view_data_entry['x1'], view_data_entry['y1'], view_data_entry['x2'], view_data_entry['y2']
            xw, yw, zw = view_data_entry['annotation__Xw'], view_data_entry['annotation__Yw'], view_data_entry['annotation__Zw']
            frame_id = view_data_entry['annotation__frame__frame_id']
            if [x1, x2, y1, y2] == [-1, -1, -1, -1]:
                continue
            camera = settings.CAMS[view_id]
            if camera not in export_dict:
                export_dict[camera] = {}
            if frame_id not in export_dict[camera]:
                export_dict[camera][frame_id] = []
            export_dict[camera][frame_id].append({
                'person_id': person_id,
                'bbox': [x1, y1, x2, y2],
                'pos': [xw, yw, zw]
            })
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = f'exports/{args.dataset}_{args.worker}_{settings.CAMS[view_id]}_{timestamp}.json'
        with open(filepath, 'w') as f:
            json.dump(export_dict[camera], f)

if __name__ == '__main__':
    main()
