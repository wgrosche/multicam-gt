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
    # view = annotation.twod_views.all()[0].view.view_id
    return view.view_id, frame_id, person_id, [x1, x2, y1, y2], [xw, yw, zw]

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SCOUT')
    parser.add_argument('--worker', type=str, default='MARTINMEANANDDEPRESSED')
    # parser.add_argument('--output', type=str, required=True, help='Output file path')

    args = parser.parse_args()
    print('Exporting data for dataset: {} and worker: {}'.format(args.dataset, args.worker))
    annotations = Annotation.objects.filter(person__dataset__name=args.dataset, person__worker__workerID=args.worker)

    export_dict = {}
    for view in View.objects.all():
        for annotation in annotations:
            try:
                view_id, frame_id, person_id, [x1, x2, y1, y2], [xw, yw, zw] = get_frame_data(annotation, view)
                camera = settings.CAMS[view_id]
                if camera not in export_dict:
                    export_dict[camera] = {}
                if frame_id not in export_dict[camera]:
                    export_dict[camera][frame_id] = []
                export_dict[camera][frame_id].append({
                    'person_id': annotation.person.person_id,
                    'bbox': [x1, x2, y1, y2],
                    'pos': [xw, yw, zw]
                })
            except:
                continue
    # Save the each camera dict to a JSON file
        with open(f'exports/{args.dataset}_{args.worker}_{settings.CAMS[view_id]}.json', 'w') as f:
            json.dump(export_dict[camera], f)

if __name__ == '__main__':
    print('testing')
    main()
