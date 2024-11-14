"""
Generates statistics about tracks for a given subset of the dataset.

Usage:
    python statistics.py --dataset SCOUT --worker HIGHRESMESH

Results:
    - Number of frames
    - Number of tracks
    - Number of frames per track


"""
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gtmarker.settings')
django.setup()
from django.db.models import Count, Avg
import matplotlib.pyplot as plt
from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person,Dataset, Annotation2DView, View
import argparse
import json
from datetime import datetime
from django.db.models.functions import Lead, RowNumber


from django.db.models import Window, F#, Lag
from django.db.models.functions import Lead


from gtm_hit.misc.scout_preprocess import preprocess_scout_data, preprocess_scout_data_from_dict
from pathlib import Path
from django.conf import settings
from gtm_hit.models import MultiViewFrame, Worker, Annotation, Person, Dataset, Annotation2DView, View
import argparse
import json
import numpy as np
from tqdm import tqdm

def get_track_duration_and_length(person:Person):
    person_track = Annotation.objects.filter(person = person).values('frame__framed_id', 'Xw', 'Yw', 'Zw')
    track_duration = len(person_track['frame__frame_id'])
    positions = np.array([[track['Xw'], track['Yw'], track['Zw']] for track in person_track])
    distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    track_length = np.sum(distances)


    return track_duration, track_length


from django.db.models import Window
from django.db.models.functions import Lead
import numpy as np

def get_binned_stats(annotations):
    positions = annotations.values(
        'person__person_id', 'frame__frame_id', 'Xw', 'Yw', 'Zw'
    ).annotate(
        next_x=Window(expression=Lead('Xw', default=None), partition_by='person__person_id', order_by='frame__frame_id'),
        next_y=Window(expression=Lead('Yw', default=None), partition_by='person__person_id', order_by='frame__frame_id'),
        next_z=Window(expression=Lead('Zw', default=None), partition_by='person__person_id', order_by='frame__frame_id')
    ).order_by('person__person_id', 'frame__frame_id')

    distances = {}
    for pos in positions:
        if pos['next_x'] is not None and pos['next_y'] is not None and pos['next_z'] is not None:
            person_id = pos['person__person_id']
            if person_id not in distances:
                distances[person_id] = 0
            distances[person_id] += np.linalg.norm([
                pos['next_x'] - pos['Xw'],
                pos['next_y'] - pos['Yw'],
                pos['next_z'] - pos['Zw']
            ])

    if distances:
        distance_list = list(distances.values())
        return np.mean(distance_list)
    return 0


        


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SCOUT')
    parser.add_argument('--worker', type=str, default='HIGHRESMESH')
    # parser.add_argument('--output', type=str, required=True, help='Output file path')

    args = parser.parse_args()
    print('Calculating metrics for dataset: {} and worker: {}'.format(args.dataset, args.worker))

    # frames = MultiViewFrame.objects.filter(dataset__name=args.dataset, worker__workerID=args.worker, frame_id__in=range(0, 2000))

    # person_ids = Person.objects.filter(
    #                 dataset__name=args.dataset, worker__workerID=args.worker,
    #                 annotation__frame__frame_id__lt=2000
    #             ).values_list('person_id', flat=True).distinct()
    
    # print(f"There are {len(person_ids)} people in the dataset")
    # print(f"Max ID: {max(person_ids)}")

    # active_annotations = Annotation.objects.filter(person__dataset__name=args.dataset, person__worker__workerID=args.worker, frame__frame_id__lt=2000)
    # active_2d_annotations = Annotation2DView.objects.filter(annotation__in=active_annotations, cuboid_points__isnull=False)

    # print(f"There are {active_annotations.count()} active annotations")
    # print(f"There are {active_2d_annotations.count()} active 2d annotations")



    # avg_annotations = Person.objects.filter(
    #                                     dataset__name=args.dataset, person_id__gt=11900
    #                                 ).annotate(
    #                                     annotation_count=Count('annotation')
    #                                 ).filter(annotation_count__gt=10).aggregate(
    #                                     avg_per_person=Avg('annotation_count')
    #                                 )['avg_per_person']
    


    # # Get annotation counts per person
    # person_counts = list(Person.objects.filter(
    #     dataset__name=args.dataset, worker__workerID=args.worker, annotation__frame__frame_id__lt=2000
    #             ).annotate(
    #                 annotation_count=Count('annotation')
    #             ).filter(annotation_count__gt=50).values_list('person_id', 'annotation_count'))
    
    interesting_annotations = Annotation.objects.filter(person__in = Person.objects.filter(
                            dataset__name=args.dataset, worker__workerID=args.worker, annotation__frame__frame_id__lt=2000
                                    ).annotate(
                                        annotation_count=Count('annotation')
                                    ).filter(annotation_count__gt=20)).values('Xw', 'Yw', 'Zw', 'person__person_id')
    
    number_interesting_annotations = interesting_annotations.count()
    print(f"Number of interesting annotations {number_interesting_annotations}")

    base_binning = Person.objects.filter(
                            dataset__name=args.dataset, 
                            worker__workerID=args.worker, 
                            annotation__frame__frame_id__lt=2000).annotate(annotation_count=Count('annotation'))

    first_bin = Annotation.objects.filter(person__in = base_binning.filter(annotation_count__gt=75))#.values('Xw', 'Yw', 'Zw', 'person__person_id')
    
    second_bin =Annotation.objects.filter(person__in = base_binning.filter(annotation_count__gt=30, annotation_count__lt=75))#.values('Xw', 'Yw', 'Zw', 'person__person_id')
    
    third_bin = Annotation.objects.filter(person__in = base_binning.filter(annotation_count__gt=20, annotation_count__lt=30))#.values('Xw', 'Yw', 'Zw', 'person__person_id')
    

    print(f"Average distance top bin: {get_binned_stats(first_bin)}")
    print(f"Average distance middle bin: {get_binned_stats(second_bin)}")
    print(f"Average distance bottom bin: {get_binned_stats(third_bin)}")
    
    # distance_list = []
    # # get average distance traveled
    # for person in list(interesting_annotations.values_list('person__person_id')):
    #     person_id = person[0] if isinstance(person, tuple) else person

    #     person_annotations = interesting_annotations.filter(person__person_id = person_id).values('Xw', 'Yw', 'Zw')
    #     positions = np.array([[track['Xw'], track['Yw'], track['Zw']] for track in person_annotations])
    #     distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    #     distance_list.append(np.sum(distances))

    # print(f"Average distance traveled: {np.mean(distance_list)}")

    


    # person_counts.sort(key=lambda x:x[1], reverse=True)
    # # Convert to lists for plotting
    # person_ids, counts = zip(*person_counts)
    # print("Creating Plot")
    # # Create the plot
    # plt.figure(figsize=(12, 6))
    # # plt.bar(person_ids, counts)
    # plt.bar(range(len(counts)), counts)
    # plt.xlabel('Person ID')
    # plt.ylabel('Number of Annotations')
    # plt.title('Annotations per Person')
    # plt.grid(True)
    # plt.savefig('annotation_counts.png')
    # plt.close()


    # print(f"Average number of annotations per person {avg_annotations}")

    # annotations = Annotation.objects.filter(person__dataset__name=args.dataset, person__worker__workerID=args.worker, frame__in=frames)
    # annotation_data = annotations.values('frame__framed_id', 'Xw', 'Yw', 'Zw', 'person').groupby('person')
    # twod_annotations = Annotation2DView.objects.filter(annotation__in = annotations)

    # people = annotations.values('person')
    # print(f"There are {len(people)} people in the dataset")
    # print(f"There are {len(twod_annotations)} 2d annotations in the dataset")

    # track_durations = []
    # track_lengths = []
    # for person in tqdm(people, total = len(people), desc="Calculating Metrics..."):
    #     track_duration, track_length = get_track_duration_and_length(person)

    # print(f"The average track duration is {np.mean(track_durations)} frames.")
    # print(f"The average track distance is {np.mean(track_lengths)} metres.")

    
if __name__ == '__main__':
    main()
