# -*- coding: utf-8 -*-
from curses.textpad import rectangle
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib.auth import authenticate, login, logout
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect, HttpResponse, HttpResponseNotFound, FileResponse, Http404
from django.core import serializers
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.conf import settings
from .models import Worker, ValidationCode, MultiViewFrame, View, Annotation, Annotation2DView, Person, Dataset
from django.template import RequestContext
from django.http import JsonResponse
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
import re
import json
import os
import random as rand
from threading import Thread
from ipdb import set_trace
import random
import glob
import numpy as np
from gtm_hit.misc import geometry
from gtm_hit.misc.db import *
from gtm_hit.misc.serializer import *
from gtm_hit.misc.utils import convert_rect_to_dict, request_to_dict, process_action
from pprint import pprint
import uuid
# from gtm_hit.misc.invision.create_video import create_video as create_video_invision

def requestID(request):
    
    context = RequestContext(request).flatten()
    if request.method == "POST":
        if 'wID' in request.POST:
            workerID = request.POST['wID']
            pattern = re.compile("^[A-Z0-9]+$")
            if pattern.match(workerID):
                if 'datasetName' in request.POST:
                    dataset_name = request.POST['datasetName']
                    return redirect(f"/gtm_hit/{dataset_name}/{workerID}/processInit")
    return render(request, 'gtm_hit/requestID.html', context)

def processInit(request, dataset_name, workerID):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == -1:
            w.state = 0
            w.save()
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")

def index(request, workerID,dataset_name):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state != 0:
            return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
        return render(request, 'gtm_hit/index.html', {'workerID': workerID, **context, 'dset_name': dataset_name})

    except Worker.DoesNotExist:
        
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")

def processIndex(request, workerID,dataset_name):
    
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 0:
            w.state = 3
            w.save()
    except Worker.DoesNotExist:
        
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    return redirect(f"/gtm_hit/{dataset_name}/{workerID}")

def dispatch(request, dataset_name, workerID):
    
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        try:
            code = ValidationCode.objects.get(worker_id=w)
            stop = False
            i = 2
            while not stop:
                try:
                    w2 = Worker.objects.get(pk=workerID+str(i))
                    c2 = ValidationCode.objects.get(worker_id=w2)
                    i = i + 1
                except Worker.DoesNotExist:
                    stop = True
                except ValidationCode.DoesNotExist:
                    return redirect(f"/gtm_hit/{dataset_name}/{workerID+str(i)}")
            return redirect(f"/gtm_hit/{dataset_name}/{workerID+str(i)}")
        except ValidationCode.DoesNotExist:
            pass
    except Worker.DoesNotExist:
        w = registerWorker(workerID)
    
    dataset,_ = Dataset.objects.get_or_create(name=dataset_name)

    urlpath = "/gtm_hit/"+dataset_name+"/"+workerID+"/"

    state = w.state

    if state == 0:
        return redirect(urlpath+'index')
    elif state == 1:
        return redirect(urlpath+'frame')
    elif state == 2:
        return redirect(urlpath+'finish')
    elif state == 3:
        return redirect(urlpath+'tuto')
    elif state == -1:
        return redirect(urlpath+'processInit')
    
    return redirect(urlpath+'index')

def frame(request, dataset_name, workerID):
    context = RequestContext(request).flatten()

    try:
        w = Worker.objects.get(pk=workerID)
        if w.state != 1:
            return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
        if w.frameNB < 0:
            w.frameNB = settings.STARTFRAME
            w.save()
        frame_number = int(w.frameNB)
        nblabeled = w.frame_labeled
        try:
            dataset,_ = Dataset.objects.get_or_create(name=dataset_name)
        except Dataset.DoesNotExist:
            return HttpResponseNotFound("Dataset not found")

        frames_path = os.path.join('gtm_hit/static/gtm_hit/dset/', dataset_name, '/frames')
        

        # Create a dictionary of frame strings for each camera
        frame_strs = {}
        for cam in settings.CAMS:
            pattern = f"{frames_path}/{cam}/*_{frame_number}.jpg"
            matching_files = glob.glob(pattern)
            # print(matching_files)
            if matching_files:
                frame_strs[cam] = matching_files[0].split('/')[-1]
        # print(frame_strs)
        # context['cams'] = json.dumps(settings.CAMS)
        # print(context['cams'])

        return render(request, 'gtm_hit/frame.html', {
            'dset_name': dataset.name, 
            'frame_number': frame_number,
            'frame_strs': json.dumps(frame_strs),  # Pass as JSON string
            'frame_inc': settings.INCREMENT,
            'workerID': workerID,
            'cams': settings.CAMS,
            'frame_size': settings.FRAME_SIZES,
            'nb_cams': settings.NB_CAMS,
            'nblabeled': nblabeled,
            **context,
            "undistort": settings.UNDISTORTED_FRAMES
        })

    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


# def frame(request, dataset_name, workerID):
#     context = RequestContext(request).flatten()
#     try:
#         w = Worker.objects.get(pk=workerID)
#         if w.state != 1:
#             return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
#         if w.frameNB < 0:
#             w.frameNB = settings.STARTFRAME
#             w.save()
#         frame_number = w.frameNB
#         nblabeled = w.frame_labeled

#         try:
#             dataset,_ = Dataset.objects.get_or_create(name=dataset_name)
#         except Dataset.DoesNotExist:
#             return HttpResponseNotFound("Dataset not found")
#         frames_path = os.path.join('gtm_hit/static/gtm_hit/dset/'+dataset_name+'/frames')
        
#         #/static/gtm_hit/dset/${dset_name}/frames/${camName[i]}/${frame_str}
#         pattern = f"{frames_path}/*/*_{frame_number}.jpg"
#         print(pattern)
#         matching_files = glob.glob(pattern)
#         print('matching: ', matching_files)
#         frame_str = matching_files[0].split('/')[-1]
        
#         return render(request, 'gtm_hit/frame.html', {'dset_name': dataset.name, 'frame_number': frame_number, 'frame_str' : frame_str,  'frame_inc': settings.INCREMENT, 'workerID': workerID, 'cams': settings.CAMS, 'frame_size': settings.FRAME_SIZES, 'nb_cams': settings.NB_CAMS, 'nblabeled': nblabeled, **context, "undistort": settings.UNDISTORTED_FRAMES})
#     except Worker.DoesNotExist:
#         return redirect(f"/gtm_hit/{dataset_name}/{workerID}")

def processFrame(request, workerID,dataset_name):
    
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 1 and w.frame_labeled >= 500:
            w.state = 2
            timelist = w.getTimeList()
            timelist.append(timezone.now().isoformat())
            w.setTimeList(timelist)
            w.save()
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def finish(request, workerID,dataset_name):
    
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 2:
            validation_code = generate_code(w)
            startframe = w.frameNB - (w.frame_labeled*5)
            try:
                settings.UNLABELED.remove(startframe)
            except ValueError:
                pass
            return render(request, 'gtm_hit/finish.html', {'workerID': workerID, 'validation_code': validation_code, **context})
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'


def get_cuboids_2d(world_point, obj, new=False):
    rectangles = list()
    rect_id = str(int(world_point[0])) + "_" + str(int(world_point[1])
                                                   ) + "_" + uuid.uuid1().__str__().split("-")[0]
    # 

    if "object_size" in obj:
        object_size = obj["object_size"]
    else:
        object_size = [settings.HEIGHT, settings.RADIUS, settings.RADIUS]

    for cam_id in range(settings.NB_CAMS):
        # 
        # try:
            # check if world point is in the camera FOV
        if geometry.is_visible(world_point, settings.CAMS[cam_id], check_mesh=True):
            cuboid = geometry.get_cuboid_from_ground_world(
            world_point, settings.CALIBS[settings.CAMS[cam_id]], *object_size, obj.get("rotation_theta", 0))
            p1, p2 = geometry.get_bounding_box(cuboid)
        # except ValueError:
        else:
            cuboid = []
            p1 = [-1, -1]
            p2 = [-1, -1]
        rectangle_as_dict = convert_rect_to_dict(
            (*p1, *p2), cuboid, cam_id, rect_id, world_point, object_size, obj.get("rotation_theta", 0))
        if "person_id" in obj:
            rectangle_as_dict["personID"] = obj["person_id"]
        rectangles.append(rectangle_as_dict)

    return rectangles

def click(request):
    if is_ajax(request):
        # print("Click endpoint hit")
        # print("POST data:", request.POST)

        # try:
        x = int(float(request.POST['x']))
        y = int(float(request.POST['y']))
        obj = request_to_dict(request)
        cam = request.POST['canv'].replace("canv", "")
        # cam = int(re.findall(r'\d+', cam)[0]) - 1
        #
        worker_id = request.POST['workerID']
        dataset_name = request.POST['datasetName']
        # print(f"Cam: {cam}")
        if cam in settings.CAMS:

        # if 0 <= cam < settings.NB_CAMS:
            feet2d_h = np.array([[x], [y]])#, [1]])
            # print("2d: ", feet2d_h, "cam: ", cam, "calib: ", settings.CALIBS[settings.CAMS[cam]])
            if settings.FLAT_GROUND:
                calib = settings.CALIBS[cam]
                K0, R0, T0 = calib.K, calib.R, calib.T
                world_point = geometry.reproject_to_world_ground_batched(feet2d_h.T, K0, R0, T0, height=-0.301)
            else:
                world_point = geometry.project_2d_points_to_mesh(
                    feet2d_h, settings.CALIBS[cam], settings.MESH)#undistort=settings.UNDISTORTED_FRAMES)
            if "person_id" not in obj:
                obj["person_id"] = get_next_available_id(worker_id=worker_id,dataset_name=dataset_name)

            # print("World point:", world_point)
            rectangles = get_cuboids_2d(world_point[0], obj)
            # print("Rectangles:", rectangles)
            rect_json = json.dumps(rectangles)
            
            
            #
            return HttpResponse(rect_json, content_type="application/json")

        return HttpResponse("OK")


# def move(request):
#     if is_ajax(request):
#         try:
#             obj = request_to_dict(request)

#             Xw = obj["Xw"]
#             Yw = obj["Yw"]
#             Zw = obj["Zw"]

#             world_point = np.array([[Xw], [Yw], [Zw]])
#             if request.POST['data[dir]'] == "down":
#                 world_point = world_point + \
#                     np.array([[0], [-settings.STEPL], [0]])

#             elif request.POST['data[dir]'] == "up":
#                 world_point = world_point + \
#                     np.array([[0], [settings.STEPL], [0]])

#             elif request.POST['data[dir]'] == "right":
#                 world_point = world_point + \
#                     np.array([[settings.STEPL], [0], [0]])

#             elif request.POST['data[dir]'] == "left":
#                 world_point = world_point + \
#                     np.array([[-settings.STEPL], [0], [0]])

#             else:
#                 return HttpResponse("Error")

#             # 
#             next_rect = get_cuboids_2d(world_point, obj)

#             next_rect_json = json.dumps(next_rect)
#             # 
#             return HttpResponse(next_rect_json, content_type="application/json")

#         except KeyError:
#             return HttpResponse("Error")
#     return HttpResponse("Error")


def action(request):
    if is_ajax(request):
        try:
            obj = json.loads(request.POST["data"])

            obj = process_action(obj)
            Xw = obj["Xw"]
            Yw = obj["Yw"]
            Zw = obj["Zw"]

            
            world_point = np.array([[Xw], [Yw], [Zw]]).reshape(-1, 3)
            print("World point:", world_point[0].shape)
            if not settings.FLAT_GROUND:
                world_point = geometry.move_with_mesh_intersection(world_point)
            if world_point is None:
                return HttpResponse("Error")
            
            next_rect = get_cuboids_2d(world_point[0], obj)

            next_rect_json = json.dumps(next_rect)
            # 
            return HttpResponse(next_rect_json, content_type="application/json")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")


def save(request):
    return save_db(request)


def load(request):
    return load_db(request)

# def load_previous(request):
#     if is_ajax(request):
#         try:
#             frameID = request.POST['ID']
#             if not frameID:
#                 return HttpResponse("No frame ID provided")
            
#             wid = request.POST['workerID']
#             current_frame = int(float(frameID))
            
#             # Use regex pattern to match frame numbers
#             pattern = rf"{wid}_(\d+)\.json$"
#             label_dir = Path(f"./gtm_hit/static/gtm_hit/dset/{settings.DSETNAME}/labels/{wid}/")
            
#             # Find closest previous frame using regex
#             frames = []
#             for f in label_dir.glob("*.json"):
#                 if match := re.search(pattern, f.name):
#                     frame_num = int(match.group(1))
#                     if frame_num < current_frame:
#                         frames.append(frame_num)
            
#             if frames:
#                 closest = max(frames)
#                 frame = f"{closest:08d}"
#                 rect_json = read_save(frame, wid)
#                 return HttpResponse(rect_json, content_type="application/json")

#         except (FileNotFoundError, KeyError):
#             return HttpResponse("Error")
#     return HttpResponse("Error")

# def read_save(frameID, workerID):
#     path_pattern = f"./gtm_hit/static/gtm_hit/dset/{settings.DSETNAME}/labels/{workerID}/{workerID}_{frameID}.json"
#     with open(path_pattern, 'r') as loadFile:
#         annotations = json.load(loadFile)
#     return json.dumps(annotations)


def load_previous(request):
    if is_ajax(request):
        try:

            frameID = request.POST['ID']
            if not frameID:
                return HttpResponse("No frame ID provided")
            wid = request.POST['workerID']
            current_frame = int(frameID)
            closest = float('inf')
            diff = float('inf')

            for f in os.listdir("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/" + wid + "/"):
                if f.endswith(".json"):
                    nb_frame = int((f.split('.')[0]).split('_')[1])
                    if nb_frame < current_frame:
                        if current_frame - nb_frame < diff:
                            diff = current_frame - nb_frame
                            closest = nb_frame
            if closest != float('inf'):
                frame = "0" * (8 - len(str(closest))) + str(closest)
                rect_json = read_save(frame, wid)
                return HttpResponse(rect_json, content_type="application/json")
        except (FileNotFoundError, KeyError):
            return HttpResponse("Error")
    return HttpResponse("Error")


def read_save(frameID, workerID):
    # 
    filename = "./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME + \
        "/labels/" + workerID + "/" + workerID + "_" + frameID + '.json'
    with open(filename, 'r') as loadFile:
        annotations = json.load(loadFile)
    return json.dumps(annotations)

def changeframe(request):
    context = RequestContext(request)
    if is_ajax(request):
        try:
            wID = request.POST['workerID']
            order = request.POST['order']
            frame_number = request.POST['frameID']
            increment = request.POST['incr']

            worker = Worker.objects.get(pk=wID)
            timelist = worker.getTimeList()
            timelist.append(timezone.now().isoformat())
            worker.setTimeList(timelist)
            # print("Frame Number: ", frame_number)
            if order == "next":
                inc = int(increment)
            elif order == "prev":
                inc = -int(increment)
            elif order == 'first':
                inc = 0
            else:
                return HttpResponse(f"Requested frame: {frame_number} doesn't exist")
            
            new_frame_number = min(max(int(frame_number) + inc, 0), settings.NUM_FRAMES - 1)
            if order == 'first':
                new_frame_number = 0
            # print("new_frame_number: ", new_frame_number)
            # Get frame strings for each camera
            frames_path = os.path.join('gtm_hit/static/gtm_hit/dset/'+settings.DSETNAME+'/frames')
            frame_strs = {}
            for cam in settings.CAMS:
                pattern = f"{frames_path}/{cam}/*_{new_frame_number}.jpg"
                matching_files = glob.glob(pattern)
                if matching_files:
                    frame_strs[cam] = matching_files[0].split('/')[-1]
            # print(frame_strs)
            response = {
                'frame': str(new_frame_number),
                'nblabeled': worker.frame_labeled,
                'frame_strs': frame_strs
            }

            worker.frameNB = new_frame_number
            worker.frame_labeled = new_frame_number
            worker.save()

            return HttpResponse(json.dumps(response))

        except KeyError:
            return HttpResponse("Error")
    else:
        return HttpResponse("Error")

def get_rect(closest):
    rects = []
    for i in range(settings.NB_CAMS):
        rdic = {}
        rdic['rectangleID'] = closest
        if closest in settings.RECT[i]:
            a, b, c, d, ratio = settings.RECT[i][closest]
        else:
            a, b, c, d, ratio = 0, 0, 0, 0, 0
        rdic['x1'] = a
        rdic['y1'] = b
        rdic['x2'] = c
        rdic['y2'] = d
        rdic['cameraID'] = i
        rdic['ratio'] = ratio
        rdic['xMid'] = (a + c) // 2
        rects.append(rdic)
    return rects

def registerWorker(workerID):
    w = Worker()
    w.workerID = workerID
    w.frameNB = settings.STARTFRAME % settings.NBFRAMES
    settings.STARTFRAME = settings.STARTFRAME + 100*settings.INCREMENT
    w.save()
    return w


def updateWorker(workerID, state):
    w = Worker.objects.get(pk=workerID)


def generate_code(worker):
    try:
        code = ValidationCode.objects.get(worker_id=worker)
    except ValidationCode.DoesNotExist:
        random_code = int(16777215 * rand.random())
        random_code = "{0:0>8}".format(random_code)
        while(random_code in settings.VALIDATIONCODES):
            random_code = int(16777215 * rand.random())
            random_code = "{0:0>8}".format(random_code)
        settings.VALIDATIONCODES.append(random_code)
        code = ValidationCode()
        code.validationCode = random_code
        code.worker = worker
        code.save()
    return code.validationCode


def tuto(request, workerID,dataset_name):
    context = RequestContext(request).flatten()
    
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state != 3:
            return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
        return render(request, 'gtm_hit/tuto.html', {'workerID': workerID, 'dset_name':dataset_name, **context})

    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def processTuto(request, workerID,dataset_name):
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk=workerID)
        if w.state == 3:
            w.state = 1
            timelist = [timezone.now().isoformat()]
            w.setTimeList(timelist)
            w.save()
    except Worker.DoesNotExist:
        return redirect(f"/gtm_hit/{dataset_name}/{workerID}")
    return redirect(f"/gtm_hit/{dataset_name}/{workerID}")


def processFinish(request):
    context = RequestContext(request)
    if request.is_ajax():
        try:
            wID = request.POST['workerID']

            w = Worker.objects.get(pk=wID)
            startframe = w.frameNB - w.frame_labeled
            # delete_and_load(startframe)
            return HttpResponse("ok")
        except KeyError:
            return HttpResponse("Error")
    else:
        return HttpResponse("Error")


def delete_and_load(startframe):
    toload = settings.LASTLOADED + 10
    # 1. remove frames
    sframe = startframe
    # 2. copy next frames
    for i in range(10):
        rm_frame = "0" * (8 - len(str(sframe))) + str(sframe)
        cp_frame = "0" * (8 - len(str(toload))) + str(toload)
        sframe = sframe + 1
        toload = toload + 1
        for j in range(settings.NB_CAMS):
            command = os.system(
                "rm gtm_hit/static/gtm_hit/frames/" + settings.CAMS[j] + "/" + rm_frame + ".png")
            command = os.system("cp gtm_hit/static/gtm_hit/day_2/annotation_final/" +
                                settings.CAMS[j] + "/begin/" + cp_frame + ".png gtm_hit/static/gtm_hit/frames/" + settings.CAMS[j] + "/")

    settings.LASTLOADED = settings.LASTLOADED + 10


def save_db(request):
    #set_trace()
    
    if is_ajax(request) and request.method == 'POST':
        try:
            data = json.loads(request.POST['data'])
            frame_id = request.POST['ID']
            worker_id = request.POST['workerID']
            # Check if the frame exists or create a new frame object
            worker, _ = Worker.objects.get_or_create(workerID=worker_id)
            dataset_name = request.POST['datasetName']
            #set_trace()
            dataset,_ = Dataset.objects.get_or_create(name=dataset_name)
            frame, created = MultiViewFrame.objects.get_or_create(
                frame_id=frame_id, worker=worker,undistorted=settings.UNDISTORTED_FRAMES,dataset=dataset)
            
            #delete all annotations for this frame (if not single person save)
            if len(data)>1:
                Annotation.objects.filter(frame=frame).delete()

            # First create all Person objects in bulk
            people_to_create = [
                Person(person_id=annotation_data['personID'], worker=worker, dataset=dataset)
                for annotation_data in data
            ]
            Person.objects.bulk_create(people_to_create, ignore_conflicts=True)

            # Get all persons at once
            people = {p.person_id: p for p in Person.objects.filter(worker=worker, dataset=dataset)}

            # Create all annotations in bulk
            annotations_to_create = [
                Annotation(
                    person=people[annotation_data['personID']],
                    frame=frame,
                    rectangle_id=annotation_data['rectangleID'],
                    rotation_theta=annotation_data['rotation_theta'],
                    Xw=annotation_data['Xw'],
                    Yw=annotation_data['Yw'],
                    Zw=annotation_data['Zw'],
                    object_size_x=annotation_data['object_size'][0],
                    object_size_y=annotation_data['object_size'][1],
                    object_size_z=annotation_data['object_size'][2]
                )
                for annotation_data in data
            ]

            # Bulk create/update annotations
            Annotation.objects.bulk_create(
                annotations_to_create,
                update_conflicts=True,
                unique_fields=['person', 'frame'],
                update_fields=['rectangle_id', 'rotation_theta', 'Xw', 'Yw', 'Zw', 
                            'object_size_x', 'object_size_y', 'object_size_z']
            )

            # Bulk create 2D views
            save_2d_views_bulk(Annotation.objects.filter(frame=frame))

            # 
            # Iterate through each annotation in the data and create an annotation object for it
            # for annotation_data in data:
            #     person, _ = Person.objects.get_or_create(
            #         person_id=annotation_data['personID'],worker=worker,dataset=dataset)
            #     # Create a new annotation object for the given person and frame
            #     try:
            #         annotation = Annotation.objects.get(
            #             person=person, frame=frame)
            #         annotation.person = person
            #         annotation.frame = frame
            #         annotation.rectangle_id = annotation_data['rectangleID']
            #         annotation.rotation_theta = annotation_data['rotation_theta']
            #         annotation.Xw = annotation_data['Xw']
            #         annotation.Yw = annotation_data['Yw']
            #         annotation.Zw = annotation_data['Zw']
            #         annotation.object_size_x = annotation_data['object_size'][0]
            #         annotation.object_size_y = annotation_data['object_size'][1]
            #         annotation.object_size_z = annotation_data['object_size'][2]
            #     except Annotation.DoesNotExist:
            #         annotation = Annotation(
            #             person=person,
            #             frame=frame,
            #             rectangle_id=annotation_data['rectangleID'],
            #             rotation_theta=annotation_data['rotation_theta'],
            #             Xw=annotation_data['Xw'],
            #             Yw=annotation_data['Yw'],
            #             Zw=annotation_data['Zw'],
            #             object_size_x=annotation_data['object_size'][0],
            #             object_size_y=annotation_data['object_size'][1],
            #             object_size_z=annotation_data['object_size'][2],
            #         )

            #     # Save the annotation object to the database
            #     annotation.save()
            #     save_2d_views(annotation)

            # # Create the directory for the labels if it doesn't exist
            # labels_directory = os.path.join('./gtm_hit/static/gtm_hit/dset/', settings.DSETNAME, 'labels', worker_id)
            # os.makedirs(labels_directory, exist_ok=True)

            # # Serialize the views associated with the frame and save them to a JSON file
            # views = View.objects.filter(frame=frame).prefetch_related('annotation_set__twod_views')
            # serialized_views = serializers.serialize('json', views)
            # with open(os.path.join(labels_directory, f'{worker_id}_{frame_id}.json'), 'w') as f:
            #     f.write(serialized_views)

            return HttpResponse("Saved")

        except KeyError:
            return HttpResponse("Error")

    else:
        return HttpResponse("Error")


def load_db(request):
    print("Loading Database")
    if is_ajax(request):
        try:
            frame_id =request.POST['ID']
            if not frame_id:
                return HttpResponse("No frame ID provided")
            frame_id = int(frame_id)
            
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']

            # print('this is frame:', frame_id)
            # print('this is worker:', worker_id)
            # print('this is dataset:', dataset_name)
            frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES, dataset__name=dataset_name)
            # 
            retjson = []
            camviews = View.objects.all()

            retjson = serialize_frame_annotations(frame)
            # print(retjson)
            # for camview in camviews:
            #     print('serialising camview: ', camview, 'for frame: ', frame)
            #     #
            #     a2l = serialize_annotation2dviews(
            #         Annotation2DView.objects.filter(annotation__frame=frame, view=camview))
            #     # print("")
            #     # print('serialised camview: ', a2l)
            #     # print("")
            #     retjson.append(a2l)
            # # print(a2l)
            # a2l = list(Annotation2DView.objects.filter(annotation__frame=frame, view=View.objects.get(view_id=0)).values())
            # print(a2l[0])
            return HttpResponse(json.dumps(retjson), content_type="application/json")

            # Read the serialized views from the JSON file and deserialize them
            # labels_directory = os.path.join(
            #     './gtm_hit/static/gtm_hit/dset/', settings.DSETNAME, 'labels', worker_id)
            # with open(os.path.join(labels_directory, f'{worker_id}_{frame_id}.json'), 'r') as rect_json:
            #     # Deserialize the views from the serialized data
            #     views = serializers.deserialize('json', rect_json)

            # Iterate through each view and prefetch the related annotation and annotation2dview objects
            # for view in views:
            #     view.object = view.object.prefetch_related(
            #         'annotation_set__twod_views').select_related('frame')

            # # Serialize the views and send them as a response
            # serialized_views = serializers.serialize('json', views)
            # return HttpResponse(serialized_views, content_type="application/json")

        except (Person.DoesNotExist, MultiViewFrame.DoesNotExist, FileNotFoundError, KeyError):
            return HttpResponse("Error")

    return HttpResponse("Error")

def change_id(request):
    #set_trace()
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            new_person_id = int(float(request.POST['newPersonID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']

            frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)

            options = json.loads(request.POST['options'])
            success = change_annotation_id_propagate(person_id, new_person_id,frame,options)
            if success:
                return HttpResponse(JsonResponse({"message": "ID changed.","options":options}))
            else:
                return HttpResponse("Error")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")

def person_action(request):
    
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            worker_id = request.POST['workerID']
            options = json.loads(request.POST['options'])
            dataset_name = request.POST['datasetName']

            person = Person.objects.get(person_id=person_id,worker_id=worker_id,dataset__name=dataset_name)
            #
            try:
                if "mark" in options:
                    person.annotation_complete = options["mark"]
                    person.save()
                    return HttpResponse(JsonResponse({"message": "Person annotation complete."}))
                if "delete" in options:
                    if "delete" in options and options["delete"]:
                        person.delete()
                    return HttpResponse(JsonResponse({"message": "Person deleted."}))
            except Person.DoesNotExist:
                return HttpResponse("Error")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")

def tracklet(request):
    
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            try:
                frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
                person = Person.objects.get(person_id=person_id,worker_id=worker_id,dataset__name = dataset_name)
            except ValueError:
                HttpResponse("Error")
            multiview_tracklet = get_annotation2dviews_for_frame_and_person(
                frame, person)
            #
            return HttpResponse(json.dumps(multiview_tracklet), content_type="application/json")
        except Exception as e:
            print('Error', e)
            return HttpResponse("Error")


def interpolate(request):
    if is_ajax(request):
        # print(request.POST)
        try:
            #
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            #
            try:
                frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
                person = Person.objects.get(person_id=person_id,worker_id=worker_id, dataset__name=dataset_name)
                message = interpolate_until_next_annotation(frame=frame, person=person)
            except ValueError:
                message = "Error while interpolating "
                return HttpResponse("Error", status=500)
            return HttpResponse(json.dumps({"message":message}), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

def cp_prev_or_next_annotation(request):
    #set_trace()
    if is_ajax(request):
        try:
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            try:
                frame = MultiViewFrame.objects.get(frame_id=frame_id, worker_id=worker_id,undistorted=settings.UNDISTORTED_FRAMES,dataset__name=dataset_name)
                person = Person.objects.get(person_id=person_id,worker_id=worker_id, dataset__name=dataset_name)
                #delete person from current frame
                try:
                    annotation = Annotation.objects.get(person=person,frame=frame)
                    annotation.delete()
                except Annotation.DoesNotExist:
                    pass
                annotation = find_closest_annotations_to(person,frame,bidirectional=False)
                success = copy_annotation_to_frame(annotation, frame)
                if success:
                    return HttpResponse(JsonResponse({"message": "Annotation copied."}))
                else:
                    return HttpResponse("Error", status=500)
            except ValueError:
                HttpResponse("Error", status=500)
        except KeyError:
            return HttpResponse("Error",status=500)
        
def timeview(request):
    
    if is_ajax(request):
        # print('retrieving timeview')
        try:
            #
            worker_id = request.POST['workerID']
            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))
            view_id = int(float(request.POST['viewID']))
            dataset_name=request.POST['datasetName']
            # print('looking for frame: ', frame_id)
            # Calculate the range of frame_ids for 5 frames before and 5 frames after the given frame

            frame_id_start = max(0, frame_id - settings.TIMEWINDOW)
            frame_id_end =  min(frame_id + settings.TIMEWINDOW, settings.NUM_FRAMES)
            
            # Filter the Annotation2DView objects using the calculated frame range and the Person object
            annotation2dviews = Annotation2DView.objects.filter(
                annotation__frame__frame_id__gte=frame_id_start,
                annotation__frame__frame_id__lte=frame_id_end,
                annotation__frame__worker__workerID=worker_id,
                annotation__frame__dataset__name=dataset_name,
                annotation__person__person_id=person_id,
                view__view_id = view_id,
                annotation__frame__undistorted = settings.UNDISTORTED_FRAMES
            ).order_by('annotation__frame__frame_id')
            # print("annotation2dviews: ", annotation2dviews)
            timeviews =  serialize_annotation2dviews(annotation2dviews)
            # 
            # print("timeviews: ", timeviews)
            return HttpResponse(json.dumps(timeviews), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

import numpy as np

def reset_ac_flag(request):
    set_trace()
    if is_ajax(request):
        try:
            worker_id = request.POST['workerID']
            dataset_name = request.POST['datasetName']
            for person in Person.objects.filter(worker_id=worker_id,dataset__name=dataset_name):
                person.annotation_complete = False
                person.save()
            return HttpResponse(json.dumps({"message":"ok"}), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")

def create_video(request):
    print("This Functionality is removed for testing purposes")
    return HttpResponse("This Functionality is removed for testing purposes")
#     if is_ajax(request):
#         try:
#             dataset_name = request.POST['datasetName']
#             worker_id = request.POST['workerID']
#             create_video_invision(f"{worker_id}.mp4",15,dataset_name,worker_id)
#             return HttpResponse(json.dumps({"message":"ok"}), content_type="application/json")
#         except KeyError:
#             return HttpResponse("Error")



def serve_frame(request):
    if is_ajax(request):
        # try:
        frame_number = int(float(request.POST['frame_number']))
        camera_name = int(request.POST['camera_name'])
        # print(camera_name)
        camera_name = settings.CAMS[camera_name]
        frames_path = os.path.join('gtm_hit/static/gtm_hit/dset/'+settings.DSETNAME+'/frames', camera_name)
        
        # os.path.join(settings.DSETPATH,'frames', camera_name)
        pattern = f"{frames_path}/*_{frame_number}.jpg"
        # print(pattern)
        matching_files = glob.glob(pattern)
        # print(matching_files)
        if matching_files:
            response = {
            'frame_string': '/'+ os.path.join(*matching_files[0].split('/')[-7:])
            }
            # print("Timeview: ", response)

            return HttpResponse(json.dumps(response))
        
        else:
            print(f"No frame found matching pattern for camera {camera_name} and frame {frame_number}")
            return HttpResponse(f"No frame found matching pattern for camera {camera_name} and frame {frame_number}")
        # except:
        #     print(f"No frame found matching pattern for camera {camera_name} and frame {frame_number}")
        #     raise Http404(f"No frame found matching pattern for camera {camera_name} and frame {frame_number}")
    # print("Error")
    return HttpResponse("Error")


from django.http import JsonResponse
import json
# from misc.geometry import Trajectory, merge_trajectory_group


# def alternate_merge(request):
#     if is_ajax(request):
#         try:
#             with transaction.atomic():
#                 person_ids = request.POST['personIDs']
#                 person_ids = list(map(int, person_ids.split(',')))
#                 dataset_name = request.POST['datasetName']
#                 worker_id = request.POST['workerID']
#                 merging_strategy = request.POST['mergingStrategy']
                
#                 # Create new Person instance for merged track
#                 worker = Worker.objects.get(workerID=worker_id)
#                 dataset = Dataset.objects.get(name=dataset_name)
#                 merged_person = Person.objects.create(
#                     person_id=max(Person.objects.all().values_list('person_id', flat=True)) + 1,
#                     worker=worker,
#                     dataset=dataset

#                 )

#                 annotations = Annotation.objects.filter(person__person_id__in=person_ids,
#                                                         person__worker=worker,
#                                                         person__dataset=dataset)
                

#                 merged_annotations = []
#                 for frame_number in range(settings.FRAME_START, settings.FRAME_END + 1):
#                     frame_annotations = annotations.filter(frame__frame_id=frame_number)
#                     coords_world = []
#                     for annotation in frame_annotations:
#                         coords_world.append([annotation.Xw, annotation.Yw, annotation.Zw])

#                     if coords_world:
#                         if merging_strategy == 'mean':
#                             coords_world = np.array(coords_world)
#                             coords_world = np.mean(coords_world, axis=0)
#                             # coords_world = coords_world.tolist()
#                             avg_Xw = coords_world[0]
#                             avg_Yw = coords_world[1]
#                             avg_Zw = coords_world[2]
#                         elif merging_strategy == 'camera_mean_top':

#                         elif merging_strategy == 'camera_mean':
                    

#                     if frame_ann_1 is not None or frame_ann_2 is not None:
#                         print("frame_ann_1: ", frame_ann_1)
#                         print("frame_ann_2: ", frame_ann_2)

                    
#                     if frame_ann_1 and frame_ann_2:
#                         pos_1 = np.array([frame_ann_1.Xw, frame_ann_1.Yw, frame_ann_1.Zw])
#                         pos_2 = np.array([frame_ann_2.Xw, frame_ann_2.Yw, frame_ann_2.Zw])
#                         avg_pos = (pos_1 + pos_2) / 2
#                     elif frame_ann_1:
#                         avg_pos = np.array([frame_ann_1.Xw, frame_ann_1.Yw, frame_ann_1.Zw])
#                     elif frame_ann_2:
#                         # If only person_id2 exists for the frame, use its position
#                         avg_pos = np.array([frame_ann_2.Xw, frame_ann_2.Yw, frame_ann_2.Zw])
#                     else:
#                         continue  # Skip if neither has an annotation for this frame
#                     avg_Xw = avg_pos[0]
#                     avg_Yw = avg_pos[1]
#                     avg_Zw = avg_pos[2]
#                     # Create a new Annotation for the merged person ID
#                     merged_annotation = Annotation(
#                             person=merged_person,
#                             frame=MultiViewFrame.objects.get(frame_id=frame_number, dataset=dataset, worker=worker),
#                             rectangle_id=uuid.uuid4().__str__().split("-")[-1],
#                             rotation_theta=0,
#                             Xw=avg_Xw,
#                             Yw=avg_Yw,
#                             Zw=avg_Zw,
#                             object_size_x=1.7,
#                             object_size_y=0.6,
#                             object_size_z=0.6,
#                             creation_method="merged_scout_tracks"
#                         )

#                     merged_annotations.append(merged_annotation)
                
                
#                 # Delete old data BEFORE creating new
#                 Annotation2DView.objects.filter(annotation__person__in=[person_id1, person_id2]).delete()
#                 Annotation.objects.filter(person__in=[person_id1, person_id2]).delete()
#                 Person.objects.filter(person_id__in=[person_id1, person_id2]).delete()
            
                
#                 # Bulk create new data
#                 Annotation.objects.bulk_create(merged_annotations)
#                 print(merged_annotations)
#                 save_2d_views_bulk(merged_annotations)

#                 return JsonResponse({"message": "ok"})
            
#         except Exception as e:
#             print("Exception:", e)
#             return JsonResponse({"message": "Error", "error": str(e)}, status=500)
#     return JsonResponse({"message": "Error"}, status=400)

def merge(request):
    if is_ajax(request):
        try:
            with transaction.atomic():
                person_id1 = int(float(request.POST['personID1']))
                person_id2 = int(float(request.POST['personID2']))
                dataset_name = request.POST['datasetName']
                worker_id = request.POST['workerID']
                
                # Create new Person instance for merged track
                worker = Worker.objects.get(workerID=worker_id)
                dataset = Dataset.objects.get(name=dataset_name)
                merged_person = Person.objects.create(
                    person_id=max(Person.objects.all().values_list('person_id', flat=True)) + 1,
                    worker=worker,
                    dataset=dataset

                )
                                # Get all frames at once
                            # Get all frames in the correct range
                frames = MultiViewFrame.objects.filter(
                    frame_id__range=(settings.FRAME_START, settings.FRAME_END),
                    dataset=dataset,
                    worker=worker
                )

                # Get all annotations for both persons across all frames
                annotations = Annotation.objects.filter(
                    person__person_id__in=[person_id1, person_id2],
                    person__worker=worker,
                    person__dataset=dataset,
                    frame__frame_id__range=(settings.FRAME_START, settings.FRAME_END)
                ).select_related('frame')

                print(len(annotations))


                # Create lookup dictionary for all frames
                annotations_by_frame = {}
                for ann in annotations:
                    frame_id = ann.frame.frame_id
                    print(frame_id)
                    if frame_id not in annotations_by_frame:
                        annotations_by_frame[frame_id] = []
                    annotations_by_frame[frame_id].append(ann)

                # Print the available frame numbers
                print("Available frames:", sorted(annotations_by_frame.keys()))

                # Create merged annotations for all frames where we have annotations
                merged_annotations = []
                for frame_number in sorted(annotations_by_frame.keys()):
                    frame_anns = annotations_by_frame.get(frame_number, [])
                    print(frame_anns)
                    if not frame_anns:
                        continue
                    
                    positions = np.array([[ann.Xw, ann.Yw, ann.Zw] for ann in frame_anns])
                    avg_pos = positions.mean(axis=0) if len(positions) > 1 else positions[0]
                    
                    frame = frames.get(frame_id=frame_number)
                    merged_annotations.append(
                        Annotation(
                            person=merged_person,
                            frame=frame,
                            rectangle_id=uuid.uuid4().__str__().split("-")[-1],
                            rotation_theta=0,
                            Xw=avg_pos[0],
                            Yw=avg_pos[1],
                            Zw=avg_pos[2],
                            object_size_x=1.7,
                            object_size_y=0.6,
                            object_size_z=0.6,
                            creation_method="merged_scout_tracks"
                        )
                    )



                
                
                # Delete old data BEFORE creating new
                Annotation2DView.objects.filter(annotation__person__in=[person_id1, person_id2]).delete()
                Annotation.objects.filter(person__in=[person_id1, person_id2]).delete()
                Person.objects.filter(person_id__in=[person_id1, person_id2]).delete()
            
                
                # Bulk create new data
                Annotation.objects.bulk_create(merged_annotations)
                merged_annotations = Annotation.objects.filter(person=merged_person)
                save_2d_views_bulk(merged_annotations)

                merge_history_file = 'merge_history.json'

                # Load existing history or create new
                if os.path.exists(merge_history_file):
                    with open(merge_history_file, 'r') as f:
                        merge_history = json.load(f)
                else:
                    merge_history = {}

                merge_history[merged_person.person_id] = [person_id1, person_id2]
                with open(merge_history_file, 'w') as f:
                    json.dump(merge_history, f, indent=4)

                return JsonResponse({"message": "ok"})
            
        except Exception as e:
            print("Exception:", e)
            return JsonResponse({"message": "Error", "error": str(e)}, status=500)
    return JsonResponse({"message": "Error"}, status=400)

# def merge(request):
#     if is_ajax(request):
#         try:
#             with transaction.atomic():
#                 person_id1 = int(float(request.POST['personID1']))
#                 person_id2 = int(float(request.POST['personID2']))
#                 dataset_name = request.POST['datasetName']
#                 worker_id = request.POST['workerID']
                
#                 # Create new Person instance for merged track
#                 worker = Worker.objects.get(workerID=worker_id)
#                 dataset = Dataset.objects.get(name=dataset_name)
#                 merged_person = Person.objects.create(
#                     person_id=max(Person.objects.all().values_list('person_id', flat=True)) + 1,
#                     worker=worker,
#                     dataset=dataset

#                 )
#                 # Retrieve Annotation2DView and Annotation objects
#                 annotation_1 = Annotation.objects.filter(person__person_id=person_id1, 
#                                                          person__worker=worker, 
#                                                          person__dataset=dataset)
#                 annotation_2 = Annotation.objects.filter(person__person_id=person_id2,
#                                                          person__worker=worker, 
#                                                          person__dataset=dataset)

                                                         

#                 merged_annotations = []
#                 for frame_number in range(settings.FRAME_START, settings.FRAME_END + 1):
#                     frame_ann_1 = annotation_1.filter(
#                                         frame__frame_id=frame_number,
#                                         person__worker=worker,
#                                         person__dataset=dataset
#                                     ).first()
#                     frame_ann_2 = annotation_2.filter(
#                                         frame__frame_id=frame_number,
#                                         person__worker=worker,
#                                         person__dataset=dataset
#                                     ).first()
#                     if frame_ann_1 is not None or frame_ann_2 is not None:
#                         print("frame_ann_1: ", frame_ann_1)
#                         print("frame_ann_2: ", frame_ann_2)

                    
#                     if frame_ann_1 and frame_ann_2:
#                         pos_1 = np.array([frame_ann_1.Xw, frame_ann_1.Yw, frame_ann_1.Zw])
#                         pos_2 = np.array([frame_ann_2.Xw, frame_ann_2.Yw, frame_ann_2.Zw])
#                         avg_pos = (pos_1 + pos_2) / 2
#                     elif frame_ann_1:
#                         avg_pos = np.array([frame_ann_1.Xw, frame_ann_1.Yw, frame_ann_1.Zw])
#                     elif frame_ann_2:
#                         # If only person_id2 exists for the frame, use its position
#                         avg_pos = np.array([frame_ann_2.Xw, frame_ann_2.Yw, frame_ann_2.Zw])
#                     else:
#                         continue  # Skip if neither has an annotation for this frame
#                     avg_Xw = avg_pos[0]
#                     avg_Yw = avg_pos[1]
#                     avg_Zw = avg_pos[2]
#                     # Create a new Annotation for the merged person ID
#                     merged_annotation = Annotation(
#                             person=merged_person,
#                             frame=MultiViewFrame.objects.get(frame_id=frame_number, dataset=dataset, worker=worker),
#                             rectangle_id=uuid.uuid4().__str__().split("-")[-1],
#                             rotation_theta=0,
#                             Xw=avg_Xw,
#                             Yw=avg_Yw,
#                             Zw=avg_Zw,
#                             object_size_x=1.7,
#                             object_size_y=0.6,
#                             object_size_z=0.6,
#                             creation_method="merged_scout_tracks"
#                         )

#                     merged_annotations.append(merged_annotation)
                
                
#                 # Delete old data BEFORE creating new
#                 Annotation2DView.objects.filter(annotation__person__in=[person_id1, person_id2]).delete()
#                 Annotation.objects.filter(person__in=[person_id1, person_id2]).delete()
#                 Person.objects.filter(person_id__in=[person_id1, person_id2]).delete()
            
                
#                 # Bulk create new data
#                 Annotation.objects.bulk_create(merged_annotations)
#                 print(merged_annotations)
#                 save_2d_views_bulk(merged_annotations)

#                 return JsonResponse({"message": "ok"})
            
#         except Exception as e:
#             print("Exception:", e)
#             return JsonResponse({"message": "Error", "error": str(e)}, status=500)
#     return JsonResponse({"message": "Error"}, status=400)

# def merge(request):
#     if is_ajax(request):
#         try:
#             # Parse request data
#             person_id1 = int(float(request.POST['personID1']))
#             person_id2 = int(float(request.POST['personID2']))
#             dataset_name = request.POST['datasetName']
#             worker_id = request.POST['workerID']
            
#             # Generate new person ID for the merged track
#             merged_person_id = max(Person.objects.all().values_list('id', flat=True)) + 1
#             print(f"Merging trajectories for persons {person_id1} and {person_id2} into trajectory {merged_person_id}")

#             # Retrieve Annotation2DView and Annotation objects
#             annotation_2d_1 = Annotation2DView.objects.filter(annotation__person=person_id1)
#             annotation_2d_2 = Annotation2DView.objects.filter(annotation__person=person_id2)
#             annotation_1 = Annotation.objects.filter(person=person_id1)
#             annotation_2 = Annotation.objects.filter(person=person_id2)


#             # Step 1: Create new Annotation objects with averaged positions
#             merged_annotations = []
#             for frame_number in range(settings.FRAME_START, settings.FRAME_END + 1):
#                 frame_ann_1 = annotation_1.filter(frame_id=frame_number).first()
#                 frame_ann_2 = annotation_2.filter(frame_id=frame_number).first()

                
#                 if frame_ann_1 and frame_ann_2:
#                     print
#                     pos_1 = np.array([frame_ann_1.Xw, frame_ann_1.Yw, frame_ann_1.Zw])
#                     pos_2 = np.array([frame_ann_2.Xw, frame_ann_2.Yw, frame_ann_2.Zw])
#                     avg_pos = (pos_1 + pos_2) / 2
#                 elif frame_ann_1:
#                     avg_pos = np.array([frame_ann_1.Xw, frame_ann_1.Yw, frame_ann_1.Zw])
#                 elif frame_ann_2:
#                     # If only person_id2 exists for the frame, use its position
#                     avg_pos = np.array([frame_ann_2.Xw, frame_ann_2.Yw, frame_ann_2.Zw])
#                 else:
#                     continue  # Skip if neither has an annotation for this frame
#                 avg_Xw = avg_pos[0]
#                 avg_Yw = avg_pos[1]
#                 avg_Zw = avg_pos[2]
#                 # Create a new Annotation for the merged person ID

#                 merged_annotation = Annotation(
#                         person=merged_person_id,
#                         frame_id=frame_number,
#                         rectangle_id=uuid.uuid4().__str__().split("-")[-1],
#                         rotation_theta=0,
#                         Xw=avg_Xw,
#                         Yw=avg_Yw,
#                         Zw=avg_Zw,
#                         object_size_x=1.7,
#                         object_size_y=0.6,
#                         object_size_z=0.6,
#                         creation_method="merged_scout_tracks"
#                     )

#                 merged_annotations.append(merged_annotation)
            
#             # Bulk create the new merged Annotations
#             Annotation.objects.bulk_create(merged_annotations)

#             # Step 2: Delete old Annotation2DView objects for the two IDs
#             # annotation_2d_1.delete()
#             # annotation_2d_2.delete()

#             # Step 3: Create new Annotation2DView objects for the merged track
#             save_2d_views_bulk(merged_annotations)
#             # for annotation in merged_annotations:
#             #     save_2d_views(annotation)

#             # Step 4: Delete old Annotation objects for the two IDs
#             Person.objects.filter(id=person_id1).delete()
#             Person.objects.filter(id=person_id2).delete()
#             # annotation_1.delete()
#             # annotation_2.delete()

#             # Respond with success
#             return JsonResponse({"message": "ok"})
        
#         except Exception as e:
#             print("Exception:", e)
#             return JsonResponse({"message": "Error", "error": str(e)}, status=500)
#     return JsonResponse({"message": "Error"}, status=400)

