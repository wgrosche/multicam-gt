# -*- coding: utf-8 -*-
from curses.textpad import rectangle
from django.shortcuts import get_object_or_404,render, redirect
from django.contrib.auth import authenticate, login, logout
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect, HttpResponse
from django.core import serializers
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.conf import settings
from .models import Worker, ValidationCode,MultiViewFrame, View, Annotation, Annotation2DView, Person
from django.template import RequestContext
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import re
import json
import os
import random as rand
from threading import Thread
from ipdb import set_trace
import random
import numpy as np
from gtm_hit.misc import geometry
from gtm_hit.misc.utils import convert_rect_to_dict, request_to_dict, process_action
from pprint import pprint
import uuid

def requestID(request):
    context = RequestContext(request).flatten()
    if request.method == "POST":
        if 'wID' in request.POST:
            workerID = request.POST['wID']
            pattern = re.compile("^[A-Z0-9]+$")
            if pattern.match(workerID):
                return redirect("/gtm_hit/"+workerID+"/processInit")
    return render(request, 'gtm_hit/requestID.html',context)

def processInit(request, workerID):
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state == -1:
            w.state = 0
            w.save()
        return redirect("/gtm_hit/"+workerID)
    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)

def index(request,workerID):
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state != 0:
            return redirect("/gtm_hit/"+workerID)
        return render(request, 'gtm_hit/index.html',{'workerID' : workerID, **context})

    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)

def processIndex(request, workerID):
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state == 0:
            w.state = 3
            w.save()
    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)
    return redirect("/gtm_hit/"+workerID)

def dispatch(request,workerID):
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk = workerID)
        try:
            code = ValidationCode.objects.get(worker_id = w)
            stop = False
            i = 2
            while not stop:
                try:
                    w2 = Worker.objects.get(pk = workerID+str(i))
                    c2 = ValidationCode.objects.get(worker_id = w2)
                    i = i + 1
                except Worker.DoesNotExist:
                    stop = True
                except ValidationCode.DoesNotExist:
                    return redirect("/gtm_hit/"+workerID+str(i))
            return redirect("/gtm_hit/"+workerID+str(i))
        except ValidationCode.DoesNotExist:
            pass
    except Worker.DoesNotExist:
        w = registerWorker(workerID)

    state = w.state
    if state == 0:
        return redirect(workerID+'/index')
        #return render(request, 'gtm_hit/frame.html',{'frame_number': frame_number, 'workerID' : workerID},context)
    elif state == 1:
        return redirect(workerID+'/frame')
#        return render(request, 'gtm_hit/finish.html',{'workerID' : workerID, 'validation_code' : validation_code},context)
    elif state == 2:
        return redirect(workerID+'/finish')
#        return render(request, 'gtm_hit/finish.html',{'workerID' : workerID, 'validation_code' : validation_code},context)
    elif state == 3:
        return redirect(workerID+'/tuto')
    elif state == -1:
        return redirect(workerID+'/processInit')
        #return render(request, 'gtm_hit/index.html',{'workerID' : workerID},context)
    else:
        return redirect(workerID+'/index')
        #return render(request, 'gtm_hit/index.html',{'workerID' : workerID},context)

def frame(request,workerID):
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state != 1:
            return redirect("/gtm_hit/"+workerID)
        if w.frameNB < 0:
            w.frameNB = settings.STARTFRAME
            w.save()
        frame_number = w.frameNB
        nblabeled = w.frame_labeled
        return render(request, 'gtm_hit/frame.html',{'dset_name':settings.DSETNAME, 'frame_number': frame_number, 'frame_inc':settings.INCREMENT, 'workerID': workerID,'cams': settings.CAMS, 'frame_size':settings.FRAME_SIZES, 'nb_cams':settings.NB_CAMS, 'nblabeled' : nblabeled, **context})
    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)

def processFrame(request,workerID):
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state == 1 and w.frame_labeled >= 500:
            w.state = 2
            timelist = w.getTimeList()
            timelist.append(timezone.now().isoformat())
            w.setTimeList(timelist)
            w.save()
        return redirect("/gtm_hit/"+workerID)
    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)

def finish(request,workerID):
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state == 2:
            validation_code = generate_code(w)
            startframe = w.frameNB - (w.frame_labeled*5)
            try:
                settings.UNLABELED.remove(startframe)
            except ValueError:
                pass
            return render(request, 'gtm_hit/finish.html',{'workerID': workerID, 'validation_code': validation_code, **context})
    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)
    return redirect("/gtm_hit/"+workerID)

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

def get_cuboids_2d(world_point,obj):
    rectangles = list()
    rect_id = str(int(world_point[0])) + "_" + str(int(world_point[1])) + "_" +uuid.uuid1().__str__().split("-")[0]
    ###set_trace()

    if "object_size" in obj:
        object_size = obj["object_size"]
    else: 
        object_size = [settings.HEIGHT, settings.RADIUS, settings.RADIUS]

    for cam_id in range(settings.NB_CAMS):
        ###set_trace()
        cuboid = geometry.get_cuboid_from_ground_world(world_point, settings.CALIBS[cam_id], *object_size, obj.get("rotation_theta",0))
        p1,p2 = geometry.get_bounding_box(cuboid)
        rectangle_as_dict = convert_rect_to_dict((*p1,*p2),cuboid, cam_id, rect_id, world_point,object_size,obj.get("rotation_theta",0))
        rectangles.append(rectangle_as_dict)    

    return rectangles

def click(request):
    if is_ajax(request):
        # try:
        x = int(float(request.POST['x']))
        y = int(float(request.POST['y']))
        obj = request_to_dict(request)
        cam = request.POST['canv']
        cam = int(re.findall('\d+',cam)[0]) - 1
        if 0 <= cam < settings.NB_CAMS:
            feet2d_h = np.array([[x], [y], [1]])
            ###set_trace()
            world_point = geometry.reproject_to_world_ground(feet2d_h, settings.CALIBS[cam].K, settings.CALIBS[cam].R, settings.CALIBS[cam].T)

            rectangles = get_cuboids_2d(world_point,obj)

            rect_json = json.dumps(rectangles)
            return HttpResponse(rect_json,content_type="application/json")

        return HttpResponse("OK")
    
def move(request):
    if is_ajax(request):
        try:
            obj = request_to_dict(request)

            Xw = obj["Xw"]
            Yw = obj["Yw"]
            Zw = obj["Zw"]
            
            world_point = np.array([[Xw], [Yw], [Zw]])  
            if request.POST['data[dir]'] == "down":
                world_point = world_point + np.array([[0], [-settings.STEPL], [0]])
                
            elif request.POST['data[dir]'] == "up":
                world_point = world_point + np.array([[0], [settings.STEPL], [0]])
                
            elif request.POST['data[dir]'] == "right":
                world_point = world_point + np.array([[settings.STEPL], [0], [0]])

            elif request.POST['data[dir]'] == "left":
                world_point = world_point + np.array([[-settings.STEPL], [0], [0]])

            else:
                return HttpResponse("Error")
            
            ###set_trace()
            next_rect = get_cuboids_2d(world_point,obj)

            next_rect_json = json.dumps(next_rect)
            ###set_trace()
            return HttpResponse(next_rect_json,content_type="application/json")

        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")

def action(request):
    ###set_trace()
    ###set_trace()
    if is_ajax(request):
        try:
            
            obj = json.loads(request.POST["data"])
           
            obj["object_size"] =obj["object_size"]#[::-1]
            
            obj = process_action(obj)
            Xw = obj["Xw"]
            Yw = obj["Yw"]
            Zw = obj["Zw"]
            
            world_point = np.array([[Xw], [Yw], [Zw]])
            next_rect = get_cuboids_2d(world_point,obj)

            next_rect_json = json.dumps(next_rect)
            ###set_trace()
            return HttpResponse(next_rect_json,content_type="application/json")
        except KeyError:
            return HttpResponse("Error")
    return HttpResponse("Error")

def save(request):
    return save_db(request)
    ###set_trace()
    if is_ajax(request):
        try:
            ###set_trace()
            data = json.loads(request.POST['data'])
            frameID = request.POST['ID']
            wid = request.POST['workerID']
            # annotations = []
            # cols = ["rectID","personID","modified","Xw","Yw","Zw"]#,"a1","b1","c1","d1","a2","b2","c2","d2","a3","b3","c3","d3","a4","b4","c4","d4","a5","b5","c5","d5","a6","b6","c6","d6","a7","b7","c7","d7"]
            # for i in range(settings.NB_CAMS):
            #     cols += [f"a{i+1}", f"b{i+1}", f"c{i+1}", f"d{i+1}"]
            # annotations.append(cols)
            # for r in data:
            #     row = data[r]
            #     row.insert(0,r)
            #     annotations.append(row)
            ###set_trace()
            if not os.path.exists("./gtm_hit/labels/"+settings.DSETNAME+"/"+ wid +"/"):
                    os.makedirs("./gtm_hit/labels/"+settings.DSETNAME+"/"+ wid +"/")
            with open("./gtm_hit/labels/"+settings.DSETNAME+"/"+ wid +"/"+ wid + "_" + frameID + '.json', 'w') as outFile:
                #create dir if not exist
                json.dump(data, outFile, sort_keys=True, indent=4, separators=(',', ': '))

            if not os.path.exists("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/"+ wid +"/"):
                    os.makedirs("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/"+ wid +"/")
            with open("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/"+ wid +"/"+ wid+ "_" + frameID + '.json', 'w') as outFile:
                json.dump(data, outFile, sort_keys=True, indent=4, separators=(',', ': '))
            return HttpResponse("Saved")
        except KeyError:
            return HttpResponse("Error")
    else:
        return("Error")

def load(request):
    return load_db(request)
    if is_ajax(request):
        try:
            frameID = request.POST['ID']
            wid = request.POST['workerID']
            rect_json = read_save(frameID,wid)
            return HttpResponse(rect_json,content_type="application/json")
        except (FileNotFoundError, KeyError):
            return HttpResponse("Error")
    return HttpResponse("Error")

def load_previous(request):
    if is_ajax(request):
        try:

            frameID = request.POST['ID']
            wid = request.POST['workerID']
            current_frame = int(frameID)
            closest = float('inf')
            diff = float('inf')

            for f in os.listdir("./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/"+ wid +"/"):
                if f.endswith(".json"):
                    nb_frame = int((f.split('.')[0]).split('_')[1])
                    if nb_frame < current_frame:
                        if current_frame - nb_frame < diff:
                            diff = current_frame - nb_frame
                            closest = nb_frame
            if closest != float('inf'):
                frame = "0" * (8 - len(str(closest))) + str(closest)
                rect_json = read_save(frame,wid)
                return HttpResponse(rect_json,content_type="application/json")
        except (FileNotFoundError, KeyError):
            return HttpResponse("Error")
    return HttpResponse("Error")

def read_save(frameID,workerID):
    ###set_trace()
    filename = "./gtm_hit/static/gtm_hit/dset/"+settings.DSETNAME+"/labels/"+ workerID +"/"+ workerID + "_" + frameID + '.json'
    with open(filename,'r') as loadFile:
        annotations = json.load(loadFile)
    return json.dumps(annotations)

def changeframe(request):
    context = RequestContext(request)
    ##set_trace()
    if is_ajax(request):
        frame = 0
        try:
            wID = request.POST['workerID']
            order = request.POST['order']
            frame_number = request.POST['frameID']
            increment = request.POST['incr']

            worker = Worker.objects.get(pk = wID)

            ###set_trace()
            worker.increaseFrame(1)
            worker.save()
            timelist = worker.getTimeList()
            timelist.append(timezone.now().isoformat())
            worker.setTimeList(timelist)
            #validation_code = generate_code()
            #return render(request, 'gtm_hit/finish.html',{'workerID' : wID, 'validation_code': validation_code},context)
            if order == "next":
                frame = int(frame_number) + int(increment)
            elif order == "prev" and (int(frame_number) - int(increment)) >= 0:
                frame = int(frame_number) - int(increment)
            else:
                return HttpResponse("Requested frame not existing")
            frame = "0" * (8 - len(str(frame))) + str(frame)
            response = {}
            response['frame'] = frame
            response['nblabeled'] = worker.frame_labeled
            worker.frameNB = frame
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
            a,b,c,d,ratio = settings.RECT[i][closest]
        else:
            a,b,c,d,ratio = 0,0,0,0,0
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
    w = Worker.objects.get(pk = workerID)

def generate_code(worker):
    try:
        code = ValidationCode.objects.get(worker_id = worker)
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

def tuto(request,workerID):
    context = RequestContext(request).flatten()
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state != 3:
            return redirect("/gtm_hit/"+workerID)
        return render(request, 'gtm_hit/tuto.html',{'workerID' : workerID, **context})

    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)

def processTuto(request, workerID):
    context = RequestContext(request)
    try:
        w = Worker.objects.get(pk = workerID)
        if w.state == 3:
            w.state = 1
            timelist = [timezone.now().isoformat()]
            w.setTimeList(timelist)
            w.save()
    except Worker.DoesNotExist:
        return redirect("/gtm_hit/"+workerID)
    return redirect("/gtm_hit/"+workerID)

def processFinish(request):
    context = RequestContext(request)
    if request.is_ajax():
        try:
            wID = request.POST['workerID']

            w = Worker.objects.get(pk = wID)
            startframe = w.frameNB - w.frame_labeled
            #delete_and_load(startframe)
            return HttpResponse("ok")
        except KeyError:
            return HttpResponse("Error")
    else:
        return HttpResponse("Error")



def delete_and_load(startframe):
    toload = settings.LASTLOADED + 10
     #1. remove frames
    sframe = startframe
     #2. copy next frames
    for i in range(10):
        rm_frame = "0" * (8 - len(str(sframe))) + str(sframe)
        cp_frame = "0" * (8 - len(str(toload))) + str(toload)
        sframe = sframe + 1
        toload = toload + 1
        for j in range(settings.NB_CAMS):
            command = os.system("rm gtm_hit/static/gtm_hit/frames/"+ settings.CAMS[j] + "/" + rm_frame + ".png")
            command = os.system("cp gtm_hit/static/gtm_hit/day_2/annotation_final/"+ settings.CAMS[j] + "/begin/" + cp_frame + ".png gtm_hit/static/gtm_hit/frames/"+ settings.CAMS[j] + "/")

    settings.LASTLOADED = settings.LASTLOADED + 10

def save_db(request):
    ##set_trace()
    if is_ajax(request) and request.method == 'POST':
        try:
            data = json.loads(request.POST['data'])
            frame_id = request.POST['ID']
            worker_id = request.POST['workerID']
            # Check if the frame exists or create a new frame object
            worker,_ = Worker.objects.get_or_create(workerID=worker_id)
            frame, created = MultiViewFrame.objects.get_or_create(frame_id=frame_id, worker=worker)
            ##set_trace()
            # Iterate through each annotation in the data and create an annotation object for it
            for annotation_data in data:
                person, _ = Person.objects.get_or_create(person_id=annotation_data['personID'])
                # Create a new annotation object for the given person and frame
                if person.person_id==42:
                    pass
                    #set_trace()
                try: 
                    annotation = Annotation.objects.get(person=person, frame=frame)
                    annotation.person=person
                    annotation.frame=frame
                    annotation.rectangle_id=annotation_data['rectangleID']
                    annotation.rotation_theta=annotation_data['rotation_theta']
                    annotation.Xw=annotation_data['Xw']
                    annotation.Yw=annotation_data['Yw']
                    annotation.Zw=annotation_data['Zw']
                    annotation.object_size_x=annotation_data['object_size'][0]
                    annotation.object_size_y=annotation_data['object_size'][1]
                    annotation.object_size_z=annotation_data['object_size'][2]
                except Annotation.DoesNotExist:
                    annotation = Annotation(
                        person=person,
                        frame=frame,
                        rectangle_id=annotation_data['rectangleID'],
                        rotation_theta=annotation_data['rotation_theta'],
                        Xw=annotation_data['Xw'],
                        Yw=annotation_data['Yw'],
                        Zw=annotation_data['Zw'],
                        object_size_x=annotation_data['object_size'][0],
                        object_size_y=annotation_data['object_size'][1],
                        object_size_z=annotation_data['object_size'][2],
                    )

                # Save the annotation object to the database
                annotation.save()
                save_2d_views(annotation)
                
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
    #set_trace()
    if is_ajax(request):
        try:
            frame_id = request.POST['ID']
            worker_id = request.POST['workerID']

            # Check if the person and frame objects exist
            worker = Worker.objects.get(workerID=worker_id)
            frame = MultiViewFrame.objects.get(frame_id=frame_id,worker=worker)
            ##set_trace()
            retjson = []
            camviews=View.objects.all()
            for camview in camviews:
                #set_trace()
                a2l = serialize_annotation2dviews_with_person_id(Annotation2DView.objects.filter(annotation__frame=frame, view=camview))
                retjson.append(a2l)
            #a2l = list(Annotation2DView.objects.filter(annotation__frame=frame, view=View.objects.get(view_id=0)).values())
            return HttpResponse(json.dumps(retjson), content_type="application/json")

            # Read the serialized views from the JSON file and deserialize them
            labels_directory = os.path.join('./gtm_hit/static/gtm_hit/dset/', settings.DSETNAME, 'labels', worker_id)
            with open(os.path.join(labels_directory, f'{worker_id}_{frame_id}.json'), 'r') as rect_json:
                # Deserialize the views from the serialized data
                views = serializers.deserialize('json', rect_json)

            # Iterate through each view and prefetch the related annotation and annotation2dview objects
            for view in views:
                view.object = view.object.prefetch_related('annotation_set__twod_views').select_related('frame')

            # Serialize the views and send them as a response
            serialized_views = serializers.serialize('json', views)
            return HttpResponse(serialized_views, content_type="application/json")

        except (Person.DoesNotExist, MultiViewFrame.DoesNotExist, FileNotFoundError, KeyError):
            return HttpResponse("Error")

    return HttpResponse("Error")

def tracklet(request):
    if is_ajax(request):
        try:

            person_id = int(float(request.POST['personID']))
            frame_id = int(float(request.POST['frameID']))

            multiview_tracklet=get_annotation2dviews_for_frame_and_person(frame_id,person_id)
            #set_trace()
            return HttpResponse(json.dumps(multiview_tracklet), content_type="application/json")
        except KeyError:
            return HttpResponse("Error")
def save_2d_views(annotation):
    ###set_trace()
    for i in range(settings.NB_CAMS):
        view,_ = View.objects.get_or_create(view_id=i)
        cuboid = geometry.get_cuboid2d_from_annotation(annotation, settings.CALIBS[i])
        p1,p2 = geometry.get_bounding_box(cuboid)
        # Set the cuboid points for the annotation2dview object
        ##set_trace()
        try:
         annotation2dview = Annotation2DView.objects.get(
            view=view,
            annotation=annotation)
        except Annotation2DView.DoesNotExist:
            annotation2dview = Annotation2DView(
                view=view,
                annotation=annotation
            )
        annotation2dview.x1 = p1[0]
        annotation2dview.y1 = p1[1]
        annotation2dview.x2 = p2[0]
        annotation2dview.y2 = p2[1]
        annotation2dview.set_cuboid_points_2d(cuboid)
        annotation2dview.save()

def serialize_annotation2dviews_with_person_id(queryset):
    serialized_data = []
    for atdv in queryset:
        serialized_view = {
            'rectangleID': atdv.annotation.rectangle_id,
            'cameraID': atdv.view.view_id,
            'person_id': atdv.annotation.person.person_id,  # Include the person_id
            'object_size': atdv.annotation.object_size,
            'rotation_theta': atdv.annotation.rotation_theta,
            'Xw': atdv.annotation.Xw,
            'Yw': atdv.annotation.Yw,
            'Zw': atdv.annotation.Zw,
            'x1': atdv.x1,
            'y1': atdv.y1,
            'x2': atdv.x2,
            'xMid': atdv.x1+(atdv.x2-atdv.x1)/2,
            'y2': atdv.y2,
            'cuboid':  [atdv.cuboid_points[0:2],
                        atdv.cuboid_points[2:4],
                        atdv.cuboid_points[4:6],
                        atdv.cuboid_points[6:8],
                        atdv.cuboid_points[8:10],
                        atdv.cuboid_points[10:12],
                        atdv.cuboid_points[12:14],
                        atdv.cuboid_points[14:16],
                        atdv.cuboid_points[16:18],
                        atdv.cuboid_points[18:20],
                        ],
        }
        serialized_data.append(serialized_view)
    return serialized_data

def get_annotation2dviews_for_frame_and_person(frame_id, person_id):
    # Get the MultiViewFrame object for the given frame_id
    #frame = MultiViewFrame.objects.get(frame_id=frame_id)

    # Calculate the range of frame_ids for 5 frames before and 5 frames after the given frame
    frame_id_start = max(1, frame_id - 10)
    frame_id_end = frame_id + 10

    # Get the Person object for the given person_id
    person = Person.objects.get(person_id=person_id)

    # Filter the Annotation2DView objects using the calculated frame range and the Person object
    annotation2dviews = Annotation2DView.objects.filter(
        annotation__frame__frame_id__gte=frame_id_start,
        annotation__frame__frame_id__lte=frame_id_end,
        annotation__person=person,
    )
    #set_trace()
    camtrackviews = {}
    for annotation2dview in annotation2dviews:
        vid = annotation2dview.view.view_id
        if vid not in camtrackviews:
            camtrackviews[vid] = []
        camtrackviews[vid].append((annotation2dview.annotation.frame.frame_id,annotation2dview.cuboid_points_2d[8]))
    for view_id in camtrackviews:
        camtrackviews[view_id].sort()
    return camtrackviews

