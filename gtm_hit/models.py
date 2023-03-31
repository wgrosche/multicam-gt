from django.db import models
from django.core.validators import validate_comma_separated_integer_list
import json
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone
import numpy as np
class Worker(models.Model):
    workerID = models.TextField(primary_key=True,max_length=40)
    frameNB = models.IntegerField(default=-1)
    frame_labeled = models.PositiveSmallIntegerField(default=0)
    #validationCode = models.PositiveIntegerField(default=0)
    finished = models.BooleanField(default=False)
    state = models.IntegerField(default=-1)
    tuto = models.BooleanField(default=False)
    time_list = models.TextField(default="")
    def increaseFrame(self,val):
        self.frame_labeled = self.frame_labeled + val
    def decreaseFrame(self,val):
        self.frame_labeled = self.frame_labeled - val
    def setTimeList(self,x):
        self.time_list = json.dumps(x)
    def getTimeList(self):
        return json.loads(self.time_list)
    def __str__(self):
        return 'Worker: ' + self.workerID
    
class ValidationCode(models.Model):
    validationCode = models.TextField(primary_key=True)
    worker = models.OneToOneField('Worker',on_delete=models.CASCADE)
    def __str__(self):
        return 'Code: ' + self.worker.workerID
class Person(models.Model):
    person_id = models.IntegerField(primary_key=True)
    annotation_complete = models.BooleanField(default=False)
    def __str__(self):
        return f"PersonID{self.person_id}"
    def __repr__(self):
        return f"PersonID{self.person_id}"
class MultiViewFrame(models.Model):
    frame_id = models.IntegerField(verbose_name="MultiView ID")
    timestamp = models.DateTimeField(default=timezone.now)
    undistorted = models.BooleanField(default=False)
    worker = models.ForeignKey(Worker, on_delete=models.CASCADE,default="IVAN")

    class Meta:
        unique_together = ('frame_id', 'undistorted', 'worker')
    def __str__(self):
        return f"{'UN' if self.undistorted else ''}DISTORTED MVFRAME{self.frame_id}"
class View(models.Model):
    view_id = models.IntegerField(primary_key=True,verbose_name="View ID")
    def __str__(self):
        return f"CAM{self.view_id+1}"
class Annotation(models.Model):
    frame = models.ForeignKey(MultiViewFrame, on_delete=models.CASCADE)
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    creation_method = models.TextField(default="existing_annotation")
    validated = models.BooleanField(default=True)
    class Meta:
        unique_together = ('frame', 'person')
    rectangle_id = models.CharField(max_length=100)
    
    rotation_theta = models.FloatField()
    Xw = models.FloatField(verbose_name="X World Coordinate")
    Yw = models.FloatField(verbose_name="Y World Coordinate")
    Zw = models.FloatField(verbose_name="Z World Coordinate")

    object_size_x = models.FloatField()
    object_size_y = models.FloatField()
    object_size_z = models.FloatField()
    
    @property
    def object_size(self):
        return [self.object_size_x, self.object_size_y, self.object_size_z]

    @property
    def world_point(self):
        return np.array([self.Xw, self.Yw, self.Zw]).reshape(-1,1)

    def __str__(self):
        return f"{'UN' if self.frame.undistorted else ''}DISTORTED MVFRAME{self.frame.frame_id} PersonID{self.person.person_id} rectangleID{self.rectangle_id.split('_')[-1]}"
class Annotation2DView(models.Model):
    view = models.ForeignKey(View, on_delete=models.CASCADE)
    annotation = models.ForeignKey(Annotation, related_name="twod_views", on_delete=models.CASCADE)
    class Meta:
        unique_together = ('view', 'annotation')
    x1 = models.FloatField(null=True)
    y1 = models.FloatField(null=True)
    x2 = models.FloatField(null=True)
    y2 = models.FloatField(null=True)
    cuboid_points = ArrayField(models.FloatField(), size=20, null=True)

    @property
    def cuboid_points_2d(self):
        return [
            self.cuboid_points[0:2],
            self.cuboid_points[2:4],
            self.cuboid_points[4:6],
            self.cuboid_points[6:8],
            self.cuboid_points[8:10],
            self.cuboid_points[10:12],
            self.cuboid_points[12:14],
            self.cuboid_points[14:16],
            self.cuboid_points[16:18],
            self.cuboid_points[18:20],
        ]

    def set_cuboid_points_2d(self, points):
        self.cuboid_points = [point for sublist in points for point in sublist]

    def __str__(self):
        return f"{'UN' if self.annotation.frame.undistorted else ''}DISTORTED FRAME{self.annotation.frame.frame_id} CAM{self.view.view_id+1} PersonID{self.annotation.person.person_id} rectangleID{self.annotation.rectangle_id}"
    

class SingleViewFrame(models.Model):
    frame_id = models.IntegerField(verbose_name="SingleView ID")
    timestamp = models.DateTimeField(default=timezone.now)
    view = models.ForeignKey(View, on_delete=models.CASCADE)
    def __str__(self):
        return f"CAM{self.view_id+1} SVFRAME{self.frame_id} {self.timestamp}" 