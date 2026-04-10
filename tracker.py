import numpy as np

class CentroidTracker:
    def __init__(self):
        self.objects = {}
        self.id_count = 0

    def update(self, boxes):

        new_objects = {}

        for box in boxes:
            x1,y1,x2,y2 = box
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            new_objects[self.id_count] = (cx,cy,box)
            self.id_count += 1

        self.objects = new_objects
        return self.objects