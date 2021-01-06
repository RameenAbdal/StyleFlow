from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
import cv2




class ReferenceDialog(QDialog):
    def __init__(self, Parent):
        QDialog.__init__(self, Parent)
        self.Form = Parent

class GraphicsScene(QGraphicsScene):
    def __init__(self, Form):
        QGraphicsScene.__init__(self)
        self.mouse_clicked = False
        self.Form = Form

        self.prev_pickedImageIndex = -1
        self.pickedImageIndex = -1



    def reset(self):
        pass

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)



    def mousePressEvent(self, event):
        self.mouse_clicked = True
        self.point = event.scenePos()
        self.showImage(self.point)

        if self.prev_pickedImageIndex != -1:
            cv2.circle(self.Form.map, tuple((self.Form.X_samples[self.prev_pickedImageIndex] * 1024).astype(int)), 6, (0, 0, 255), -1)

        cv2.circle(self.Form.map, tuple((self.Form.X_samples[self.pickedImageIndex] * 1024).astype(int)), 6, (255, 0, 0), -1)

        self.Form.update_scene_image()




    def mouseReleaseEvent(self, event):
        self.mouse_clicked = False
        self.prev_pickedImageIndex = self.pickedImageIndex



    def mouseMoveEvent(self, event):
        pass


    def showImage(self, point):
        point2D = np.array([point.x(), point.y()]) / 1024
        distances, indices = self.Form.nbrs.kneighbors(point2D.reshape(-1, 2))
        # print(distances, indices)


        self.pickedImageIndex = int(indices)
        # img_path = self.Form.img_list[self.pickedImageIndex]


        self.Form.update_GT_scene_image()



    def undo(self):
        pass