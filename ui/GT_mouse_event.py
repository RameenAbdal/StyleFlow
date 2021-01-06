from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np



class GTScene(QGraphicsScene):
    def __init__(self, Form):
        QGraphicsScene.__init__(self)
        self.Form = Form

    def reset(self):
        pass

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)
