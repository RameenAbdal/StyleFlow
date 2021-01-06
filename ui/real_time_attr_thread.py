
import numpy as np
from PyQt5.QtCore import QThread
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import tensorflow as tf
import PIL
import datetime
import os
import skimage.io
import time


class RealTimeAttrThread(QThread):


    def __init__(self, Form):
        super().__init__()
        self.Form = Form


    def render(self, attr_index, raw_slide_value, sess):
        self.sess = sess
        self.attr_index = attr_index
        self.raw_slide_value = raw_slide_value
        self.start()


    def run(self):
        with self.sess.as_default():


            self.Form.real_time_editing(self.attr_index, self.raw_slide_value)
            self.Form.real_scene_update.emit(True)







