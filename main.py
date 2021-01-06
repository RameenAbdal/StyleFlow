import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import qdarkstyle
import qdarkgraystyle
from time import time

from options.test_options import TestOptions
from ui.ui import Ui_Form

import numpy as np
from sklearn.neighbors import NearestNeighbors
from glob import glob
import cv2

from ui.mouse_event import GraphicsScene
from ui.GT_mouse_event import GTScene
from utils import Build_model
import pickle
from sklearn.manifold import TSNE
from ui.ui import transfer_real_to_slide, invert_slide_to_real, light_transfer_real_to_slide, \
    light_invert_slide_to_real, attr_degree_list
import torch
from module.flow import cnf
import os
import tensorflow as tf

from ui.real_time_attr_thread import RealTimeAttrThread
from ui.real_time_light_thread import RealTimeLightThread

# np.random.seed(2)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ExWindow(QMainWindow):
    def __init__(self, opt):
        super().__init__()
        self.EX = Ex(opt)


class Ex(Ui_Form):
    real_scene_update = pyqtSignal(bool, name='update_real_scene')

    def __init__(self, opt):

        super().__init__()

        self.lock_mode = False
        self.sample_num = 10
        self.truncation_psi = 0.5
        self.snapshot = 0
        self.his_image = []
        self.at_intial_point = False

        self.keep_indexes = [2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
                             369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301,
                             599]
        # self.keep_indexes = [i for i in range(0,100)]
        # self.keep_indexes = [0]
        self.keep_indexes = np.array(self.keep_indexes).astype(np.int)

        self.zero_padding = torch.zeros(1, 18, 1).cuda()
        self.real_scene_update.connect(self.update_real_scene)

        self.attr_order = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
        self.lighting_order = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

        self.init_deep_model(opt)
        self.init_data_points()

        self.setupUi(self)
        self.show()

        self.scene = GraphicsScene(self)
        # self.scene.setSceneRect(0, 0, 1024, 1024)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.show()

        self.lock_scene = GTScene(self)
        self.lockView.setScene(self.lock_scene)
        self.lockView.setAlignment(Qt.AlignCenter)
        self.lockView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lockView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lockView.hide()

        self.GT_scene = GTScene(self)
        self.resultView.setScene(self.GT_scene)
        self.resultView.setAlignment(Qt.AlignCenter)
        self.resultView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.resultView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.realtime_attr_thread = RealTimeAttrThread(self)

        self.realtime_light_thread = RealTimeLightThread(self)

        self.init_screen()

    def init_deep_model(self, opt):
        self.opt = opt
        self.model = Build_model(self.opt)
        self.w_avg = self.model.Gs.get_var('dlatent_avg')

        self.prior = cnf(512, '512-512-512-512-512', 17, 1)

        self.prior.load_state_dict(torch.load('flow_weight/modellarge10k.pt'))

        self.prior.eval()

    def init_screen(self):
        self.update_scene_image()

    def update_scene_image(self):
        qim = QImage(self.map.data, self.map.shape[1], self.map.shape[0], self.map.strides[0],
                     QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qim)
        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()
        self.scene.addPixmap(pixmap)

    def update_GT_scene_image(self):

        self.at_intial_point = True

        # self.scene.pickedImageIndex = 28

        self.w_current = self.all_w[self.scene.pickedImageIndex].copy()

        self.attr_current = self.all_attr[self.scene.pickedImageIndex].copy()
        self.light_current = self.all_lights[self.scene.pickedImageIndex].copy()

        self.attr_current_list = [self.attr_current[i][0] for i in range(len(self.attr_order))]

        self.light_current_list = [0 for i in range(len(self.lighting_order))]

        for i, j in enumerate(self.attr_order):
            self.slider_list[i].setValue(transfer_real_to_slide(j, self.attr_current_list[i]))

        for i, j in enumerate(self.lighting_order):
            self.lighting_slider_list[i].setValue(0)

        ################################  calculate attributes array first, then change the values of attributes

        self.q_array = torch.from_numpy(self.w_current).cuda().clone().detach()
        self.array_source = torch.from_numpy(self.attr_current).type(torch.FloatTensor).cuda()
        self.array_light = torch.from_numpy(self.light_current).type(torch.FloatTensor).cuda()
        self.pre_lighting_distance = [self.pre_lighting[i] - self.array_light for i in range(len(self.lighting_order))]

        self.final_array_source = torch.cat([self.array_light, self.array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
        self.final_array_target = torch.cat([self.array_light, self.array_source.unsqueeze(0).unsqueeze(-1)], dim=1)
        # print(self.q_array.shape, self.final_array_source.shape, self.zero_padding.shape)
        self.fws = self.prior(self.q_array, self.final_array_source, self.zero_padding)

        self.GAN_image = self.model.generate_im_from_w_space(self.w_current)[0]

        qim = QImage(self.GAN_image.data, self.GAN_image.shape[1], self.GAN_image.shape[0], self.GAN_image.strides[0],
                     QImage.Format_RGB888)

        showedImagePixmap = QPixmap.fromImage(qim)
        # showedImagePixmap = showedImagePixmap.scaled(QSize(256, 256), Qt.IgnoreAspectRatio)
        self.GT_scene.reset()
        if len(self.GT_scene.items()) > 0:
            self.GT_scene.reset_items()
        self.lock_scene.reset()
        if len(self.lock_scene.items()) > 0:
            self.lock_scene.reset_items()

        self.GT_scene.addPixmap(showedImagePixmap)
        self.lock_scene.addPixmap(showedImagePixmap)

        for i in range(15):
            self.style_button_list[i].setIcon(QIcon())

        self.style_button_list[0].setIcon(QIcon(showedImagePixmap.scaled(128, 128)))
        self.his_image = []
        self.his_image.append(qim.copy())

        self.at_intial_point = False

    def update_lock_scene(self):
        qim = QImage(self.GAN_image.data, self.GAN_image.shape[1], self.GAN_image.shape[0], self.GAN_image.strides[0],
                     QImage.Format_RGB888)

        showedImagePixmap = QPixmap.fromImage(qim)
        if len(self.lock_scene.items()) > 0:
            self.lock_scene.reset_items()
        self.lock_scene.addPixmap(showedImagePixmap)
        self.snapshot += 1
        self.style_button_list[self.snapshot].setIcon(QIcon(showedImagePixmap.scaled(128, 128)))
        self.his_image.append(qim.copy())

    def update_real_scene(self):
        qim = QImage(self.GAN_image.data, self.GAN_image.shape[1], self.GAN_image.shape[0], self.GAN_image.strides[0],
                     QImage.Format_RGB888)

        showedImagePixmap = QPixmap.fromImage(qim)

        self.GT_scene.addPixmap(showedImagePixmap)

    def show_his_image(self, i):

        qim = self.his_image[i]
        showedImagePixmap = QPixmap.fromImage(qim)
        if len(self.lock_scene.items()) > 0:
            self.lock_scene.reset_items()
        self.lock_scene.addPixmap(showedImagePixmap)

    def real_time_editing_thread(self, attr_index, raw_slide_value):
        self.realtime_attr_thread.render(attr_index, raw_slide_value, tf.get_default_session())

    def real_time_light_thread(self, light_index, raw_slide_value):

        self.realtime_light_thread.render(light_index, raw_slide_value, tf.get_default_session())

    def real_time_lighting(self, light_index, raw_slide_value):

        if not self.at_intial_point:

            real_value = light_invert_slide_to_real(self.lighting_order[light_index], raw_slide_value)

            self.light_current_list[light_index] = real_value

            ###############################
            ###############  calculate attributes array first, then change the values of attributes

            lighting_final = self.array_light.clone().detach()
            for i in range(len(self.lighting_order)):
                lighting_final += self.light_current_list[i] * self.pre_lighting_distance[i]

            self.final_array_target[:, :9] = lighting_final

            self.rev = self.prior(self.fws[0], self.final_array_target, self.zero_padding, True)
            self.rev[0][0][0:7] = self.q_array[0][0:7]
            self.rev[0][0][12:18] = self.q_array[0][12:18]

            self.w_current = self.rev[0].detach().cpu().numpy()
            self.q_array = torch.from_numpy(self.w_current).cuda().clone().detach()

            self.fws = self.prior(self.q_array, self.final_array_target, self.zero_padding)

            self.GAN_image = self.model.generate_im_from_w_space(self.w_current)[0]

        else:
            pass

    def real_time_editing(self, attr_index, raw_slide_value):

        if not self.at_intial_point:

            real_value = invert_slide_to_real(self.attr_order[attr_index], raw_slide_value)

            attr_change = real_value - self.attr_current_list[attr_index]
            attr_final = attr_degree_list[attr_index] * attr_change + self.attr_current_list[attr_index]

            self.final_array_target[0, attr_index + 9, 0, 0] = attr_final

            self.rev = self.prior(self.fws[0], self.final_array_target, self.zero_padding, True)

            if attr_index == 0:
                self.rev[0][0][8:] = self.q_array[0][8:]

            elif attr_index == 1:
                self.rev[0][0][:2] = self.q_array[0][:2]
                self.rev[0][0][4:] = self.q_array[0][4:]

            elif attr_index == 2:

                self.rev[0][0][4:] = self.q_array[0][4:]



            elif attr_index == 3:
                self.rev[0][0][4:] = self.q_array[0][4:]

            elif attr_index == 4:
                self.rev[0][0][6:] = self.q_array[0][6:]

            elif attr_index == 5:
                self.rev[0][0][:5] = self.q_array[0][:5]
                self.rev[0][0][10:] = self.q_array[0][10:]

            elif attr_index == 6:
                self.rev[0][0][0:4] = self.q_array[0][0:4]
                self.rev[0][0][8:] = self.q_array[0][8:]

            elif attr_index == 7:
                self.rev[0][0][:4] = self.q_array[0][:4]
                self.rev[0][0][6:] = self.q_array[0][6:]

            self.w_current = self.rev[0].detach().cpu().numpy()
            self.q_array = torch.from_numpy(self.w_current).cuda().clone().detach()

            self.fws = self.prior(self.q_array, self.final_array_target, self.zero_padding)

            self.GAN_image = self.model.generate_im_from_w_space(self.w_current)[0]
        else:
            pass

    def reset_Wspace(self):

        self.update_GT_scene_image()

    def init_data_points(self):

        self.raw_w = pickle.load(open("data/sg2latents.pickle", "rb"))

        self.raw_TSNE = np.load('data/TSNE.npy')

        self.raw_attr = np.load('data/attributes.npy')

        self.raw_lights2 = np.load('data/light.npy')
        self.raw_lights = self.raw_lights2

        self.all_w = np.array(self.raw_w['Latent'])[self.keep_indexes]
        self.all_attr = self.raw_attr[self.keep_indexes]
        self.all_lights = self.raw_lights[self.keep_indexes]

        light0 = torch.from_numpy(self.raw_lights2[8]).type(torch.FloatTensor).cuda()
        light1 = torch.from_numpy(self.raw_lights2[33]).type(torch.FloatTensor).cuda()
        light2 = torch.from_numpy(self.raw_lights2[641]).type(torch.FloatTensor).cuda()
        light3 = torch.from_numpy(self.raw_lights2[547]).type(torch.FloatTensor).cuda()
        light4 = torch.from_numpy(self.raw_lights2[28]).type(torch.FloatTensor).cuda()
        light5 = torch.from_numpy(self.raw_lights2[34]).type(torch.FloatTensor).cuda()

        self.pre_lighting = [light0, light1, light2, light3, light4, light5]

        self.X_samples = self.raw_TSNE[self.keep_indexes]

        self.map = np.ones([1024, 1024, 3], np.uint8) * 255

        for point in self.X_samples:
            ######### don't use np.uint8 in tuple((point*1024).astype(int))
            cv2.circle(self.map, tuple((point * 1024).astype(int)), 6, (0, 0, 255), -1)

        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.X_samples)

    @pyqtSlot()
    def lock_switch(self):
        self.lock_mode = not self.lock_mode

        if self.lock_mode:
            self.brushButton.setStyleSheet("background-color:")

            self.lockView.show()
            self.graphicsView.hide()
        else:
            self.brushButton.setStyleSheet("background-color:")
            self.brushButton.setStyleSheet("background-color: #85adad")
            self.graphicsView.show()
            self.lockView.hide()


if __name__ == '__main__':
    opt = TestOptions().parse()

    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = ExWindow(opt)
    # ex = Ex(opt)
    sys.exit(app.exec_())
