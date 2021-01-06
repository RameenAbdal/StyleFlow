from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from functools import partial
import glob
import copy
import os
from ui.mouse_event import ReferenceDialog


Lb_width = 100
Lb_height = 40
Lb_row_shift = 25
Lb_col_shift = 20
Lb_x = 110
Lb_y = 700


Tb_width = 150
Tb_height = 60
Tb_row_shift = 50
Tb_col_shift = 5
Tb_x = 150
Tb_y = 63


square_size = 100




attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7,0.93, 1.]


min_dic = {'Gender': 0, 'Glasses': 0, 'Yaw': 0, 'Pitch': 0, 'Baldness': 0, 'Beard': 0, 'Age': 0, 'Expression': 0}
max_dic = {'Gender': 1, 'Glasses': 1, 'Yaw': 1, 'Pitch': 1, 'Baldness': 1, 'Beard': 1, 'Age': 1, 'Expression': 1}

attr_interval = 80
interval_dic = {'Gender': attr_interval, 'Glasses': attr_interval, 'Yaw': attr_interval, 'Pitch': attr_interval,
                'Baldness': attr_interval, 'Beard': attr_interval, 'Age': attr_interval, 'Expression': attr_interval}
# set_values_dic = {i: int(interval_dic[i]/2) for i in interval_dic}
gap_dic = {i: max_dic[i] - min_dic[i] for i in max_dic}



light_degree = 1
light_min_dic = {'light': 0}
light_max_dic = {'light': light_degree}


light_interval = 80
light_interval_dic = {'light': light_interval}
light_set_values_dic = {i: 0 for i in light_interval_dic}
light_gap_dic = {i: light_max_dic[i] - light_min_dic[i] for i in light_max_dic}




def transfer_real_to_slide(name, real_value):
    return int((real_value - min_dic[name]) / (gap_dic[name]) * interval_dic[name])

def invert_slide_to_real(name, slide_value):
    return float(slide_value /interval_dic[name] * (gap_dic[name]) + min_dic[name])



def light_transfer_real_to_slide(name, real_value):
    return int((real_value - light_min_dic[name]) / (light_gap_dic[name]) * light_interval_dic[name])

def light_invert_slide_to_real(name, slide_value):
    return float(slide_value /light_interval_dic[name] * (light_gap_dic[name]) + light_min_dic[name])



class Ui_Form(QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowTitle("Let's Party StyleFlow")
        Form.resize(3456, 1380 + 400 + 164)


        self.graphicsView = QtWidgets.QGraphicsView(self)
        self.graphicsView.setGeometry(QtCore.QRect(1250, 150, 1028, 1028))
        self.graphicsView.setObjectName("graphicsView")

        self.lockView = QtWidgets.QGraphicsView(self)
        self.lockView.setGeometry(QtCore.QRect(150, 150, 1028, 1028))
        self.lockView.setObjectName("lockView")

        self.referenceView = QtWidgets.QGraphicsView(self)
        self.referenceView.setGeometry(QtCore.QRect(2350, 150, 1028, 1028))
        self.referenceView.setObjectName('referenceView')


        self.resultView = QtWidgets.QGraphicsView(self)
        self.resultView.setGeometry(QtCore.QRect(1250, 150, 1028, 1028))
        self.resultView.setObjectName("blendingView")


        self.referDialog = ReferenceDialog(self)
        self.referDialog.setObjectName('Reference Dialog')
        self.referDialog.setWindowTitle('Reference Image')
        self.referDialogImage = QtWidgets.QLabel(self.referDialog)
        self.referDialogImage.setFixedSize(1024, 1024)


        self.add_tool_buttons(Form)
        self.add_intermediate_results_button(Form)
        self.add_Parameters_widgets(Form)
        # self.add_lighting_widgets(Form)

        QtCore.QMetaObject.connectSlotsByName(self)


    def add_tool_buttons(self, Form):
        KaustLogo = QtWidgets.QLabel(self)
        KaustLogo.setPixmap(QPixmap('icons/1999780_200.png').scaled(90, 90))
        KaustLogo.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 25, 110, 110))

        self.newButton = QtWidgets.QPushButton(Form)
        self.newButton.setGeometry(QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 150, square_size, square_size))
        self.newButton.setObjectName("openButton")
        self.newButton.setIcon(QIcon('icons/add_new_document.png'))
        self.newButton.setIconSize(QSize(square_size, square_size))
        # self.newButton.clicked.connect(Form.run_deep_model)

        self.openButton = QtWidgets.QPushButton(Form)
        self.openButton.setGeometry(
            QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 150 + square_size * 1 + 25 * 1, square_size, square_size))
        self.openButton.setObjectName("openButton")
        self.openButton.setIcon(QIcon('icons/open.png'))
        self.openButton.setIconSize(QSize(square_size, square_size))
        # self.openButton.clicked.connect(Form.open)

        self.fillButton = QtWidgets.QPushButton(Form)
        self.fillButton.setGeometry(
            QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 150 + square_size * 2 + 25 * 2, square_size, square_size))
        self.fillButton.setObjectName("fillButton")
        self.fillButton.setIcon(QIcon('icons/paint_can.png'))
        self.fillButton.setIconSize(QSize(square_size, square_size))
        # self.fillButton.clicked.connect(partial(Form.mode_select, 2))

        self.brushButton = QtWidgets.QPushButton(Form)
        self.brushButton.setGeometry(
            QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 150 + square_size * 3 + 25 * 3, square_size, square_size))
        self.brushButton.setObjectName("brushButton")
        self.brushButton.setIcon(QIcon('icons/foot2.png'))
        self.brushButton.setIconSize(QSize(square_size, square_size))
        self.brushButton.clicked.connect(Form.lock_switch)
        self.brushButton.setStyleSheet("background-color: #85adad")

        self.undoButton = QtWidgets.QPushButton(Form)
        self.undoButton.setGeometry(
            QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 150 + square_size * 4 + 25 * 4, square_size, square_size))
        self.undoButton.setObjectName("undolButton")
        self.undoButton.setIcon(QIcon('icons/undo.png'))
        self.undoButton.setIconSize(QSize(square_size, square_size))
        # self.undoButton.clicked.connect(Form.undo)

        self.saveButton = QtWidgets.QPushButton(Form)
        self.saveButton.setGeometry(
            QtCore.QRect(int(Lb_x - 1 * Lb_row_shift - 60), 150 + square_size * 5 + 25 * 5, square_size, square_size))
        self.saveButton.setObjectName("saveButton")
        self.saveButton.setIcon(QIcon('icons/reset1.png'))
        self.saveButton.setIconSize(QSize(square_size, square_size))
        self.saveButton.clicked.connect(Form.reset_Wspace)


        # self.newButton.clicked.connect(Form.init_screen)


    def add_intermediate_results_button(self, Form):

        self.reset_snapshot_button = QtWidgets.QPushButton(Form)
        self.reset_snapshot_button.setGeometry(QtCore.QRect(int(Lb_x - 1*Lb_row_shift - 60), 1211 + 400 + 100, 100, 100))
        self.reset_snapshot_button.setIcon(QIcon('icons/save.png'))
        self.reset_snapshot_button.setIconSize(QSize(100, 100))
        self.reset_snapshot_button.clicked.connect(Form.update_lock_scene)


        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(150 - 10, 1200 + 400 + 100, 3380, 155))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollArea.setAlignment(Qt.AlignCenter)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 2250 + 400, 128))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        # horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(27)
        self.horizontalLayout.setAlignment(Qt.AlignLeft)

        self.style_button_list = []
        for i in range(Form.opt.max_result_snapshots):
            style_button = QtWidgets.QPushButton()
            style_button.setFixedSize(128, 128)
            style_button.setIcon(QIcon())
            style_button.setIconSize(QSize(128, 128))
            style_button.clicked.connect(partial(Form.show_his_image, i))
            style_button.snap_shot_name = None
            style_button.setStyleSheet("background-color: transparent")
            self.style_button_list.append(style_button)
            #style_button.hide()
            self.horizontalLayout.addWidget(style_button)



        self.scrollArea.setWidget(self.scrollAreaWidgetContents)



    def add_Parameters_widgets(self,Form):

        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # sizePolicy.setHeightForWidth(True)

        icon_list = ['icons/Attr/gender2.png','icons/Attr/glasses4.png','icons/Attr/yaw.png','icons/Attr/pitch.png',
                     'icons/Attr/bald3.png','icons/Attr/beard3.png','icons/Attr/age1.png','icons/Attr/expression2.png','icons/Attr/lighting2.png']


        self.formGroupBox1 = QtWidgets.QGroupBox("Attributes", Form)
        self.formGroupBox1.setGeometry(QtCore.QRect(1764 - 200 -100 -400 -300, 1190 + 30, 400+3*400 + 200 + 200, 380 + 50))
        # formlayout1 = QtWidgets.QFormLayout()

        formlayout1 = QtWidgets.QGridLayout()
        formlayout1.setHorizontalSpacing(30)
        # formlayout1.setAlignment(Qt.AlignCenter)
        # formlayout1.setVerticalSpacing(5)

        # formlayout1.setFormAlignment(Qt.AlignCenter)
        # formlayout1.setVerticalSpacing(15)


        self.slider_list = []

        for j, i in enumerate(self.attr_order):
            slider = QtWidgets.QSlider(Form)
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setObjectName(i)
            slider.setMaximum(interval_dic[i])
            slider.setValue(0)

            slider.valueChanged.connect(partial(Form.real_time_editing_thread, j))


            self.slider_list.append(slider)

            # formlayout1.addRow(QtWidgets.QLabel(i + ':'), slider)
            layout = QtWidgets.QHBoxLayout()
            # layout.setStretch(0, 3)
            # layout.setStretch(1, 7)
            label = QtWidgets.QLabel(i + ':')
            font = label.font()
            font.setPointSize(14)
            label.setFont(font)
            label.setAlignment(Qt.AlignCenter)
            icon = QtWidgets.QPushButton()
            icon.setSizePolicy(sizePolicy)
            # icon.resize(800,800)
            # icon.setSizePolicy(sizePolicy)
            icon.setIcon(QIcon(icon_list[j]))

            layout.addWidget(icon)
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.setStretch(0, 1.5)
            layout.setStretch(1, 2)
            layout.setStretch(2, 6.5)
            # layout.setAlignment(Qt.AlignRight)
            formlayout1.addLayout(layout, j//3,j%3)


            # formlayout1.addRow(lb_v_box, slider_vbox)
            # formlayout1.addRow(totoal_h)

        self.lighting_slider_list = []
        for j, i in enumerate(self.lighting_order):
            slider = QtWidgets.QSlider(Form)
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setObjectName(i)
            slider.setMaximum(light_interval_dic[i])
            slider.setValue(0)
            # slider.sliderReleased.connect(partial(Form.real_time_editing, j))
            # slider.valueChanged.connect(partial(Form.real_time_editing, j))

            slider.valueChanged.connect(partial(Form.real_time_light_thread, j))


            self.lighting_slider_list.append(slider)

            layout = QtWidgets.QHBoxLayout()
            # layout.setStretch(0, 3)
            # layout.setStretch(1, 7)
            label = QtWidgets.QLabel(i + ':')
            font = label.font()
            font.setPointSize(14)
            label.setFont(font)
            label.setAlignment(Qt.AlignCenter)
            icon = QtWidgets.QPushButton()
            icon.setSizePolicy(sizePolicy)
            # icon.resize(800, 800)
            # icon.setSizePolicy(sizePolicy)
            icon.setIcon(QIcon(icon_list[-1]))
            layout.addWidget(icon)
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.setStretch(0, 1.5)
            layout.setStretch(1, 2)
            layout.setStretch(2, 6.5)
            # layout.setAlignment(Qt.AlignRight)
            formlayout1.addLayout(layout, 2, 2)

            # formlayout1.addRow(QtWidgets.QLabel(i + ':'), slider)


        self.formGroupBox1.setLayout(formlayout1)



    def add_lighting_widgets(self,Form):

        self.formGroupBox2 = QtWidgets.QGroupBox("Lighting", Form)
        #self.formGroupBox1.setGeometry(QtCore.QRect(2350, 150, 300, 200))
        self.formGroupBox2.setGeometry(QtCore.QRect(1764 + 25 , 1200, 400, 350))
        formlayout2 = QtWidgets.QFormLayout()

        formlayout2.setFormAlignment(Qt.AlignCenter)
        formlayout2.setVerticalSpacing(20)


        self.lighting_slider_list = []

        for j, i in enumerate(self.lighting_order):
            slider = QtWidgets.QSlider(Form)
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setObjectName(i)
            slider.setMaximum(40)
            slider.setValue(0)
            # slider.sliderReleased.connect(partial(Form.real_time_editing, j))
            # slider.valueChanged.connect(partial(Form.real_time_editing, j))

            slider.valueChanged.connect(partial(Form.real_time_light_thread, j))


            self.lighting_slider_list.append(slider)

            formlayout2.addRow(QtWidgets.QLabel(i + ':'), slider)




        self.formGroupBox2.setLayout(formlayout2)







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
