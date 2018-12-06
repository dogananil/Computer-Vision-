#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 00:14:00 2018

@author: anildogan
"""
import sys
from PyQt5.QtWidgets import QMainWindow,QMessageBox, QApplication,QScrollArea, QWidget, QPushButton, QAction, QGroupBox, QFileDialog, QLabel, QVBoxLayout, QGridLayout, QHBoxLayout,QFrame, QSplitter,QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap, QPalette,QImage
from PyQt5.QtCore import pyqtSlot, Qt
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import math
class App(QMainWindow):
    
    def __init__(self):
        super(App,self).__init__()
        
        self.window = QWidget(self)
        self.setCentralWidget(self.window)
    
        self.inputBox = QGroupBox('Input')
        inputLayout = QVBoxLayout()
        self.inputBox.setLayout(inputLayout)
        
        self.targetBox = QGroupBox('Target')
        targetLAyout = QVBoxLayout()
        self.targetBox.setLayout(targetLAyout)
        
        self.resultBox = QGroupBox('Result')
        resultLayout = QVBoxLayout()
        self.resultBox.setLayout(resultLayout)
        
        self.layout = QGridLayout()
        self.layout.addWidget(self.inputBox, 0, 0)
        self.layout.addWidget(self.targetBox, 0, 1)
        self.layout.addWidget(self.resultBox, 0, 2)
        
        self.window.setLayout(self.layout)
        
        self.image = None
        self.image2 = None
        self.tmp_im = None
        self.imageLabel = None
        self.figure = Figure()
        self.figure2 = Figure()
        self.figure3 = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas3 = FigureCanvas(self.figure3)
        self.lookupRed = np.zeros((256,1))
        self.lookupGreen = np.zeros((256,1))
        self.lookupBlue = np.zeros((256,1))
        self.arnoldpoints = None
        self.bushpoints = None
        self.eq = None
        self.qImg = None
        self.qImg2 = None
        self.qImgResult = None
        self.pixmap01 = None
        self.pixmap_image = None
        self.delaunay_color = (255,0,0)
        
        self.createActions()
        self.createMenu()
        self.createToolBar()
        
        self.setWindowTitle("Histogram")
        self.showMaximized()
        self.show()
        
        
    def createActions(self):
        self.open_inputAct = QAction(' &Open Input',self)
        self.open_inputAct.triggered.connect(self.open_Input)
        self.open_targetAct = QAction(' &Open Target', self)
        self.open_targetAct.triggered.connect(self.open_Target)
        self.exitAct = QAction(' &Exit', self)
        self.exitAct.triggered.connect(self.exit)
        self.triang = QAction(' &Create Triangulation',self)
        self.triang.triggered.connect(self.createTri)
        self.morphing = QAction(' &Morph',self)
        self.morphing.triggered.connect(self.morphFunc)
    
    def createMenu(self):
        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('File')
        self.fileMenu.addAction(self.open_inputAct)
        self.fileMenu.addAction(self.open_targetAct)
        self.fileMenu.addAction(self.exitAct)
    def createToolBar(self):
        self.tri = self.addToolBar("Create Triangulation")
        self.tri.addAction(self.triang)
        self.morph = self.addToolBar("Morph")
        self.morph.addAction(self.morphing)
    def rectcontains(self,rect,point) :
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True
    def draw_delaunay(self, subdiv ) :
        triangleList = subdiv.getTriangleList();
        size = self.image.shape
        r = (0, 0, size[1], size[0])
        for t in triangleList :
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if self.rectcontains(r, pt1) and self.rectcontains(r, pt2) and self.rectcontains(r, pt3) :
                cv2.line(self.image, pt1, pt2, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.image, pt2, pt3, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.image, pt3, pt1, self.delaunay_color, 1, cv2.LINE_AA, 0)
        
        self.qImg = QImage(self.image.data,size[1],size[0],size[1]*3,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageLabel.setAlignment(Qt.AlignCenter)
        #self.inputBox.setLayout(imageLabel)
        self.inputBox.layout().addWidget(self.imageLabel)
    def draw_delaunay1(self, subdiv ) :
        triangleList = subdiv.getTriangleList();
        size = self.image2.shape
        r = (0, 0, size[1], size[0])
        for t in triangleList :
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            for b in self.arnoldpoints:
                temp = (b[1],b[0])
                if(pt1==temp):
                    a = temp
                if(pt2==temp):
                    c = temp
                if(pt3==temp):
                    d = temp
            
            if self.rectcontains(r, pt1) and self.rectcontains(r, pt2) and self.rectcontains(r, pt3) :
                pt1 = a
                pt2 = c
                pt3 = d
                cv2.line(self.image2, pt1, pt2, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.image2, pt2, pt3, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.image2, pt3, pt1, self.delaunay_color, 1, cv2.LINE_AA, 0)
        
        self.qImg = QImage(self.image2.data,size[1],size[0],size[1]*3,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel1.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageLabel1.setAlignment(Qt.AlignCenter)
        self.targetBox.layout().addWidget(self.imageLabel1)
    def createTri(self):
        heightI,widthI,channelsI = self.image.shape
        self.red = self.image[:,:,2]
        self.green = self.image[:,:,1]
        self.blue = self.image[:,:,0] 
        rect1 = (0,0,widthI,heightI)
        subdiv1 = cv2.Subdiv2D(rect1)
        rect2 = (0,0,widthI,heightI)
        subdiv2 = cv2.Subdiv2D(rect2)
       # self.arnoldpoints = [[236,48],[174,54],[116,52],[98,92],[100,132],[194,192],[88,222],[160,242],[208,258],[200,290],[246,288],[282,274],[316,252],[360,228],[388,202],[408,150],[380,96],[348,70],[296,60],[210,62],[206,86],[206,100],[202,172],[202,196],[202,214],[208,224],[226,74],[229,95],[223,11],[217,193],[219,81],[219,173],[225,193],[219,211],[210,191],[296,141],[290,155],[292,165],[272,165],[294,119],[288,111],[268,119],[246,139],[210,135],[142,133],[336,101],[246,143],[328,187],[326,145],[326,115],[120,120],[264,214],[278,80],[282,48],[176,118],[178,154],[4,4],[14,308],[8,150],[472,4],[472,310],[462,150],[322,224]]
       # self.bushpoints =   [[236,12],[282,22],[326,44],[366,66],[416,128],[402,204],[372,238],[312,268],[270,284],[214,286],[246,262],[272,256],[182,236],[146,238],[110,220],[92,204],[106,138],[102,102],[110,74],[120,52],[140,42],[168,42],[204,42],[232,38],[268,38],[242,20],[238,10],[274,16],[220,58],[196,108],[204,138],[198,162],[196,194],[202,216],[212,232],[238,70],[236,89],[234,105],[228,115],[220,103],[224,89],[230,77],[222,167],[226,189],[220,209],[212,191],[214,171],[282,145],[284,173],[242,143],[286,125],[330,109],[324,155],[322,197],[344,153],[138,131],[12,6],[14,308],[12,150],[474,6],[474,150],[474,306],[148,134]]
       #self.arnoldpoints = [[4,4],[475,315],[240,315],[240,4],[4,315],[475,4],[4,160],[240,160],[475,160]]
        self.arnoldpoints = [[5,5],[5,315],[475,5],[475,315],[290,50],[238,52],[240,8],[348,66],[404,118],[396,190],[358,230],[288,264],[224,294],[200,286],[240,258],[174,240],[90,226],[100,132],[112,56],[202,82],[200,194],[224,92],[218,194],[142,136],[202,138],[246,136],[290,136],[284,168],[296,110],[334,102],[332,142],[328,188]]
        self.bushpoints = [[5,5],[5,315],[475,5],[475,315],[324,42],[260,38],[240,8],[360,60],[416,116],[410,200],[360,240],[310,262],[258,284],[216,282],[248,260],[184,236],[92,202],[106,128],[120,52],[206,84],[200,196],[230,98],[222,186],[142,132],[198,142],[244,142],[286,146],[282,178],[286,110],[334,100],[332,152],[326,200]]
        print("HeightI",heightI)
        print("Width",widthI)
        for a in self.arnoldpoints:
            my_tuple = (a[1],a[0])
            subdiv1.insert(my_tuple)
            
        for b in self.bushpoints:
            my_tuple1 = (b[1],b[0])
            subdiv2.insert(my_tuple1)

        
        self.draw_delaunay(subdiv1)
        self.draw_delaunay1(subdiv1)
        #self.draw_delaunay(subdiv1) 
    def morphFunc(self):
        self.red = self.image[:,:,2]
        self.green = self.image[:,:,1]
        self.blue = self.image[:,:,0]
        self.width,self.height = self.blue.shape
        self.blueArray = [0]*256
        self.redArray = [0]*256
        self.greenArray = [0]*256
        for w in range(0,self.width):
            for h in range(0,self.height):
                temp = self.blue[w][h]
                self.blueArray[temp]+=1
        for w in range(0,self.width):
            for h in range(0,self.height):
                temp = self.red[w][h]
                self.redArray[temp]+=1        
        for w in range(0,self.width):
            for h in range(0,self.height):
                temp = self.green[w][h]
                self.greenArray[temp]+=1 
        blueplot = self.figure.add_subplot(313)
        redplot = self.figure.add_subplot(311)
        greenplot = self.figure.add_subplot(312)
        
        blueplot.bar(range(256),self.blueArray,color = 'blue')
        redplot.bar(range(256),self.redArray,color = 'red')
        greenplot.bar(range(256),self.greenArray,color = 'green')
        self.canvas.draw()
        self.inputBox.layout().addWidget(self.canvas)
    def open_Input(self):
        #fileName, _ = QFileDialog.getOpenFileName(self, "Open File",QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Input', '.')
        if fileName:
            self.image = cv2.imread(fileName)
            heightI,widthI,channelsI = self.image.shape
            bytesPerLine = 3 * widthI
            if not self.image.data:
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            
        self.qImg = QImage(self.image.data,widthI,heightI,bytesPerLine,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel = QLabel('image')
        self.imageLabel.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageLabel.setAlignment(Qt.AlignCenter)
        
        self.inputBox.layout().addWidget(self.imageLabel)
       
    
    def open_Target(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Input', '.')
        if fileName:
            self.image2 = cv2.imread(fileName)
            heightT,widthT,channelsT = self.image2.shape
            bytesPerLine = 3 * widthT
            if not self.image2.data:
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            
        self.qImg2 = QImage(self.image2.data,widthT,heightT,bytesPerLine,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel1 = QLabel('image')
        self.imageLabel1.setPixmap(QPixmap.fromImage(self.qImg2))
        self.imageLabel1.setAlignment(Qt.AlignCenter)
        
        self.targetBox.layout().addWidget(self.imageLabel1)
    
    def exit(self):
        sys.exit()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
