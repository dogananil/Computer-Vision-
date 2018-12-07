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
from numpy.linalg import inv
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
        self.triimage = None
        self.triimage2 = None
        self.tmp_im = None
        self.imageLabel = None
        self.imageLabel2 = None
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
        self.triimage = self.image
        r = (0, 0, size[1], size[0])
        for t in triangleList :
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if self.rectcontains(r, pt1) and self.rectcontains(r, pt2) and self.rectcontains(r, pt3) :
                cv2.line(self.triimage, pt1, pt2, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.triimage, pt2, pt3, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.triimage, pt3, pt1, self.delaunay_color, 1, cv2.LINE_AA, 0)
        
        self.qImg = QImage(self.triimage.data,size[1],size[0],size[1]*3,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.inputBox.layout().addWidget(self.imageLabel)
    def draw_delaunay1(self, subdiv ) :
        triangleList = subdiv.getTriangleList();
        size = self.image2.shape
        self.triimage2 =self.image2
        r = (0, 0, size[1], size[0])

        for t in triangleList :
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            for b in range(0,len(self.bushpoints)):
                temp = (self.bushpoints[b][1],self.bushpoints[b][0])
                if(pt1==temp):
                    a = (self.arnoldpoints[b][1],self.arnoldpoints[b][0])
                if(pt2==temp):
                    c = (self.arnoldpoints[b][1],self.arnoldpoints[b][0])
                if(pt3==temp):
                    d = (self.arnoldpoints[b][1],self.arnoldpoints[b][0]) 

            
            if self.rectcontains(r, pt1) and self.rectcontains(r, pt2) and self.rectcontains(r, pt3) :
                ptt1 = pt1
                ptt2 = pt2
                ptt3 = pt3
                pt1 = a
                pt2 = c
                pt3 = d
                bmatrix1=[a[0],c[0],d[0]] # target image x's
                mmatrix = [[ptt1[0],ptt1[1],1],[ptt2[0],ptt2[1],1],[ptt3[0],ptt3[1],1]]# input image 
                bmatrix2 =[a[1],c[1],d[1]] # target image y's
                minverse = inv(mmatrix)
                amatrix1 = np.matmul(minverse,bmatrix1)
                amatrix2 = np.matmul(minverse,bmatrix2)
                amatrix = [[amatrix1[0],amatrix1[1],amatrix1[2]],[amatrix2[0],amatrix2[1],amatrix2[2]],[0,0,1]]
                targettriangle = (pt1,pt2,pt3)
                inputtriangle = (ptt1,ptt2,ptt3)
                self.morphFunc(amatrix,targettriangle,inputtriangle)

                cv2.line(self.triimage2, pt1, pt2, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.triimage2, pt2, pt3, self.delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(self.triimage2, pt3, pt1, self.delaunay_color, 1, cv2.LINE_AA, 0)
        
        self.qImg = QImage(self.triimage2.data,size[1],size[0],size[1]*3,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel1.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageLabel1.setAlignment(Qt.AlignCenter)
        self.targetBox.layout().addWidget(self.imageLabel1)
    def getbilinearpixel(self,posX,posY):
        out = []
        heightI,widthI,channelsI = self.image.shape      
    	#Get integer and fractional parts of numbers
        modXi = int(posX)
        modYi = int(posY)
        modXf = posX - modXi
        modYf = posY - modYi
        modXiPlusOneLim = min(modXi+1,self.image.shape[1]-1)
        modYiPlusOneLim = min(modYi+1,self.image.shape[0]-1)
     
    	#Get pixels in four corners
        for chan in range(self.image.shape[2]):
            bl = self.image[modYi, modXi, chan]
            br = self.image[modYi, modXiPlusOneLim, chan]
            tl = self.image[modYiPlusOneLim, modXi, chan]
            tr = self.image[modYiPlusOneLim, modXiPlusOneLim, chan]
 
		#Calculate interpolation
            b = modXf * br + (1. - modXf) * bl
            t = modXf * tr + (1. - modXf) * tl
            pxf = modYf * t + (1. - modYf) * b
            out.append(int(pxf+0.5))
        
        return out  
    def morphFunc(self,amatrix,targettriangle,inputtriangle):
        heightI,widthI,channelsI = self.image.shape
        invamatrix = inv(amatrix)
        index = 0
        coordt = []
        for a in range(0,heightI):
            for b in range(0,widthI):
                bx=targettriangle[1][0]-targettriangle[0][0]
                by=targettriangle[1][1]-targettriangle[0][1]
                cx=targettriangle[2][0]-targettriangle[0][0]
                cy=targettriangle[2][1]-targettriangle[0][1]
                x=b-targettriangle[0][0]
                y=a-targettriangle[0][1]
                d=bx*cy-cx*by
                wa = (x*(by-cy)+y*(cx-bx)+bx*cy-cx*by)/d
                wb = (x*cy-y*cx)/d
                wc = (y*bx-x*by)/d
                if wa>0 and wa<1 and wb>0 and wb<1 and wc>0 and wc<1:
                    coord = np.matmul(invamatrix,[b,a,1])
                    coordinate = (coord[0],coord[1])
                    coordt.append(coordinate)
                    
        for a in range(0,heightI):
            for b in range(0,widthI):   
                bix=inputtriangle[1][0]-inputtriangle[0][0]
                biy=inputtriangle[1][1]-inputtriangle[0][1]
                cix=inputtriangle[2][0]-inputtriangle[0][0]
                ciy=inputtriangle[2][1]-inputtriangle[0][1]
                xi=b-inputtriangle[0][0]
                yi=a-inputtriangle[0][1]
                di=bix*ciy-cix*biy
                wai = (xi*(biy-ciy)+yi*(cix-bix)+bix*ciy-cix*biy)/di
                wbi = (xi*ciy-yi*cix)/di
                wci = (yi*bix-xi*biy)/di
                if wai>0 and wai<1 and wbi>0 and wbi<1 and wci>0 and wci<1:
                    if index<len(coordt):
                        self.image[a][b]=self.getbilinearpixel(coordt[index][0],coordt[index][1])
                        index=index+1
        self.morphPrint()
        
    def morphPrint(self):
        size = self.image.shape
        self.qImg3 = QImage(self.image.data,size[1],size[0],size[1]*3,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel2 = QLabel('Result')
        self.imageLabel2.setPixmap(QPixmap.fromImage(self.qImg3))
        self.imageLabel2.setAlignment(Qt.AlignCenter)
        self.resultBox.layout().addWidget(self.imageLabel2)           
    def createTri(self):
        heightI,widthI,channelsI = self.image.shape
        self.red = self.image[:,:,2]
        self.green = self.image[:,:,1]
        self.blue = self.image[:,:,0] 
        rect1 = (0,0,widthI,heightI)
        subdiv1 = cv2.Subdiv2D(rect1)
        rect2 = (0,0,widthI,heightI)
        subdiv2 = cv2.Subdiv2D(rect2)
        self.arnoldpoints = [[5,5],[5,315],[475,5],[475,315],[290,50],[238,52],[240,8],[348,66],[404,118],[396,190],[358,230],[288,264],[224,294],[200,286],[240,258],[174,240],[90,226],[100,132],[112,56],[202,82],[200,194],[224,92],[218,194],[142,136],[202,138],[246,136],[290,136],[284,168],[296,110],[334,102],[332,142],[328,188]]
        self.bushpoints = [[5,5],[5,315],[475,5],[475,315],[324,42],[260,38],[240,8],[360,60],[416,116],[410,200],[360,240],[310,262],[258,284],[216,282],[248,260],[184,236],[92,202],[106,128],[120,52],[206,84],[200,196],[230,98],[222,186],[142,132],[198,142],[244,142],[286,146],[282,178],[286,110],[334,100],[332,152],[326,200]]
        print("HeightI",heightI)
        print("Width",widthI)
        for a in self.bushpoints:
            my_tuple = (a[1],a[0])
            subdiv1.insert(my_tuple)
            
        for b in self.arnoldpoints:
            my_tuple1 = (b[1],b[0])
            subdiv2.insert(my_tuple1)

        
        self.draw_delaunay(subdiv1)
        self.draw_delaunay1(subdiv1)
    
                    
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
