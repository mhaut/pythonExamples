import cv2
import os
import sys
import numpy as np
from math import sqrt 
from osgeo import gdal, osr
from PySide import QtCore
from PyQt4 import uic,QtGui
from matplotlib import pyplot as plt




formclass =  uic.loadUiType('..'+os.sep+'uis'+os.sep+'main.ui')[0]  # Ui
#form_class =  uic.loadUiType('..{0}uis{1}main.ui'.format(os.sep, os.sep))[0]  # Ui

class GuiGEO(QtGui.QMainWindow, formclass):
    
    def __init__(self, parent=None):     
        """
        -------------------------
        :param:
        :return:
        """
        super(GuiGEO, self).__init__(parent)
        self.setupUi(self) 
        
        self.sceneImageQuery = QtGui.QGraphicsScene()
        self.sceneImageComplete = QtGui.QGraphicsScene()
        self.graphicsViewImageQuery.setScene(self.sceneImageQuery)
        self.graphicsViewImageComplete.setScene(self.sceneImageComplete)
        self.oveja = None
        self.becerro = None
        
        self.actionOpenAbrir.triggered.connect(self.abrirComplete)
        self.actionOpenCompare.triggered.connect(self.abrirQuery)
        self.actionOBR.triggered.connect(self.algorithm_OBR)
        self.actionSURF.triggered.connect(self.algorithm_SURF)
        self.actionSIFT.triggered.connect(self.algorithm_SIFT)


    def abrir(self):
        qpixmap = ''
        try:
            currentDir = os.getcwd()+'/images'
            filters = "Images (*.png *.tif *.jpg)"
            selected_filter = "Images (*.png *.tif *.jpg)"
            filename = QtGui.QFileDialog.getOpenFileName(self, " File dialog ", currentDir, filters, selected_filter)
            if filename:
                img = cv2.imread(str(filename),0)
                filename = filename.split('/')
                name = "images/"+filename[len(filename)-1]
                qimage = QtGui.QImage(name)
                qpixmap = QtGui.QPixmap.fromImage(qimage)
            else:
                QtGui.QMessageBox.information(self, 'Info Message', ''' No image selected ''', QtGui.QMessageBox.Ok)
        except:
            print "sheep"
            return None        
        return qpixmap, img


    def abrirComplete(self):
        qpixmap, img = self.abrir()
        if qpixmap:
            self.imgBase = img
            if self.becerro != None:
                self.sceneImageComplete.removeItem(self.becerro)        
            self.becerro = self.sceneImageComplete.addPixmap(qpixmap)
            self.sceneImageComplete.update()

    def abrirQuery(self):
        qpixmap, img = self.abrir()
        if qpixmap:
            self.imgQuery = img
            if self.oveja != None:
                self.sceneImageQuery.removeItem(self.oveja)
            self.oveja = self.sceneImageQuery.addPixmap(qpixmap)
            self.sceneImageQuery.update()


if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)

    main_class = GuiGEO()
    main_class.show()
    sys.exit(app.exec_())