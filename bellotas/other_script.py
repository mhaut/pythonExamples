from PyQt4 import uic, QtGui, QtCore
from PyQt4.QtCore import QDir
from PyQt4.QtGui import QDialog
import numpy as np
import os, sys, inspect

local_path=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+os.sep
mainWindowsUI =  uic.loadUiType('uis'+os.sep+'mainWindows.ui')[0]  # Ui 

class mainWindows(QDialog, mainWindowsUI):
    def __init__(self, parent = None):
        super(mainWindows, self).__init__(parent)
        self.setupUi(self)
        #QtGui.QScrollBar(self.widgetImage)
        #QtGui.QScrollBar(QtCore.Qt.Horizontal,self.widgetImage)
        
        self.sceneImage = QtGui.QGraphicsScene()
        self.graphicsViewImage.setScene(self.sceneImage)
        #self.buttonLoad.clicked.connect(self.import_from_file)
        self.oveja = None
        self.widgetImage.setMinimumSize(0,0)
        self.graphicsViewImage.setMinimumSize(0,0)
        self.ancho = 650
        self.alto = 850
        self.widgetImage.resize(self.ancho,self.alto)
        self.scrollAreaWidgetContentsImage.resize(self.ancho,self.alto)
        self.scrollAreaWidgetContentsImage.update()
        self.graphicsViewImage.adjustSize()
        self.graphicsViewImage.resize(self.ancho,self.alto)
        self.zoom = 0
        self.qpixImage = None        
        #self.graphicsViewImage.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        #self.graphicsViewImage.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        #QtGui.QScrollBar(QtCore.Qt.Vertical,self.widgetImage)
        #QtGui.QScrollBar(QtCore.Qt.Horizontal,self.widgetImage)
        #self.widgetImage.addWidget(scroll)
        #self.widgetImage.setWidgetResizable(True)

        self.aux = None
        self.listaPuntos = list()
        self.pushButtonClean.clicked.connect(self.clearPolygon)
        
        self.maxScrollbar = 100
        self.minScrollbar = 0
        
        self.widgetImage.horizontalScrollBar().setValue(self.maxScrollbar)
        self.widgetImage.horizontalScrollBar().setValue(self.minScrollbar)
        
        policy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding);
        policy.expandingDirections()
        self.graphicsViewImage.setSizePolicy(policy)
        #self.sceneImage.setScaledContents(True)        
        self.import_from_file()

    
    def clearPolygon(self):
        self.listaPuntos = list()
        if self.aux != None:
            self.sceneImage.removeItem(self.aux)
            self.aux = None
            self.graphicsViewImage.update()
        else:
            # no hay poligono
            pass

    def import_from_file(self):
        #fileName = QtGui.QFileDialog.getOpenFileName(self, 'Open File', 'images', 'JPG(*.jpg);; PNG(*.png)')
        fileName = "/home/mario/software/appFotosBellota/images/006_12.jpg"
        #fileName = "/home/mario/software/appFotosBellota/images/19_5.JPG"        
        fileName = str(QDir.toNativeSeparators(fileName))
        if fileName != '':
            if ('png' in fileName) or ('jpg' in fileName) or ('PNG' in fileName) or ('JPG' in fileName):
                try:
                    self.sceneImage.removeItem(self.oveja)
                    self.qpixImage = QtGui.QPixmap.fromImage(QtGui.QImage(fileName))
                    image = QtGui.QImage(fileName)                  
                    if '6' not in fileName: # horizontal o vertical. Esto hay que cambiarlo
                        aux = self.alto
                        self.alto = self.ancho
                        self.ancho = aux
                        del aux
                        self.scrollAreaWidgetContentsImage.resize(self.ancho,self.alto)
                        self.graphicsViewImage.resize(self.ancho,self.alto)
                        #self.scrollAreaWidgetContentsImage.resize(850,650)
                        #self.graphicsViewImage.resize(850,650)
                        self.widgetImage.resize(self.ancho,self.alto)
                        #self.scrollAreaWidgetContentsImage.update()
                        #self.graphicsViewImage.update()
                        #self.widgetImage.update()
                        #self.graphicsViewImage.update()
                        #self.scrollAreaWidgetContentsImage.update()
                        #self.sceneImage.update()
                        pass
                    self.qpixImage = self.qpixImage.scaled(self.graphicsViewImage.width(),self.graphicsViewImage.height(),2)

                    self.oveja = self.sceneImage.addPixmap(self.qpixImage)
                    self.sceneImage.update()
                    
                    #self.scrollAreaWidgetContentsImage.update()
                    #self.graphicsViewImage.update()
                    #self.widgetImage.update()
                    #self.graphicsViewImage.update()
                    #self.scrollAreaWidgetContentsImage.update()
                    #self.sceneImage.update()                    
                except:
                    print "error load image"
            else:
                print "fatal error"
                exit()

    def wheelEvent(self,event):
        if event.modifiers() and QtCore.Qt.ControlModifier:
            listZoom = np.arange(1, 5, 0.5)
            if event.delta() > 0: # zoom in
                if self.zoom < len(listZoom):
                    self.zoom += 1
            else:
                if self.zoom > 0:
                    self.zoom -= 1
            self.changeViewZoom(listZoom[self.zoom])
            
    def changeViewZoom(self,zoom):
        qpixImageLocal = self.qpixImage.scaled(self.qpixImage.width()*zoom,self.qpixImage.height()*zoom,QtCore.Qt.KeepAspectRatio)
        self.scrollAreaWidgetContentsImage.resize(self.ancho*zoom,self.alto*zoom)        
        #self.widgetImage.horizontalScrollBar().setValue(self.maxScrollbar*zoom)
        #self.widgetImage.horizontalScrollBar().setValue(self.minScrollbar*zoom)
        #self.widgetImage.verticalScrollBar().setValue(self.maxScrollbar*zoom)
        #self.widgetImage.verticalScrollBar().setValue(self.minScrollbar*zoom)
        self.scrollAreaWidgetContentsImage.update()
        self.graphicsViewImage.adjustSize()
        #self.graphicsViewImage.update()    
            
        if self.oveja != None:
            self.sceneImage.removeItem(self.oveja)
        #self.graphicsViewImage.resize(qpixImageLocal.width(),qpixImageLocal.height())
        self.oveja = self.sceneImage.addPixmap(qpixImageLocal)
        if self.listaPuntos:
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(3)
            self.sceneImage.removeItem(self.aux)         
            self.aux = self.sceneImage.addPolygon(QtGui.QPolygonF([i*zoom for i in self.listaPuntos]),pen)
            self.sceneImage.update()
        self.graphicsViewImage.update()
        self.scrollAreaWidgetContentsImage.update()        

    
    def mousePressEvent(self,event):
        pen = QtGui.QPen(QtCore.Qt.red)
        pen.setWidth(3)
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() and QtCore.Qt.ShiftModifier:
                # si hemos hecho clic en el graphicview
                mousePoint = self.graphicsViewImage.mapFromParent(event.pos())
                if  self.graphicsViewImage.rect().contains(mousePoint):
                    #mousePoint = self.graphicsViewImage.mapToScene(event.pos())
                    mousePoint = self.graphicsViewImage.mapToScene(event.pos())
                    #if not self.listaPuntos:
                    #self.aux = self.sceneImage.addEllipse(mousePoint.x()-20,mousePoint.y()-50,10,10,pen,QtGui.QBrush(QtCore.Qt.red))
                    if self.aux != None:
                        self.sceneImage.removeItem(self.aux)
                    self.listaPuntos.append(QtCore.QPointF(mousePoint.x(),mousePoint.y()))
                    self.aux = self.sceneImage.addPolygon(QtGui.QPolygonF(self.listaPuntos),pen)#,QtGui.QBrush(QtCore.Qt.red))
                    self.sceneImage.update()
                    #else:
                    #self.sceneImage.removeItem(self.aux)
                        #pass                    
                    #print mousePoint
                else:
                    print "fuera"
        else:
            pass
            #needle = self.sceneImage.addPolygon(QtGui.QPolygonF(self.listaPuntos),pen)#,QtGui.QBrush(QtCore.Qt.red))
        



if __name__ == '__main__':  
   app = QtGui.QApplication(sys.argv)
   win  = mainWindows()  
   win.show()
   app.connect(app, QtCore.SIGNAL("lastWindowClosed()"), app, QtCore.SLOT("quit()"))
   app.exec_()
