# -*- coding: utf-8 -*-
from PySide import QtGui, QtCore
# from PyQt4 import QtGui
import sys
import os
import cv2
import collections
import numpy as np
sys.path.append('class/')
from autogeo import AlgorithmsGEO, UtilsGEO, GuiGEO

ALGORITHM_MATCH = "SURF"
# IMAGE_BASE = 'images/pruebaGeoDef/puerto_con_SRC.tif'
#IMAGE_QUERY = 'images/imagenOK/imagen.jpg'
# IMAGE_QUERY = 'images/pruebaGeoDef/query.png'
# IMAGE_BASE = 'images/pruebaRegion/puerto_con_SRC.tif'
# IMAGE_QUERY = 'images/pruebaRegion/query.png'
# IMAGE_BASE = 'images/puerto_con_SRC.tif'
# IMAGE_QUERY = 'images/puerto_sin_SRC.TIF'
# IMAGE_QUERY = 'images/puerto_con_SRC.tif'
# THREADS = QtCore.QThread.idealThreadCount() - 1
IMAGE_BASE = 'images/pruebaReal/modeloIrradiancia/mdi681.tif'
IMAGE_QUERY = 'images/pruebaReal/nogeo/query.png'
THREADS = 1
CROPH = 1
CROPV = 1


# 10
##########################################################################
# ALGORITHM | SUBMAP LOCATE |    TIME(s)   |     GOOD     |     TOTAL    #
#   sift    |     25/25     |      060     |     25/25    |     25/25    #
#   surf    |     25/25     |      103     |     25/25    |     25/25    #
#   obr     |     21/25     |      006     |     00/21    |     00/25    #
#  brisk    |     21/25     |      156     |     20/21    |     20/25    #
#  freak    |     18/25     |      084     |     18/25    |     18/25    #
##########################################################################



class MainThread(QtCore.QThread):
    def __init__(self, parent=super, imgBase=None, listImagesQuery=None, idthread=None, drawDescriptors=False,
                 filterpoint=False):
        QtCore.QThread.__init__(self, None)
        self.algGEO = AlgorithmsGEO()
        self.utilsGEO = UtilsGEO()
        self.algGEO.setImageBase(imgBase)
        self.listImagesQuery = listImagesQuery
        self.idthread = idthread
        self.drawDescriptors = drawDescriptors
        self.filterpoint = filterpoint

    def setImageBase(self, imgBase=None):
        self.algGEO.setImageBase(imgBase)
    
    def setImageQuery(self, imgQuery=None):
        self.algGEO.setImageQuery(imgQuery)

    def setProcessImageCounter(self, counter):
        self.cont = counter

    def setDrawDescriptors(self, drawDescriptors):
        self.drawDescriptors = drawDescriptors

    def setDrawFilterPoints(self, filterpoint):
        self.filterpoint = filterpoint

    def filterPoints(self, src_pts, dst_pts):
        # return src_pts,dst_pts
        # zones = [4,4] # [X,Y]
        # ALERT esto hay que cambiarlo por cropv y croph, lo dejo asi solo para probar sin que se creen mas de un hilo
        croh = crov = 3
        zones = [croh, crov]
        ######################################################
        maxPointPerRegion = 2
        imageBase = self.algGEO.getImageBase()
        imageQuery = self.algGEO.getImageQuery()
        q = imageQuery.shape
        q2 = imageBase.shape
        
        matrix_dic = collections.OrderedDict()
        # INICIALIZAMOS
        initX = round(1 / float(zones[0]), 2)
        initY = round(1 / float(zones[0]), 2)
        limitsX = []
        limitsY = []
        for i, j in zip(range(1, zones[0] + 1), range(1, zones[1] + 1)):
            limitsX.append(i * initX)
            limitsY.append(j * initY)

        for zoneX in range(zones[0]):
            for zoneY in range(zones[1]):
                zone_key = str(zoneX) + ", " + str(zoneY)
                matrix_dic[zone_key] = collections.OrderedDict()
                matrix_dic[zone_key]['nPuntos'] = 0
                matrix_dic[zone_key]['src_list'] = []
                matrix_dic[zone_key]['dst_list'] = []
        
        # METEMOS PUNTOS:
        for pts, pts2 in zip(src_pts, dst_pts):
            posX = round(pts[0][0] / float(q[1]), 2)
            posY = round(pts[0][1] / float(q[0]), 2)
            pos2X = round(pts2[0][0] / float(q2[1]), 2)
            pos2Y = round(pts2[0][1] / float(q2[0]), 2)

            for index in range(len(limitsX)):
                if posX < limitsX[index]:
                    break
            zoneX = index
            for index in range(len(limitsY)):
                if posY < limitsY[index]:
                    break
            zoneY = index
            
            for index in range(len(limitsX)):
                if pos2X < limitsX[index]:
                    break
            zone2X = index
            for index in range(len(limitsY)):
                if pos2Y < limitsY[index]:
                    break
            zone2Y = index

            pos = [zoneX, zoneY]
            pos2 = [zone2X, zone2Y]

            if pos == pos2:
                key_src = str(zoneX) + ", " + str(zoneY)
                key_dst = str(zone2X) + ", " + str(zone2Y)
                matrix_dic[key_src]['nPuntos'] += 1
                matrix_dic[key_src]['src_list'].append(pts)
                matrix_dic[key_dst]['dst_list'].append(pts2)
        
        # Sacamos diferencias:
        src_pts = []
        dst_pts = []
        for key in matrix_dic:
            print key, matrix_dic[key]['nPuntos']
            if matrix_dic[key]['nPuntos'] > maxPointPerRegion:
                diff = []
                for src_p, dst_p in zip(matrix_dic[key]['src_list'], matrix_dic[key]['dst_list']):
                    # ponemos en tanto por 1
                    src_p = np.array([[src_p[0][0] / float(q[1]), src_p[0][1] / float(q[0])]])
                    dst_p = np.array([[dst_p[0][0] / float(q2[1]), dst_p[0][1] / float(q2[0])]])
                    diff.append(np.linalg.norm(src_p - dst_p))

                while len(diff) > maxPointPerRegion:
                    maxIndex = diff.index(max(diff))
                    del diff[maxIndex]
                    del matrix_dic[key]['src_list'][maxIndex]
                    del matrix_dic[key]['dst_list'][maxIndex]

            if matrix_dic[key]['src_list']:
                src_pts += matrix_dic[key]['src_list']
            if matrix_dic[key]['dst_list']:
                dst_pts += matrix_dic[key]['dst_list']

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        return src_pts, dst_pts

    def run(self):
        # imgQuery = cv2.fastNlMeansDenoising(self.listImagesQuery[self.cont])
        # imgQuery = cv2.fastNlMeansDenoising(self.listImagesQuery[self.cont], 4, 21, 35)
        # cv2.imwrite('out/imagesFilter/imgFilter'+str(self.cont)+'.png',imgQuery)

        typeAlgorithm = True
        print "Executing algorithm: ", ALGORITHM_MATCH, " for image: ", self.cont, "thread: " + str(
                self.idthread) + "\n"
        
        # remove noise Image
        # imgQueryNOTnoise = cv2.fastNlMeansDenoising(self.listImagesQuery[self.cont], 4, 21, 35)

        print "removing noise"
        imgQueryNOTnoise = cv2.fastNlMeansDenoising(self.listImagesQuery[self.cont], 4, 12, 35)
        imgQueryNOTnoise2 = cv2.equalizeHist(imgQueryNOTnoise)
        print "nomalized 1"
        # cv2.imwrite('out/imagesFilter/imgFilter_1_' + str(self.cont) + '.png', imgQueryNOTnoise2)
        # imgQueryNOTnoise = cv2.fastNlMeansDenoising(imgQueryNOTnoise2, 4, 15, 35)
        # imgQueryNOTnoise2 = cv2.equalizeHist(imgQueryNOTnoise2)
        # cv2.imwrite('out/imagesFilter/imgFilter_2_' + str(self.cont) + '.png', imgQueryNOTnoise2)
        #exit()

        # imgQueryNOTnoise2 -= np.ma.median(self.algGEO.getImageBase())
        # imgQueryNOTnoise2 = imgQueryNOTnoise2
        # exit()

        # i = 0
        # print len(imgQueryNOTnoise)
        # for i in range(len(imgQueryNOTnoise)):
        # print "numero",i,"    ",imgQueryNOTnoise[i]
        # if i % 3 == 1:
        # exit()


        # imgQueryNOTnoise2 = cv2.adaptiveThreshold(imgQueryNOTnoise2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        # imgQueryNOTnoise2 = cv2.bilateralFilter(self.listImagesQuery[self.cont],9,75,75)
        # imgQueryNOTnoise2 = cv2.GaussianBlur(self.listImagesQuery[self.cont],(5,5),0)
        # imgQueryNOTnoise2 = cv2.Canny(imgQueryNOTnoise2,100,200)


        # res = np.hstack((imgQueryNOTnoise,imgQueryNOTnoise2)) #stacking images side-by-side
        # kernel = np.ones((5,5),np.uint8)
        # imgQueryNOTnoise = cv2.morphologyEx(self.listImagesQuery[self.cont], cv2.MORPH_OPEN, kernel)
        # imgQueryNOTnoiseDilate = cv2.dilate(imgQueryNOTnoise, np.ones((11, 11)))
        # kernel = np.ones((2,2),np.uint8)
        # imgQueryNOTnoiseDilate = cv2.erode(self.listImagesQuery[self.cont],kernel,iterations = 1)
        cv2.imwrite('out/imagesFilter/imgFilter' + str(self.cont) + '.png', imgQueryNOTnoise2)

        self.algGEO.setImageQuery(imgQueryNOTnoise2)
        # TODO: ELIMINAR matches porque no se usa, los metodos deberian devolver src_pts y dst_pts como variables locales. Eliminar dichas variables de clases
        if ALGORITHM_MATCH == "SURF":
            src_pts, dst_pts, kp1, kp2 = self.algGEO.algorithm_SURF(typeAlgorithm)
        elif ALGORITHM_MATCH == "SIFT":
            src_pts, dst_pts, kp1, kp2 = self.algGEO.algorithm_SIFT(typeAlgorithm)
        elif ALGORITHM_MATCH == "OBR":
            src_pts, dst_pts, kp1, kp2 = self.algGEO.algorithm_OBR(typeAlgorithm)
        elif ALGORITHM_MATCH == "BRISK":
            src_pts, dst_pts, kp1, kp2 = self.algGEO.algorithm_BRISK(typeAlgorithm)
        elif ALGORITHM_MATCH == "FREAK":
            src_pts, dst_pts, kp1, kp2 = self.algGEO.algorithm_FREAK(typeAlgorithm)
        else:
            print "Fatal error, algorithm not exist"
            exit()
        
        if self.filterpoint == True:
            src_pts, dst_pts = self.filterPoints(src_pts, dst_pts)

        if self.drawDescriptors == True:
            # ALERT esto hay que cambiarlo por cropv y croph, lo dejo asi solo para probar sin que se creen mas de un hilo
            croh = crov = 3
            #####################################################
            imgQ2 = self.algGEO.getImageQuery()
            imgB2 = self.algGEO.getImageBase()
            imgQ2 = cv2.cvtColor(imgQ2, cv2.COLOR_GRAY2RGB)
            imgB2 = cv2.cvtColor(imgB2, cv2.COLOR_GRAY2RGB)
            cont = 0
            
            tamQ = [imgQ2.shape[0] / crov, imgQ2.shape[1] / croh]
            tamB = [imgB2.shape[0] / crov, imgB2.shape[1] / croh]

            contador = 0
            contador2 = 0
            contador3 = 0
            contador4 = 0
            for i in range(1, croh):
                # horizontal
                cv2.line(imgQ2, (0, tamQ[0] * i), (imgQ2.shape[1], tamQ[0] * i), (255, 0, 0), 1, cv2.LINE_AA)
                cv2.line(imgB2, (0, tamB[0] * i), (imgB2.shape[1], tamB[0] * i), (255, 0, 0), 1, cv2.LINE_AA)
                # vertical
                cv2.line(imgQ2, (tamQ[1] * i, 0), (tamQ[1] * i, imgQ2.shape[0]), (255, 0, 0), 1, cv2.LINE_AA)
                cv2.line(imgB2, (tamB[1] * i, 0), (tamB[1] * i, imgB2.shape[0]), (255, 0, 0), 1, cv2.LINE_AA)

            for pts, pts2 in zip(src_pts, dst_pts):
                posX = round(pts[0][0] / float(imgQ2.shape[1]), 2)
                posY = round(pts[0][1] / float(imgQ2.shape[0]), 2)

                if posX == 0 and posY == 3:
                    cv2.circle(imgQ2, (pts[0][0], pts[0][1]), 4, (0, 255, 0), -1)
                else:
                    cv2.circle(imgQ2, (pts[0][0], pts[0][1]), 3, (0, 0, 255), -1)

                # ALERT COMPROBAR ESTO!!!!
                cv2.circle(imgB2, (pts2[0][0], pts2[0][1]), 3, (0, 0, 255), -1)
                # cv2.putText(imgQ2, str(cont)+"("+str(pts[0][0])+","+str(pts[0][1]), (int(pts[0][0]+5) ,int(pts[0][1])), 1, 1, (0,255,0))
                cv2.putText(imgQ2, str(cont) + "(" + str(posX) + "," + str(posY) + ")",
                            (int(pts[0][0] + 5), int(pts[0][1])), 1, 1, (0, 255, 0))
                # cv2.putText(imgQ2, str(cont)+"("+str(pts[0][0])+","+str(pts[0][1]), (int(pts[0][0]+5) ,int(pts[0][1])), 1, 1, (0,255,0))
                cv2.putText(imgB2, str(cont), (int(pts2[0][0] + 5), int(pts2[0][1])), 1, 1, (0, 255, 0))
                cont += 1

                # for pts,pts2 in zip(src_pts,dst_pts):
                # print pts[0][1],pts[0][0]
                # cv2.circle(imgQ2,(pts[0][1],pts[0][0]), 2, (0,0,255), -1)
                # cv2.circle(imgB2,(pts2[0][1],pts2[0][0]), 2, (0,0,255), -1)
                # cv2.putText(imgQ2, str(cont), (int(pts[0][1]+5) ,int(pts[0][0])), 1, 1, (0,255,0))
                # cv2.putText(imgB2, str(cont), (int(pts2[0][1]+5),int(pts2[0][0])), 1, 1, (0,255,0))
                ##cv2.putText(imgQ2, str(cont)+"("+str(pts[0][1])+","+str(pts[0][0]), (int(pts[0][1]+5) ,int(pts[0][0])), 1, 1, (0,255,0))
                ##cv2.putText(imgB2, str(cont)+"("+str(pts[0][1])+","+str(pts[0][0]), (int(pts2[0][1]+5),int(pts2[0][0])), 1, 1, (0,255,0))
                # cont += 1

            imgQ = cv2.drawKeypoints(self.algGEO.getImageQuery(), kp1, None)
            imgB = cv2.drawKeypoints(self.algGEO.getImageBase(), kp2, None)
            cv2.imwrite('out/descriptors/imageBase' + str(self.cont) + '.png', imgQ)
            cv2.imwrite('out/descriptors/imageQuery' + str(self.cont) + '.png', imgB)
            cv2.imwrite('out/puntos/imagePoints' + str(self.cont) + '.png', imgQ2)
            cv2.imwrite('out/puntos/imageQuery' + str(self.cont) + '.png', imgB2)
        if src_pts != [] and dst_pts != []:
            print "Getting points of georeference image", self.cont, "thread: " + str(self.idthread) + "\n"
            puntosComando = self.utilsGEO.emparejar([src_pts, dst_pts], 'out/datas/data' + str(self.cont) + '.points',
                                                    IMAGE_BASE)
            if puntosComando != None:
                self.utilsGEO.geoReference(puntosComando, (str(self.cont)))
        else:
            print "Not geoReferencing image" + str(self.cont) + '\n'


class MyFirstScene():
    def __init__(self, parent=None):
        app = QtGui.QApplication(sys.argv)
        main_class = GuiGEO()
        ############################################################
        ############################################################
        ############################################################
        self.testingAPP()
        ############################################################
        ############################################################
        ############################################################
        main_class.show()
        app.exec_()
        sys.exit(app.exec_())
        pass

    def testingAPP(self):
        imgBase = cv2.imread(IMAGE_BASE, cv2.IMREAD_GRAYSCALE)

        # imgBase = cv2.Canny(imgBase,100,200)
        imgQuery = cv2.imread(IMAGE_QUERY, cv2.IMREAD_GRAYSCALE)

        listImagesQuery = UtilsGEO().crop(imgQuery, CROPH, CROPV)
        print "numero de imagenes", len(listImagesQuery)

        allThreads = list(None for i in range(THREADS))
        for i in range(THREADS):
            allThreads[i] = MainThread(imgBase=imgBase, listImagesQuery=listImagesQuery, idthread=i,
                                       drawDescriptors=True, filterpoint=True)

        cont = 0
        finish = False
        while finish == False:
            if cont == len(listImagesQuery):
                finish = True
            else:
                for i in range(THREADS):
                    if allThreads[i].isRunning() == False:
                        print "Arrancando hilo", i, "imagen", cont
                        allThreads[i].setProcessImageCounter(cont)
                        allThreads[i].start()
                        cont += 1

        for i in range(len(allThreads)):
            while (allThreads[i].isRunning()): pass
            print "hilo terminado", i

        if os.path.exists("temp_commands.sh"):
            os.system("sh temp_commands.sh; rm temp_commands.sh")
        else:
            print "No images to georeference"
        exit()


if __name__ == "__main__":
    a = MyFirstScene()
    # a.testingAPP()
    # app = QtGui.QApplication(sys.argv)
    # main_class = GuiGEO()
    # main_class.show()
    # sys.exit(app.exec_())
