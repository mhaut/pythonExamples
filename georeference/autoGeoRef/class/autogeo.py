# -*- coding: utf-8 -*-

# import thautogeo
# import geogui
import cv2
import os
import numpy as np
from math import sqrt
from osgeo import gdal, osr
# from PySide import QtCore
from PyQt4 import uic, QtGui, QtCore
from PyQt4 import QtCore as pyqtcore
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 2  # minimo numero de puntos georeferenciacion vecinos y lineal
form_class = uic.loadUiType('uis' + os.sep + 'main.ui')[0]  # Ui
form_class_algorithm = uic.loadUiType('uis' + os.sep + 'algorithm.ui')[0]  # Ui


class AlgorithmsGEO(QtCore.QThread):
    def __init__(self, parent=super):
        QtCore.QThread.__init__(self, None)

    def setImageQuery(self, img):
        self.imgQuery = img

    def setImageBase(self, img):
        self.imgBase = img

    def getImageBase(self):
        return self.imgBase

    def getImageQuery(self):
        return self.imgQuery

    # Oriented FAST and Rotate Brief
    def algorithm_OBR(self, flann):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(self.imgQuery, None)
        kp2, des2 = orb.detectAndCompute(self.imgBase, None)
        if flann == True:
            src_pts, dst_pts = self.matchImagesFlannLSH(kp1, kp2, des1, des2)
        else:
            # TODO: hacer esto!!!
            src_pts, dst_pts = None
        return src_pts, dst_pts, kp1, kp2




    def algorithm_SURF(self, flann):
        surf = cv2.xfeatures2d.SURF_create(nOctaves=3, nOctaveLayers=2, extended=True, upright=True)
        kp1, des1 = surf.detectAndCompute(self.imgQuery, None)
        kp2, des2 = surf.detectAndCompute(self.imgBase, None)
        if flann == True:
            src_pts, dst_pts = self.matchImagesFlann(kp1, kp2, des1, des2)
        else:
            src_pts, dst_pts = self.matchImagesBruteForce(kp1, kp2, des1, des2)
        return src_pts, dst_pts, kp1, kp2







    # def algorithm_SURF(self, flann):
    #     # Initiate SIFT detector
    #     # surf = cv2.xfeatures2d.SURF_create()
    #     # Descriptores de 128 elementos
    #     # (double hessianThreshold=100, int nOctaves=4, int nOctaveLayers=3, bool extended=false, bool upright=false)
    #     #cv2.xfeatures2d.SURF_create()
    #     # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=500, extended=False, upright=True)
    #     # Como las imágenes tienen la misma orientación, upright lo ponemos a true para que no tenga en cuenta las rotaciones
    #     list1 = list()
    #     list2 = list()
    #     list3 = list()
    #     list4 = list()
    #     list5 = list()
    #     list6 = list()
    #     locateGoodValue=False
    #     i = 0
    #     kp1old = None
    #     # outfile = open('texto.txt', 'w') # Indicamos el valor 'w'.
    #     # gradiente de descenso hasta encontrar una convergencia menor del 2%
    #     gradienteMin = 0.02
    #     allDecre=1
    #     while locateGoodValue is False:
    #         surf = cv2.xfeatures2d.SURF_create(nOctaves=3, nOctaveLayers=2, hessianThreshold=i, extended=True, upright=True)
    #         # surf = cv2.xfeatures2d.SURF_create(extended = True, nOctaves=8, nOctaveLayers=4, upright=False )
    #         # orientación arriba si o no. Si es si, entonces no se tiene en cuenta
    #         # find the keypoints and descriptors with SIFT
    #         kp1, des1 = surf.detectAndCompute(self.imgQuery, None)
    #         kp2, des2 = surf.detectAndCompute(self.imgBase, None)
    #
    #         list1.append(len(kp1))
    #         list2.append(len(kp2))
    #         # decre indica en tanto por 1 lo que decrece el número de descriptores con respecto a la etapa anterior
    #         # establecemos 0.2 como valor de corte
    #         if kp1old != None:
    #             # valor de la función
    #             decre = (len(kp1old) - len(kp1)) / float(len(kp1old))
    #             # print decre
    #             allDecre -= decre
    #             # print oldDecre
    #             if decre < 0.04:
    #                 locateGoodValue = True
    #                 print "Remove ",round((1-allDecre)*100,0),"% of keypoints\n"
    #             list4.append(decre)
    #         else:
    #             list4.append(1)
    #         kp1old = kp1
    #         list3.append(i)
    #         i += 50
    #
    #     if flann == True:
    #         src_pts, dst_pts = self.matchImagesFlann(kp1, kp2, des1, des2)
    #     else:
    #         src_pts, dst_pts = self.matchImagesBruteForce(kp1, kp2, des1, des2)
    #     # plt.plot(list1),plt.ylim(0,2000),plt.show()
    #     # plt.plot(list2),plt.ylim(0,2000),plt.show()
    #     # plt.plot(list3),plt.ylim(0,2000),plt.show()
    #
    #     # plt.plot(list4),plt.ylim(0,1),plt.xticks(np.arange(min(list4), max(list4)+1, 1.0)),plt.show()
    #     # plt.plot(list5),plt.show()
    #     # plt.plot(list6),plt.show()
    #     aux = list()
    #     for i in range(len(list5)):
    #         aux.append(i)
    #     # xnew = np.linspace(0, 120, 40)
    #     # from scipy.interpolate import interp1d, UnivariateSpline
    #     from sympy import roots, solve_poly_system
    #     # f2 = interp1d(aux, list5, kind='cubic')
    #     # spl = UnivariateSpline(xnew,f2(xnew), k=4, s=0)
    #     # listaAppend = list()
    #     # for x in xnew:
    #     #     if round(spl(f2(x))-f2(x),1) == 0:
    #     #         print "corte en",x
    #     #     listaAppend.append(x)
    #     # print listaAppend
    #     # exit()
    #
    #     # print "las rectas se cortan en: ",solve(spl(f2(x))-f2(x),x)
    #     # print "F(X)-F'(X) =",solve(spl - f2)
    #     # TODO deberíamos hacer un filtro cuando f'(x) = f(x) averiguar el motivo
    #     # plt.plot(list5,'o', xnew,f2(xnew),'-', xnew,spl(f2(xnew)),'--'),plt.ylim(0,100),plt.show()
    #     # plt.plot(list5,'o', xnew,f2(xnew),'-', xnew,splev(xnew,spl(xnew),der=1),'--'),plt.show()
    #
    #     return src_pts, dst_pts, kp1, kp2

    def algorithm_BRISK(self, flannLSH):
        # Initiate BRISK detector
        brisk = cv2.BRISK_create(thresh=10, octaves=1)
        # orientación arriba si o no. Si es si, entonces no se tiene en cuenta
        # find the keypoints and descriptors with SIFT
        kp1, des1 = brisk.detectAndCompute(self.imgQuery, None)
        kp2, des2 = brisk.detectAndCompute(self.imgBase, None)
        if flannLSH == True:
            src_pts, dst_pts = self.matchImagesFlannLSH(kp1, kp2, des1, des2)
        else:
            src_pts, dst_pts = self.matchImagesBruteForce(kp1, kp2, des1, des2)
        return src_pts, dst_pts, kp1, kp2

    def algorithm_FREAK(self, flannLSH):
        surfDetector = cv2.xfeatures2d.SIFT_create()
        kp1 = surfDetector.detect(self.imgQuery, None)
        kp2 = surfDetector.detect(self.imgBase, None)
        freakExtractor = cv2.xfeatures2d.FREAK_create()
        kp1, des1 = freakExtractor.compute(self.imgQuery, kp1)
        kp2, des2 = freakExtractor.compute(self.imgBase, kp2)
        del freakExtractor
        if flannLSH == True:
            src_pts, dst_pts = self.matchImagesFlannLSH(kp1, kp2, des1, des2)
        else:
            src_pts, dst_pts = self.matchImagesBruteForce(kp1, kp2, des1, des2)
        return src_pts, dst_pts, kp1, kp2

    # SIFT(int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)
    # nfeatures – The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    # nOctaveLayers – The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    # contrastThreshold – The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
    # edgeThreshold – The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    # sigma – The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    def algorithm_SIFT(self, flann):
        # Initiate SIFT detector
        ##sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=120)
        
        # SEGUN ARTICULO, CUANTA MAS RESOLUCION TENGA LA IMAGEN MAYOR DEBE SER EL NUMERO DE nOctaveLayers
        # EL CONTRASTE ES JUSTO ESO, CONTRASTE CON SUS VECINOS. PARA EVITAR CAMBIOS DE LUMINOSIDAD. TENEMOS IMAGENES PARECIDAS. ENTONCES ESTO DEBERÍA SER BAJO
        # EL edgeThreshold
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.015, edgeThreshold=9, nOctaveLayers=3)
        # sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.imgQuery, None)
        kp2, des2 = sift.detectAndCompute(self.imgBase, None)
        if flann == True:
            src_pts, dst_pts = self.matchImagesFlann(kp1, kp2, des1, des2)
        else:
            src_pts, dst_pts = self.matchImagesBruteForce(kp1, kp2, des1, des2)
        return src_pts, dst_pts, kp1, kp2


        # def algorithm_SIFT(self, flann):
        # print "iiiii"
        # dense = cv2.xfeatures2d.Dense_create()
        ##dense=cv2.FeatureDetector_create("Dense")
        # kp1=dense.detect(self.imgQuery)
        # kp,des=sift.compute(self.imgQuery,kp)
        # exit()

    def matchImagesBruteForce(self, kp1, kp2, des1, des2):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        matches = sorted(matches, key=lambda x: x.distance)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        return good

    def matchImagesFlann(self, kp1, kp2, des1, des2):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            if len(good) > 10:
                # TODO: PONER UN FILTRO
                pass
            # en src_pts y dst_pts están las coordenadas pixel de la imagen origen y destino
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        else:
            print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            src_pts = []
            dst_pts = []
        return src_pts, dst_pts

    def matchImagesFlannLSH(self, kp1, kp2, des1, des2):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=12,  # 12
                            key_size=20,  # 20
                            multi_probe_level=2)  # 2
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.              
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m, n) = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            # en src_pts y dst_pts están las coordenadas pixel de la imagen origen y destino
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        else:
            print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            dst_pts = src_pts = []
        return src_pts, dst_pts


class UtilsGEO(QtCore.QThread):
    def __init__(self, parent=super):
        QtCore.QThread.__init__(self, None)

    def pixel2Coordinate(self, geotifAddr, pixelPairs):
        # Load the image dataset
        ds = gdal.Open(geotifAddr)
        # Get a geo-transform of the dataset
        gt = ds.GetGeoTransform()
        # Create a spatial reference object for the dataset
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        # Set up the coordinate transformation object
        srsLatLong = srs.CloneGeogCS()
        ct = osr.CoordinateTransformation(srs, srsLatLong)
        # Go through all the point pairs and translate them to pixel pairings
        latLonPairs = []
        for point in pixelPairs:
            # Translate the pixel pairs into untranslated points
            ulon = point[0][0] * gt[1] + gt[0]
            ulat = point[0][1] * gt[5] + gt[3]
            # Transform the points to the space
            # (lon,lat,holder) = ct.TransformPoint(ulon,ulat)
            # Add the point to our return array
            latLonPairs.append([ulat, ulon])
        return latLonPairs

        ###############def crop(self,image,numImagesH,numImagesW,lenMaxList):
        ###############height, width = image.shape   # Get dimension
        ###############listImages = []
        ###############cropH = height / numImagesH
        ###############cropW = width / numImagesW
        ###############initH = 0
        ###############finishH = cropH
        ###############for i in range (0,numImagesH):
        ###############initW = 0
        ###############finishW = cropW
        ###############for j in range (0,numImagesW):
        ###############listImages.append(image[initH : finishH, initW : finishW])
        ###############initW = finishW
        ###############finishW += cropW
        ###############initH = finishH
        ###############finishH += cropH
        ###############cont = 0
        ###############ListofListImages = []
        ###############print "-------",lenMaxList
        ################exit()
        ###############for i in range(lenMaxList):
        ###############ListofListImages.append([])
        ###############for image in listImages:
        ###############ListofListImages[cont%lenMaxList].append(image)
        ###############cv2.imwrite('out/images/im'+str(cont)+'_th'+str(cont/lenMaxList)+'.png',image)
        ###############cont += 1
        ################ListofListImages = map(list,map(None,*ListofListImages))
        ###############return ListofListImages

    def crop(self, image, numImagesH, numImagesW):
        height, width = image.shape  # Get dimension
        listImages = []
        cropH = height / numImagesH
        cropW = width / numImagesW
        initH = 0
        finishH = cropH
        for i in range(0, numImagesH):
            initW = 0
            finishW = cropW
            for j in range(0, numImagesW):
                listImages.append(image[initH: finishH, initW: finishW])
                initW = finishW
                finishW += cropW
            initH = finishH
            finishH += cropH
        cont = 0
        for image in listImages:
            cv2.imwrite('out/images/im' + str(cont) + '.png', image)
            cont += 1
        return listImages

    def euclideanDistance(self, p1, p2):
        return sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def filterPoints(self, kp1, points):
        return points
        # # src_pts es la imagen base
        # src_pts = np.float32([kp1[point.queryIdx].pt for point in points]).reshape(-1, 1, 2)
        # allDistances = [[None] * len(src_pts) for i in range(len(src_pts))]
        # finish = False
        # cont = 0
        # aux = 0
        # for i in range(0, len(src_pts)):
        #     for j in range(i, len(src_pts) - 1):
        #         allDistances[i][j] = self.euclideanDistance(src_pts[j][0], src_pts[j + 1][0])
        #         cont += 1
        #         aux += allDistances[i][j]
        # mean = aux / float(cont)
        # cont = 0
        # for i in range(len(src_pts)):
        #     for j in range(len(src_pts)):
        #         if allDistances[i][j] is not None:
        #             if mean <= allDistances[i][j]:
        #                 allDistances[i][j] = None
        #                 print len(points), cont
        #                 if cont < len(points):
        #                     del points[cont]
        #                     cont -= 1
        #             else:
        #                 cont += 1
        #         else:
        #             pass
        # return points

    # def emparejar(self, geoTransform, matchesPoints, filename):
    def emparejar(self, matchesPoints, filename, fileCoordinates):
        src_pts = matchesPoints[0]
        dst_pts = matchesPoints[1]
        if src_pts == None and dst_pts == None:
            # QtGui.QMessageBox.information(self, 'Info Message', ''' No match found! '''+str(filename), QtGui.QMessageBox.Ok)
            print  'No match found! ' + str(filename)
            return None
        else:
            puntosComando = ''
            outfile = open(filename, 'w')
            # enable siempre a 1
            outfile.write("mapX,mapY,pixelX,pixelY,enable\n")
            listLatLon = self.pixel2Coordinate(fileCoordinates, dst_pts)
            
            listlengh = []
            cont = 0
            mean = 0
            for g in listLatLon:
                suma = sum(g)
                listlengh.append(suma)
                mean += suma
            mean = mean / len(listLatLon)
            
            for i in range(len(listLatLon), 0):
                if listLatLon[i] > mean:
                    listLatLon.pop(i)
                    src_pts.pop(i)
                    dst_pts.pop(i)
            # exit()
            listSTD = []
            for geoLoc in zip(listLatLon):
                g = np.array(geoLoc[0])
                # listSTD.append(np.linalg.norm(g,2)) # norma 2
                listSTD.append(np.std(g))  # desviacion estandar

            listSTD = np.array(listSTD)
            listSTD = listSTD - np.mean(listSTD)
            mean = np.mean(np.absolute(listSTD))
            listremove = []
            for i in range(0, len(listSTD)):
                if abs(listSTD[i]) >= mean * 2:
                    listremove.append(i)
            listLatLon = np.delete(listLatLon, listremove, axis=0)
            src_pts = np.delete(src_pts, listremove, axis=0)

            for geoLoc, point in zip(listLatLon, src_pts):
                # print str(geoLoc[1])+","+str(geoLoc[0])+","+str(point[0][0])+","+str(point[0][1])
                # TODO: Revisar el simbolo negativo
                outfile.write(
                    str(geoLoc[1]) + "," + str(geoLoc[0]) + "," + str(point[0][0]) + "," + str(-point[0][1]) + ",1\n")
                puntosComando += "-gcp " + str(point[0][0]) + " " + str(point[0][1]) + " " + str(geoLoc[1]) + " " + str(
                        geoLoc[0]) + " "
            outfile.close()
            return puntosComando

    def geoReference(self, puntosComando, idImage):
        import subprocess
        # print "geoReferencing image",numberImage
        # (str(cont)+"_th"+str(self.idthread))
        comando1 = 'gdal_translate -of GTiff ' + puntosComando + '"out/images/im' + idImage + '.png" "/tmp/im' + idImage + '.png"'  # 1>/dev/null'
        # TODO añadir mas métodos.
        comando2 = 'gdalwarp -r near -order 1 -co COMPRESS=NONE  "/tmp/im' + idImage + '.png" "out/imagesGeoref/im' + idImage + '_modified.tif"'  # 1>/dev/null'
        # comando2 = 'gdalwarp -r bilinear -order 2 -co COMPRESS=NONE  "/tmp/im'+idImage+'.png" "/home/pbustos/software/surfSift/app/out/imagesGeoref/im'+idImage+'_modified.tif"'# 1>/dev/null'


        outfile = open("temp_commands.sh", "a")
        # enable siempre a 1
        outfile.write(comando1 + "; " + comando2 + "; ")
        outfile.close
        # os.system(comando)


##################
#### GUI classes
##################
class AlgorithmsWindowGEO(QtGui.QDialog, form_class_algorithm):
    def __init__(self, parent=None):
        # QtGui.QWidget.__init__(self, parent)
        super(AlgorithmsWindowGEO, self).__init__(parent)
        self.setupUi(self)
        
        self.spinBoxThreads.setRange(1, QtCore.QThread.idealThreadCount())
        self.spinBoxCropW.setMinimum(1)
        self.spinBoxCropH.setMinimum(1)

        self.setWindowTitle("Algorithm Window")
        
        self.comboBoxAlgorithm.addItem("SIFT")
        self.comboBoxAlgorithm.addItem("SURF")
        self.comboBoxAlgorithm.addItem("OBR")
        self.comboBoxAlgorithm.addItem("FREAK")
        
        self.pushButtonAcepted.clicked.connect(self.accepted_function)
        self.pushButtonCancel.clicked.connect(self.cancel_function)
        self.pushButtonAcepted.setEnabled(True)
        self.accepted = False  # flag to accept button
        self.data = {'typeAlgorithm': None, 'threads': None, 'cropZones': list(), 'options': list()}

    def accepted_function(self):
        self.accepted = True
        self.close()

    def cancel_function(self):
        self.close()

    def loadAlgOptions(self):
        self.exec_()
        if self.accepted is True:
            self.data['typeAlgorithm'] = str(self.comboBoxAlgorithm.currentText())
            self.data['threads'] = self.spinBoxThreads.value()
            self.data['cropZones'] = [self.spinBoxCropW.value(), self.spinBoxCropH.value()]
            self.data['options'] = [self.checkBoxDescriptors.isChecked(), self.checkBoxKeyPoints.isChecked(),
                                    self.checkBoxPoints.isChecked(), self.checkBoxImagesInit.isChecked()]
            return self.data
        else:
            return None


class myGraphicsSceneImageBase(QtGui.QGraphicsScene):
    def __init__(self, parent):
        super(myGraphicsSceneImageBase, self).__init__()
        self.parent = parent

    def mouseMoveEvent(self, mouseEvent):
        position = mouseEvent.scenePos()
        # obtengo el punto en coordenadas pixeles
        position = self.parent.graphicsViewImageComplete.mapFromScene(position)
        if mouseEvent.buttons() == pyqtcore.Qt.NoButton:
            if self.parent.filename != None:
                self.parent.labelBase.setText(str(position.x()) + ", " + str(position.y()))
                position = self.parent.getlocalizationPoint([position.x(), position.y()])
                self.parent.labelQuery.setText(str(round(position[1])) + ", " + str(round(position[0])))
        elif mouseEvent.buttons() == pyqtcore.Qt.LeftButton:
            self.parent.labelBase.setText("Left click drag")
        elif mouseEvent.buttons() == pyqtcore.Qt.RightButton:
            self.parent.labelBase.setText("Right click drag")


class myGraphicsSceneQuery(QtGui.QGraphicsScene):
    def __init__(self, parent):
        super(myGraphicsSceneQuery, self).__init__()
        self.parent = parent
        self.buttons = None
        self.posRect = []
        self.newRect = None

    def mouseReleaseEvent(self,mouseEvent):
        ##################################################################
        ##################################################################
        ##################################################################
        ####################################################################################################################################
        ##################################################################
        ####################################################################################################################################

        if mouseEvent.buttons() == pyqtcore.Qt.LeftButton:
            ##################################################################
            ##################################################################
            ##################################################################
            ##################################################################
            ##################################################################
            ##################################################################
            ##################################################################
            position = mouseEvent.scenePos()
            position = self.parent.graphicsViewImageQuery.mapFromScene(position)
            self.posRect.append([position.x(),position.y()])
            x1 = self.posRect[0][0] * self.parent.proportion[0]
            y1 = self.posRect[0][1] * self.parent.proportion[1]
            x2 = self.posRect[1][0] * self.parent.proportion[0]
            y2 = self.posRect[1][1] * self.parent.proportion[1]
            if x2 - x1 < 0:
                aux = x2
                x2 = x1
                x1 = aux
            xDistance = x2 - x1
            if y2 - y1 < 0:
                aux = y2
                y2 = y1
                y1 = aux
            yDistance = y2 - y1

            if self.newRect != None:
                self.removeItem(self.newRect)
            pen = QtGui.QPen()
            pen.setStyle(pyqtcore.Qt.DashDotLine);
            pen.setWidth(8)
            pen.setBrush(QtGui.QColor(255, 0, 0))
            pen.setCapStyle(pyqtcore.Qt.RoundCap)
            pen.setJoinStyle(pyqtcore.Qt.RoundJoin)
            # el último parámetro del color es la transparencia
            brush = QtGui.QBrush(QtGui.QColor(255, 0, 0, 87))
            self.newRect = self.addRect(x1, y1, xDistance, yDistance, pen, brush)
            self.roi = clone[x1:xDistance, y1:yDistance]
            # print "ratón soltado en" + str(position.x()) + ", " + str(position.y())
        else:
            print "eeeeeeeee2"

    def mousePressEvent(self,mouseEvent):
        if mouseEvent.buttons() == pyqtcore.Qt.RightButton and self.posRect != []:
            if self.newRect != None:
                self.removeItem(self.newRect)
                self.posRect = []
                self.newRect = None
            else:
                print "ERROR GRAVE!!!!!!!!!!!!!!!!!!"
        elif mouseEvent.buttons() == pyqtcore.Qt.LeftButton:
            if self.posRect != []:
                self.posRect = []
                position = mouseEvent.scenePos()
                position = self.parent.graphicsViewImageQuery.mapFromScene(position)
                self.posRect.append([position.x(),position.y()])
                # print "ratón pulsado en" + str(position.x()) + ", " + str(position.y())
            else:
                pass


    def mouseMoveEvent(self, mouseEvent):
        pass
        # position = mouseEvent.scenePos()
        # if mouseEvent.buttons() == pyqtcore.Qt.LeftButton:
        # obtengo el punto en coordenadas pixeles
        # position = self.parent.graphicsViewImageComplete.mapFromScene(position)

        # if mouseEvent.buttons() == pyqtcore.Qt.NoButton:
        #     self.parent.labelQuery.setText(str(position.x()) + ", " + str(position.y()))
        # elif mouseEvent.buttons() == pyqtcore.Qt.LeftButton:
        #     print "<---------->"
        #     print self.parent.proportion
        #     # self.addRect(mouseEvent.scenePos().x(), mouseEvent.scenePos().y(), 10, 10)
        #    # if mouseEvent.buttons() == pyqtcore.Qt.Button
        #     # self.parent.labelQuery.setText("Left click drag")
        # elif mouseEvent.buttons() == pyqtcore.Qt.RightButton:
        #     self.parent.labelQuery.setText("Right click drag")


class GuiGEO(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        """
        -------------------------
        :param:
        :return:
        """
        super(GuiGEO, self).__init__(parent)
        self.setupUi(self)
        
        # self.sceneImageComplete = QtGui.QGraphicsScene()
        self.sceneImageComplete = myGraphicsSceneImageBase(self)
        self.sceneImageQuery = myGraphicsSceneQuery(self)
        self.graphicsViewImageQuery.setScene(self.sceneImageQuery)
        self.graphicsViewImageComplete.setScene(self.sceneImageComplete)
        self.oveja = None
        self.becerro = None
        
        self.actionOpenAbrirBase.triggered.connect(self.abrirComplete)
        self.actionOpenAbrirQuery.triggered.connect(self.abrirQuery)
        self.actionOpenProcess.triggered.connect(self.algorithmsOptions)
        self.imgQuery = None
        self.imgBase = None
        
        self.filename = None
        self.proportion = 0
        self.sceneImageQuery.sceneRect()
        self.graphicsViewImageQuery.sceneRect()
        self.update()



    def algorithmsOptions(self):
        winalg = AlgorithmsWindowGEO(self)  # Raise up the window
        data_algorithm = winalg.loadAlgOptions()
        
        if data_algorithm != None:
            print data_algorithm['typeAlgorithm']
            print data_algorithm['threads']
            print data_algorithm['cropZones']
            print data_algorithm['options']

            # print "a", self.qpixmapBase.size()
            # print "c", self.graphicsViewImageComplete.width(), self.graphicsViewImageComplete.height()

    def scalePixmap(self):
        # qpixmap = self.qpixmapBase
        # width = self.graphicsViewImageComplete.width()
        # height = self.graphicsViewImageComplete.height()
        qpixmap = self.qpixmapQuery
        width = self.graphicsViewImageQuery.width()
        height = self.graphicsViewImageQuery.height()
        scaledW = qpixmap.size().width() / width
        scaledH = qpixmap.size().height() / height
        # WARNING si es más ancha que alta
        if qpixmap.size().width() > qpixmap.size().height():
            # QtGui.QMessageBox.information(self, 'Info Message', ''' Aspect Ration Warning ''', QtGui.QMessageBox.Ok)
            pass  # TODO: Aspect Warning in load
        correction = 10
        if scaledW > 0:
            scaledW = qpixmap.size().width() / scaledW - correction
        else:
            scaledW = width
        if scaledH > 0:
            scaledH = qpixmap.size().height() / scaledH - correction
        else:
            scaledH = height
        qpixmap = qpixmap.scaled(pyqtcore.QSize(scaledW, scaledH), pyqtcore.Qt.IgnoreAspectRatio)
        return qpixmap

    def getProportionPixmap(self):
        qpixmap = self.qpixmapBase
        width = self.graphicsViewImageComplete.width()
        height = self.graphicsViewImageComplete.height()
        scaledW = qpixmap.size().width() / width
        scaledH = qpixmap.size().height() / height
        if scaledW > 0:
            scaledW = qpixmap.size().width() / scaledW
        else:
            scaledW = width
        if scaledH > 0:
            scaledH = qpixmap.size().height() / scaledH
        else:
            scaledH = height
        return scaledW, scaledH

    def showPixmapBase(self, qpixmap, Zone):
        if Zone == True:  # Imagen base
            if self.becerro != None:
                self.sceneImageComplete.removeItem(self.becerro)
            self.becerro = self.sceneImageComplete.addPixmap(
                qpixmap.scaledToWidth(self.graphicsViewImageComplete.geometry().width()))
            self.graphicsViewImageComplete.fitInView(self.becerro)
            self.sceneImageComplete.update()
        else:
            if self.oveja != None:
                self.sceneImageQuery.removeItem(self.oveja)
            self.oveja = self.sceneImageQuery.addPixmap(
                qpixmap.scaledToWidth(self.graphicsViewImageQuery.geometry().width()))
            self.graphicsViewImageQuery.fitInView(self.oveja)
            self.sceneImageQuery.update()

    def abrir(self):
        qpixmap = ''
        # try:
        currentDir = os.getcwd() + '/images'
        filters = "Images (*.png *.tif *.jpg)"
        selected_filter = "Images (*.png *.tif *.jpg)"
        self.filename = QtGui.QFileDialog.getOpenFileName(self, " File dialog ", currentDir, filters, selected_filter)
        if self.filename:
            img = cv2.imread(str(self.filename), 0)
            # filename = filename.split('/')
            # name = "images/"+filename[len(filename)-1]
            # print filename
            qimage = QtGui.QImage(self.filename)
            qpixmap = QtGui.QPixmap.fromImage(qimage)
            # width = self.graphicsViewImageComplete.geometry().width()
            # height = self.graphicsViewImageComplete.geometry().height()
            # qpixmap = self.scalePixmap(qpixmap,width,height)
        else:
            QtGui.QMessageBox.information(self, 'Info Message', ''' No image selected ''', QtGui.QMessageBox.Ok)
            # except:
        # print "sheep"
        # return None
        return qpixmap, img

    def initInfoCoordinates(self):
        # Load the image dataset
        ds = gdal.Open(str(self.filename))
        # Get a geo-transform of the dataset
        self.gt = ds.GetGeoTransform()
        # Create a spatial reference object for the dataset
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        # Set up the coordinate transformation object
        srsLatLong = srs.CloneGeogCS()
        self.ct = osr.CoordinateTransformation(srs, srsLatLong)
        # Go through all the point pairs and translate them to pixel pairings
        latLonPairs = []
        return (sum(self.gt) > 2)

    def getlocalizationPoint(self, point):
        # Translate the pixel pairs into untranslated points
        ulon = point[0] * self.gt[1] + self.gt[0]
        ulat = point[1] * self.gt[5] + self.gt[3]
        # Transform the points to the space
        (lon, lat, holder) = self.ct.TransformPoint(ulon, ulat)
        print lon, lat, holder
        # Add the point to our return array
        return [ulat, ulon]
        # return latLonPairs

        # def abrirComplete(self):
        # qpixmap, img = self.abrir()
        # if self.initInfoCoordinates():
        # if qpixmap:
        # self.imgBase = img
        # self.showPixmapBase(qpixmap, True)
        # self.qpixmapBase = qpixmap
        # else:
        # QtGui.QMessageBox.information(self, 'Info Message', ''' No geoReference image''', QtGui.QMessageBox.Ok)

        # def abrirQuery(self):
        # qpixmap, img = self.abrir()
        # if qpixmap:
        # self.imgQuery = img
        # self.showPixmapBase(qpixmap, False)

    def abrirComplete(self):
        qpixmap, img = self.abrir()
        self.qpixmapBase = qpixmap
        self.proportion = self.getProportions()
        qpixmap = self.scalePixmap()
        if qpixmap:
            self.imgBase = img
        if self.becerro != None:
            self.sceneImageComplete.removeItem(self.becerro)
        self.becerro = self.sceneImageComplete.addPixmap(self.qpixmapBase)
        self.sceneImageComplete.update()

    def abrirQuery(self):
        qpixmap, img = self.abrir()
        self.qpixmapQuery = qpixmap
        self.proportion = self.getProportions()
        qpixmap = self.scalePixmap()
        if qpixmap:
            self.imgQuery = img
            if self.oveja != None:
                self.sceneImageQuery.removeItem(self.oveja)
            self.oveja = self.sceneImageQuery.addPixmap(self.qpixmapQuery)
            self.sceneImageQuery.update()
            self.resizeEvent(None)


            # def changeEvent(self, event):
            # if event.type() == QtCore.QEvent.WindowStateChange:
            # if self.isMinimized():
            # print "minimizar"
            ##return True
            # if self.windowState() == pyqtcore.Qt.WindowNoState:
            # print "normal"
            # self.update()
            # self.showPixmapBase()
            ##return True
            # elif self.windowState() == pyqtcore.Qt.WindowMaximized:
            # print "maximizada"
            # self.update()
            # self.showPixmapBase()
            ##qpixmap = self.scalePixmap(qpixmap)
            ##self.graphicsViewImageQuery.update()
            ##return True
            # else:
            # print "assssssssssss"
            ##self.sceneImageQuery.update()
            # super(GuiGEO, self).changeEvent(event)

    def getProportions(self):
        # wOriginal = self.qpixmapBase.width()
        # hOriginal = self.qpixmapBase.height()
        # wView = self.graphicsViewImageComplete.width()
        # hView = self.graphicsViewImageComplete.height()
        wOriginal = self.qpixmapQuery.width()
        hOriginal = self.qpixmapQuery.height()
        wView = self.graphicsViewImageQuery.width()
        hView = self.graphicsViewImageQuery.height()
        # print "------------"
        # print wOriginal, hOriginal
        # print wView, hView
        # print "------------"
        return [wOriginal / float(wView), hOriginal / float(hView)]

    def resizeEvent(self, event):
        bounds = self.sceneImageComplete.itemsBoundingRect()
        # bounds.setWidth(bounds.width()*0.9)
        # bounds.setHeight(bounds.height()*0.9)
        # bounds.setWidth(bounds.width())
        bounds.setHeight(bounds.height())
        self.graphicsViewImageComplete.fitInView(bounds, QtCore.Qt.IgnoreAspectRatio)
        self.graphicsViewImageComplete.centerOn(0, 0)
        bounds = self.sceneImageQuery.itemsBoundingRect()
        # bounds.setWidth(bounds.width()*0.9)
        # bounds.setHeight(bounds.height()*0.9)
        # bounds.setWidth(bounds.width())
        bounds.setHeight(bounds.height())
        self.graphicsViewImageQuery.fitInView(bounds, QtCore.Qt.IgnoreAspectRatio)
        self.graphicsViewImageQuery.centerOn(0, 0)

        if(self.imgQuery != None):
            self.proportion = self.getProportions()
        super(GuiGEO, self).resizeEvent(event)
