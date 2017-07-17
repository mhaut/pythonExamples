# -*- coding: utf-8 -*-
from PySide import QtGui
from PyQt4 import QtCore, QtGui, uic
import sys
import os
import cv2
import numpy as np
from math import sqrt 
from osgeo import gdal, osr
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 0
ALGORITHM_MATCH = "SIFT"

                                                                        #10
                                        ##########################################################################
                                        # ALGORITHM | SUBMAP LOCATE |    TIME(s)   |     GOOD     |     TOTAL    #
                                        #   sift    |     25/25     |      208     |     25/25    |     25/25    #
                                        #   surf    |     25/25     |      103     |     25/25    |     25/25    #
                                        #   obr     |     21/25     |      006     |     00/21    |     00/25    #
                                        #  brisk    |     21/25     |      156     |     20/21    |     20/25    #
                                        #  freak    |     18/25     |      084     |     18/25    |     18/25    #
                                        ##########################################################################
                                                                        #puerto_sin_SRC
                                        ##########################################################################
                                        # ALGORITHM | SUBMAP LOCATE |    TIME(s)   |     GOOD     |     TOTAL    #
                                        #   sift    |     00/25     |      160     |     00/00    |     00/25    #
                                        #   surf    |     00/25     |      089     |     00/00    |     00/25    #
                                        #   obr     |     11/25     |      005     |     00/11    |     00/25    #
                                        #  brisk    |     00/25     |      114     |     00/00    |     00/25    #
                                        #  freak    |     00/25     |      034     |     00/00    |     00/25    #
                                        ##########################################################################
                                                                            #3
                                        ##########################################################################
                                        # ALGORITHM | SUBMAP LOCATE |    TIME(s)   |     GOOD     |     TOTAL    #
                                        #   sift    |     07/25     |      161     |     00/07    |     00/25    #
                                        #   surf    |     03/25     |      081     |     00/03    |     00/25    #
                                        #   obr     |     17/25     |      006     |     00/17    |     00/25    #
                                        #  brisk    |     00/25     |      120     |     00/00    |     00/25    #
                                        #  freak    |     00/25     |      034     |     00/00    |     00/25    #                                        ##########################################################################
                                        
form_class =  uic.loadUiType('uis'+os.sep+'main.ui')[0]  # Ui 

class MyFirstScene(QtGui.QMainWindow, form_class):
#class MyFirstScene(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self,parent)
        
        self.setupUi(self)
        self.sceneImageQuery = QtGui.QGraphicsScene()
        self.sceneImageComplete = QtGui.QGraphicsScene()
        self.graphicsViewImageQuery.setScene(self.sceneImageQuery)
        self.graphicsViewImageComplete.setScene(self.sceneImageComplete)
        self.oveja = None
        self.becerro = None
        self.geoData = None
        self.src_pts = None
        self.dst_pts = None
        self.testingAPP()
        
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


    def algorithm_OBR(self):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(self.imgQuery,None)
        kp2, des2 = orb.detectAndCompute(self.imgBase,None)
        if self.actionCheckFlannMatcher.isChecked() == True:
            src_pts,dst_pts = self.matchImagesFlannLSH(kp1, kp2, des1, des2)
        else:
            #TODO: hacer esto!!!
            print "no hecho!!!!!!!!!!!!!!"
            ## create BFMatcher object
            #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            ## Match descriptors.
            #m = bf.match(des1,des2)
            ## Sort them in the order of their distance.
            #m = sorted(m, key = lambda x:x.distance)
        return src_pts, dst_pts

    def algorithm_SURF(self):
        # Initiate SIFT detector
        surf = cv2.xfeatures2d.SURF_create()
        # orientación arriba si o no. Si es si, entonces no se tiene en cuenta
        # find the keypoints and descriptors with SIFT
        kp1, des1 = surf.detectAndCompute(self.imgQuery,None)
        kp2, des2 = surf.detectAndCompute(self.imgBase,None)
        if self.actionCheckFlannMatcher.isChecked() == True:
            src_pts, dst_pts = self.matchImagesFlann(kp1,kp2,des1,des2)
        else:
            src_pts,dst_pts = self.matchImagesBruteForce(kp1,kp2,des1,des2)
        return src_pts,dst_pts

    def algorithm_BRISK(self):
        # Initiate BRISK detector
        #detector = cv2.BRISK(thresh=10, octaves=1)
        #surf = cv2.BRISK(thresh=10, octaves=1)
        brisk = cv2.BRISK_create(thresh=10, octaves=1)
        #brisksurf = cv2.xfeatures2d.BRISK_create()
        #brisk = cv2.DescriptorExtractor_create('BRISK')  # non-patented. Thank you!
        #surf = cv2.xfeatures2d.SURF_create()
        # orientación arriba si o no. Si es si, entonces no se tiene en cuenta
        # find the keypoints and descriptors with SIFT
        #print "aqui estamos"
        kp1, des1 = brisk.detectAndCompute(self.imgQuery,None)
        kp2, des2 = brisk.detectAndCompute(self.imgBase,None)
        if self.actionCheckFlannMatcher.isChecked() == True:
            src_pts,dst_pts = self.matchImagesFlannLSH(kp1,kp2,des1,des2)
        else:
            m = self.matchImagesBruteForce(kp1,kp2,des1,des2)
        return src_pts,dst_pts           
            
    def algorithm_FREAK(self):
        surfDetector = cv2.xfeatures2d.SURF_create()
        #surfDetector=cv2.GridAdaptedFeatureDetector(surfDetector,50)
        kp1 = surfDetector.detect(self.imgQuery,None)
        kp2 = surfDetector.detect(self.imgBase,None)
        #freakExtractor = cv2.DescriptorExtractor_create('FREAK')
        freakExtractor = cv2.xfeatures2d.FREAK_create()
        kp1,des1 = freakExtractor.compute(self.imgQuery,kp1)
        kp2,des2 = freakExtractor.compute(self.imgBase,kp2)
        del freakExtractor

        # find the keypoints and descriptors with SIFT
        #kp1, des1 = freak.detectAndCompute(self.imgQuery,None)
        #kp2, des2 = freak.detectAndCompute(self.imgBase,None)
        if self.actionCheckFlannMatcher.isChecked() == True:
            src_pts,dst_pts = self.matchImagesFlannLSH(kp1,kp2,des1,des2)
        else:
            m = self.matchImagesBruteForce(kp1,kp2,des1,des2)
        return src_pts,dst_pts

    def algorithm_SIFT(self):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.imgQuery,None)
        kp2, des2 = sift.detectAndCompute(self.imgBase,None)
        if self.actionCheckFlannMatcher.isChecked() == True:
            src_pts,dst_pts = self.matchImagesFlann(kp1,kp2,des1,des2)
        else:
            m = self.matchImagesBruteForce(kp1,kp2,des1,des2)
        return src_pts,dst_pts


    def matchImagesBruteForce(self, kp1, kp2, des1, des2):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        matches = sorted(matches, key = lambda x:x.distance)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(self.imgQuery,kp1,self.imgBase,kp2,good,None,flags=2)
        #plt.imshow(img3),plt.show()
        return good
    
        ##### Draw first 10 matches.
        #####img3 = cv2.drawMatches(self.imgQuery,kp1,self.imgBase,kp2,matches[:10],None, flags=2)
        ####img3 = cv2.drawMatches(self.imgQuery,kp1,self.imgBase,kp2,matches,None, flags=2)
        ####plt.imshow(img3),plt.show()

    def euclideanDistance(self, p1, p2):
        return sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

    def filterPoints(self,kp1,points):
        # src_pts es la imagen base
        src_pts = np.float32([ kp1[point.queryIdx].pt for point in points ]).reshape(-1,1,2)
        allDistances = [[None] * len(src_pts) for i in range(len(src_pts))]
        finish = False
        cont = 0
        aux = 0
        for i in range (0,len(src_pts)):
            for j in range (i,len(src_pts)-1):
                allDistances[i][j] = self.euclideanDistance(src_pts[j][0],src_pts[j+1][0])
                cont += 1
                aux  += allDistances[i][j]
        mean = aux/float(cont)
        cont = 0
        for i in range(len(src_pts)):
            for j in range(len(src_pts)):
                if allDistances[i][j] is not None:
                    if mean <= allDistances[i][j]:
                        allDistances[i][j] = None
                        print len(points),cont
                        if cont < len(points):
                            del points[cont]
                            cont -= 1
                    else:
                        cont += 1
                else:
                    pass
        return points
        
        
        
    def matchImagesFlann(self, kp1, kp2, des1, des2):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            #if m.distance < 0.75*n.distance:
            if m.distance < 0.68*n.distance:
                good.append(m)
        print "GOOOOOOOOOD",len(good)
        print good
        print "-----------"
        if len(good)>MIN_MATCH_COUNT:
            good = sorted(good, reverse = True)
            if len(good) > 2:
                good = good[:1]
                #good = self.filterPoints(kp1,good)
                #good = good[:10]
                #pass
            # en src_pts y dst_pts están las coordenadas pixel de la imagen origen y destino
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            #matchesMask = mask.ravel().tolist()
            #h,w = self.imgQuery.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            #dst = cv2.perspectiveTransform(pts,M)
            #imgAux = cv2.polylines(self.imgBase,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            src_pts = []
            dst_pts = []
            #matchesMask = None
            #good = None
        #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        #singlePointColor = None,
                        #matchesMask = matchesMask, # draw only inliers
                        #flags = 2)
        #img3 = cv2.drawMatches(self.imgQuery,kp1,imgAux,kp2,good,None,**draw_params)
        #plt.imshow(img3, 'gray'),plt.show()
        return src_pts,dst_pts


    def matchImagesFlannLSH(self, kp1, kp2, des1, des2):
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 20,     # 20
                   multi_probe_level = 2) #2
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.              
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m,n) = m_n
            if m.distance < 0.75*n.distance:
                good.append(m)
        #plt.imshow(self.imgBase),plt.show()
        #plt.imshow(self.imgQuery),plt.show()
        if len(good)>MIN_MATCH_COUNT:
            # en src_pts y dst_pts están las coordenadas pixel de la imagen origen y destino
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            src_pts = []
            dst_pts = []
        return src_pts, dst_pts

    def pixel2Coordinate(self,geotifAddr,pixelPairs):
	# Load the image dataset
	ds = gdal.Open(geotifAddr)
	# Get a geo-transform of the dataset
	gt = ds.GetGeoTransform()
	# Create a spatial reference object for the dataset
	srs = osr.SpatialReference()
	srs.ImportFromWkt(ds.GetProjection())
	# Set up the coordinate transformation object
	srsLatLong = srs.CloneGeogCS()
	ct = osr.CoordinateTransformation(srs,srsLatLong)
	# Go through all the point pairs and translate them to pixel pairings
	latLonPairs = []
	for point in pixelPairs:
		# Translate the pixel pairs into untranslated points
		ulon = point[0][0]*gt[1]+gt[0]
		ulat = point[0][1]*gt[5]+gt[3]
		# Transform the points to the space
		#(lon,lat,holder) = ct.TransformPoint(ulon,ulat)
		# Add the point to our return array
		latLonPairs.append([ulat,ulon])
	return latLonPairs
        
    
    def emparejar(self, geoTransform, matchesPoints, filename):
        src_pts = matchesPoints[0]
        dst_pts = matchesPoints[1]
        print "----------     ",src_pts
        if src_pts == None and dst_pts == None:
            #QtGui.QMessageBox.information(self, 'Info Message', ''' No match found! '''+str(filename), QtGui.QMessageBox.Ok)
            print  'No match found! '+str(filename)
            return None
        else:
            puntosComando = ''
            outfile = open(filename, 'w')
            # enable siempre a 1
            outfile.write("mapX,mapY,pixelX,pixelY,enable\n")
            listLatLon = self.pixel2Coordinate('images/montain_con_modified.tif',dst_pts)

            listlengh = []
            cont = 0
            mean = 0
            for g in listLatLon:
                suma = sum(g)
                listlengh.append(suma)
                mean += suma
            mean = mean / len(listLatLon)
            
            #for i in range(len(listLatLon),0):
                #if listLatLon[i] > mean:
                    #listLatLon.pop(i)
                    #src_pts.pop(i)
                    #dst_pts.pop(i)
            #exit()
            listSTD = []
            for geoLoc in zip(listLatLon):
                g = np.array(geoLoc[0])
                #listSTD.append(np.linalg.norm(g,2)) # norma 2
                listSTD.append(np.std(g)) # desviacion estandar
                
            listSTD = np.array(listSTD)
            listSTD = listSTD - np.mean(listSTD)
            mean = np.mean(np.absolute(listSTD))
            listremove = []
            for i in range(0,len(listSTD)):
                if abs(listSTD[i]) > mean*2:
                    print "eliminando:",i,"desviación estándar: ",listSTD[i],"Media: ", mean
                    listremove.append(i)
            listLatLon = np.delete(listLatLon, listremove, axis = 0)
            src_pts = np.delete(src_pts, listremove, axis = 0)
            
            print "----------     ",src_pts
            #exit()
            for geoLoc,point in zip(listLatLon,src_pts):
                #print str(geoLoc[1])+","+str(geoLoc[0])+","+str(point[0][0])+","+str(point[0][1])
                # TODO: Revisar el simbolo negativo
                outfile.write(str(geoLoc[1])+","+str(geoLoc[0])+","+str(point[0][0])+","+str(-point[0][1])+",1\n")
                puntosComando += "-gcp "+str(point[0][0])+" "+str(point[0][1])+" "+str(geoLoc[1])+" "+str(geoLoc[0])+" "
            outfile.close()
            return puntosComando


    def crop(self,image,numImagesH,numImagesW):
        height, width = image.shape   # Get dimension
        listImages = []
        cropH = height / numImagesH
        cropW = width / numImagesW
        initH = 0
        finishH = cropH
        for i in range (0,numImagesH):
            initW = 0
            finishW = cropW
            for j in range (0,numImagesW):
                listImages.append(image[initH : finishH, initW : finishW])
                initW = finishW
                finishW += cropW
            #listImages.append(self.imgBase[initH : finishH, finishW : width-1])
            initH = finishH
            finishH += cropH
        cont = 0
        for image in listImages:
            #plt.imshow(image),plt.show()
            cv2.imwrite('out/images/im'+str(cont)+'.png',image)
            cont += 1
        if listImages == []:
            return image
        else:
            return listImages
    
    def geoReference(self, puntosComando, numberImage):
        import subprocess
        #print "geoReferencing image",numberImage
        
        comando1 = 'gdal_translate -of GTiff '+puntosComando+'"/home/pbustos/software/surfSift/app/out/images/im'+str(numberImage)+'.png" "/tmp/im'+str(numberImage)+'.png"'# 1>/dev/null'
        comando2 = 'gdalwarp -r near -order 1 -co COMPRESS=NONE  "/tmp/im'+str(numberImage)+'.png" "/home/pbustos/software/surfSift/app/out/imagesGeoref/im'+str(numberImage)+'_modified.tif"'# 1>/dev/null'
        outfile = open("temp_commands.sh", "a")
        # enable siempre a 1
        outfile.write(comando1+"; "+comando2+"; ")
        outfile.close
        #os.system(comando)
    
    
    def work(self,listImages):
        self.actionCheckFlannMatcher.setChecked(True)
        cont = 0
        georef = False
        for imageQuery in listImages:
            #if cont > 9 and cont < 12:
            self.imgQuery = imageQuery
            #plt.imshow(self.imgQuery),plt.show()
            
            #TODO: ELIMINAR matches porque no se usa, los metodos deberian devolver src_pts y dst_pts como variables locales. Eliminar dichas variables de clases
            print "algorithm"
            if ALGORITHM_MATCH == "SURF":
                src_pts,dst_pts = self.algorithm_SURF()
            elif ALGORITHM_MATCH == "SIFT":
                src_pts, dst_pts = self.algorithm_SIFT()
            elif ALGORITHM_MATCH == "OBR":
                src_pts, dst_pts = self.algorithm_OBR()
            elif ALGORITHM_MATCH == "BRISK":
                src_pts, dst_pts = self.algorithm_BRISK()
            elif ALGORITHM_MATCH == "FREAK":
                src_pts, dst_pts = self.algorithm_FREAK()

            if src_pts != [] and dst_pts != []:
                print "emparejar"
                puntosComando = self.emparejar(self.geoData,[src_pts, dst_pts],'out/datas/data'+str(cont)+'.points')
                if puntosComando != None:
                    georef = True
                    print "georeferenciar"
                    self.geoReference(puntosComando,cont)
            else:
                print "Not geoReferencing image",cont
            cont += 1
        if georef == True:
            os.system("sh temp_commands.sh; rm temp_commands.sh")


    def testingAPP(self):
        #self.imgBase = cv2.imread('images/puerto_con_SRC.tif',cv2.IMREAD_GRAYSCALE)
        #imgQuery = cv2.imread('images/puerto_sin_SRC.TIF',cv2.IMREAD_GRAYSCALE)
        #listImagesQuery = self.crop(imgQuery,5,5)
        self.imgBase = cv2.imread('images/montain_con_modified.tif',cv2.IMREAD_GRAYSCALE)
        #imgQuery = cv2.imread('images/montain_con.png',cv2.IMREAD_GRAYSCALE)
        img = cv2.imread('images/mountain_sin.png',cv2.IMREAD_GRAYSCALE)
        print "imagenes cargadas"
        #imgQuery = cv2.fastNlMeansDenoising(img, 4, 21, 35)
        #imgQuery = cv2.fastNlMeansDenoising(img)
        #cv2.imwrite('images/pruebaaaaaaaa.png',imgQuery)
        imgQuery = cv2.imread('images/pruebaaaaaaaa.png',cv2.IMREAD_GRAYSCALE)
        print "imagen filtrada"
        listImagesQuery = self.crop(imgQuery,3,3)
        print "imagen segmentada"
        self.work(listImagesQuery)
        exit()
        
if __name__=="__main__":
    app=QtGui.QApplication(sys.argv)
    firstScene = MyFirstScene()
    firstScene.show()
    sys.exit(app.exec_())