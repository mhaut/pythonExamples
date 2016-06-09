import cv2
import math
import matplotlib.pyplot as plt

import numpy as np

class ClusteringClass():
    
    def __init__(self, n_classes_, max_iter_, epsilon_, attempts_):
        print '[CLUSTERING CLASS] Initializating Clustering Class'
        self.n_classes = n_classes_
        self.max_iter = max_iter_
        self.epsilon = epsilon_
        self.attempts = attempts_
        self.z = None
        self.criteria = None
        self.ret = None
        self.label = None
        self.center =  None
        self.clasesPossibles = []
    
    def make_clustering(self, img):
        self.z = img.reshape((-1,3))  # Gives a new shape to an array without changing its data.
        self.z = np.float32(self.z)  # Convert to float
        # criteria max_iter = 10 , epsilon accuracy = 1.0
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.epsilon)
        self.ret, self.label, self.center = cv2.kmeans( self.z, self.n_classes, self.criteria, self.attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    #def getStatistics(filename,image,num_clases,imageLabelClass):
    def getStatistics(self, path, image, drawFunction=True):
        listMeanCrCb = []
        listMaxCrCb = []
        listMinCrCb = []
        listDesvCrCb = []
        filas, columnas = image.shape[:2]
        #generalDirectory, currentDirectory, filename = path.split("/")
        for nclass in range(self.n_classes):
            listaAux = []
            value = 0
            cont = 0
            maximo = -1
            minimo = 256
            for i in range (filas*columnas):
                if self.label[i] == nclass:
                    pixel = image[i/columnas, i%columnas]
                    #diferencia = int(pixel[2] - pixel[1])
                    diferencia = abs(int(pixel[2]) - int(pixel[1]))
                    listaAux.append(diferencia)
                    if diferencia < minimo:
                        minimo = diferencia
                    if diferencia > maximo:
                        maximo = diferencia
                    value += diferencia
                    cont += 1
            desv = 0
            mean = value/cont
            for diff in listaAux:
                desv += ((diff - mean) ** 2)

            desv = math.sqrt( desv / (cont - 1))        
            listMeanCrCb.append(mean)
            listMaxCrCb.append(maximo)
            listMinCrCb.append(minimo)
            listDesvCrCb.append(desv)
            
            x = np.arange(len(listaAux))
            listamean = np.full(len(listaAux),mean)
            nameF, extF = path.split(".")
            if drawFunction == True:
                fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                ax.plot(x,listaAux,'o')
                ax.plot(x,listamean,'-',linewidth=4,c='r')
                fig.savefig(nameF+"_class_"+str(nclass+1)+"."+extF)
                plt.close(fig) 

        ####string = ""
        ####stringaAll = ""
        ####atleastone = False
        self.clasesPossibles = []
        #####for num in range(len(listMeanCrCb)):
            #####nulistDesvCrCb
            ######if listMeanCrCb[num] > 200 and listMinCrCb[num] > 150:
            ######if 1 == 1:
                #####atleastone = True
                #####self.clasesPossibles.append(num)
                #####string += filename+','+str(num+1)+","+str(listMeanCrCb[num])+","+str(listMinCrCb[num])+","+str(listMaxCrCb[num])+","+str(listDesvCrCb[num]) + "\n"
            #####elif listMinCrCb[num] > 10:
                #####stringaAll += filename+','+str(num+1)+","+str(listMeanCrCb[num])+","+str(listMinCrCb[num])+","+str(listMaxCrCb[num])+","+str(listDesvCrCb[num]) + "\n"        

        num = listDesvCrCb.index(max(listDesvCrCb))
        self.clasesPossibles.append(num)
        nameF = nameF.replace("exportPlot","data")
        nameF = nameF.split("/")
        imagename = nameF[2]
        nameF = nameF[0] +"/"+ nameF[1]
        outfile = open(nameF+'.csv', 'a')
        outfile.write(imagename+','+str(num+1)+","+str(listMeanCrCb[num])+","+str(listMinCrCb[num])+","+str(listMaxCrCb[num])+","+str(listDesvCrCb[num]) + "\n")
        outfile.close()


    def getMasks(self, size, removeNoise=True):
        if self.clasesPossibles:
            listmask = [None] * self.n_classes
            filas, columnas = size
            colors = [None] * self.n_classes
            for i in self.clasesPossibles:
                colors[i] = 255/(i+1)

            for nclass in self.clasesPossibles:
                print "[INFORMATION] Get mask for the class number", nclass+1,"of",len(self.clasesPossibles),"classes"
                clustered = np.zeros((filas,columnas))
                for i in range (filas*columnas):
                    try:
                        clase = self.label[i]
                        color = colors[clase]
                        if clase == nclass:
                            clustered[i/columnas, i%columnas] = 255
                        else:
                            clustered[i/columnas, i%columnas] = 0
                    except:
                        print i/columnas, i%columnas
                        print clustered[i/columnas, i%columnas]
                        print "---> ",i
                        exit()
                clustered = np.uint8(clustered)
                if removeNoise == True:
                    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                    mask = cv2.morphologyEx(clustered, cv2.MORPH_OPEN, se1)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
                mask = np.dstack([mask, mask, mask]) / 255
                listmask[nclass] = mask
        else:
            print "[IMFORMATION] no classes possibles found!!!"
        return listmask


    def getContours(self, image):
        imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,0,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def drawContourInImage(self, image, classID, filename, contours, area=0, drawContour=True, export=True):
        areas = []
        i = 0
        for contour in contours:
            if cv2.contourArea(contour) > area:
                cv2.drawContours(image, contours, i, (0,255,0), 1)
            i += 1
        nameF, extF = filename.split(".")
        cv2.imwrite(nameF+"_class_"+str(classID+1)+"."+extF, image)

    def draw_clusters(listContentClass,center):
        # Now separate the data, Note the flatten()
        listContentClass = []
        for i in range(k):
            listContentClass.append(z[label.ravel()==i])
        listColor = ['r','g','b','y','m','c','r','g','b','y','m','c','r','g','b','y','m','c','r','g','b','y','m','c']
        i = 0
        for contentClass in listContentClass:            
            plt.scatter(contentClass[:,1],contentClass[:,2],c = listColor[i])
            i += 1
        plt.scatter(center[:,1],center[:,2],s = 80,c = 'y', marker = 's')
        plt.xlabel('Cr'),plt.ylabel('Cb')
        plt.show()