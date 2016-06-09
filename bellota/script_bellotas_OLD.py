import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import time

def drawClusters(listContentClass,center):
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
    
def cvShowManyImages(img1, img2):
    width,height = img1.shape[:2]
    new_height = height/3
    new_width = (width/3)*2
    multiple_img = np.zeros((new_height, new_width, 3), np.uint8)
    print img1.shape, img2.shape
    try:
        multiple_img = np.append(img1, img2, axis=1)
        cv2.imshow('Result images',multiple_img)
    except:
        print 'DIMENSIONES DISTINTAS JOER'
        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def deleteYcomponent(img):
    aux_img = []
    for row in img:
        fil = []
        for pixel in row:
            pixel[0] = 0
            fil.append(pixel)
        aux_img.append(fil)
    return np.asarray(aux_img)

def deleteYcomponentandCrCb(img):
    aux_img = []
    for row in img:
        fil = []
        for pixel in row:
            pixel[0] = 0
            CrCb = pixel[2] - pixel[1]
            #if CrCb >= 220 and CrCb < 240:
            if CrCb < 150:
                fil.append(pixel)
            else:
                pixel[2] = 0
                pixel[1] = 0
                fil.append(pixel)
        aux_img.append(fil)
    
    return np.asarray(aux_img)

def changeColors(image,export = False):
    colors = []
    if export == False:
        colors.append([0.0,1.0,0.0])
        colors.append([0.2,0.6,1.0])
    else:
        colors.append([0.0,255.0,0.0])
        colors.append([0.0,176.0,255.0])
    aux_img = []
    for row in image:
        fil = []
        for pixel in row:
            if np.mean(pixel) != 0:
                fil.append(colors[0])
            else:
                fil.append(colors[1])
        aux_img.append(fil)
    return np.asarray(aux_img)

def showHistogram3D(image):
    import pylab
    from mpl_toolkits.mplot3d import Axes3D
    fig = pylab.figure()
    ax = Axes3D(fig)

    listY = []
    listCb = []
    listCr = []
    for row in image:
        for pixel in row:
            listY.append(pixel[0])
            listCb.append(pixel[1])
            listCr.append(pixel[2])

    ax.scatter(listCb, listCr, listY)
    plt.xlabel('Cb')
    plt.ylabel('Cr')
    plt.show()

def showHistogram2D(image):
    hist_fullR = cv2.calcHist([image],[0],None,[256],[0,256])
    hist_fullG = cv2.calcHist([image],[1],None,[256],[0,256])
    hist_fullB = cv2.calcHist([image],[2],None,[256],[0,256])
    plt.plot(hist_fullR)
    plt.plot(hist_fullG)
    plt.plot(hist_fullB)
    plt.xlim([0,256])
    plt.ylim([0,100])
    plt.show()

    
def mixImageLine(imagenOrig,imageLine):   
    imagenCanny = imagenOrig.copy()
    print '[IMFORMATION] Canny make'

    numfil = 0
    #aux_img = []
    for row in imagenCanny:
        numcol = 0
        fil = []
        for pixel in row:
            # si es borde                        
            if imageLine[numfil][numcol] != 0:
                pixel[0] = 255
                pixel[1] = 0
                pixel[2] = 0
            #fil.append(pixel)
            numcol += 1
        #aux_img.append(fil)
        numfil += 1
    return imagenCanny


def getStatistics(filename,image,num_clases,imageLabelClass):
    filename = filename.split("/")
    listMeanCrCb = []
    listMaxCrCb = []
    listMinCrCb = []    
    listDesvCrCb = []
    filas, columnas = image.shape[:2]
    for nclass in range(num_clases):
        listaAux = []
        value = 0
        cont = 0
        maximo = -1
        minimo = 256
        for i in range (filas*columnas):
            if imageLabelClass[i] == nclass:
                pixel = image[i/columnas, i%columnas]
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

        x = np.arange(len(listaAux))
        listamean = []
        for i in listaAux:
            listamean.append(mean)

        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.plot(x,listaAux,'o')
        ax.plot(x,listamean,'-',linewidth=4,c='r')
        fig.savefig('exportPlot/'+filename[1].split(".")[0]+str(nclass)+".png")   # save the figure to file
        plt.close(fig) 
        
        import math
        desv = math.sqrt( desv / (cont - 1))
        print "------------->   ",mean,desv
        listMeanCrCb.append(mean)
        listMaxCrCb.append(maximo)
        listMinCrCb.append(minimo)
        listDesvCrCb.append(desv)

    string = ""
    stringaAll = ""
    atleastone = False
    classesOK = []
    for num in range(len(listMeanCrCb)):
        #if listMeanCrCb[num] > 200 and listMinCrCb[num] > 150:
        if 1 == 1:
            atleastone = True
            classesOK.append(num)
            string += filename[1]+','+str(num+1)+","+str(listMeanCrCb[num])+","+str(listMinCrCb[num])+","+str(listMaxCrCb[num])+","+str(listDesvCrCb[num]) + "\n"
        elif listMinCrCb[num] > 10:
            stringaAll += filename[1]+','+str(num+1)+","+str(listMeanCrCb[num])+","+str(listMinCrCb[num])+","+str(listMaxCrCb[num])+","+str(listDesvCrCb[num]) + "\n"        

    if atleastone == True:
        outfile = open('data/'+filename[0]+'.csv', 'a')
        outfile.write(string)
    else:
        outfile = open('data/error_'+filename[0]+'.csv', 'a') # Indicamos el valor 'w'.
        print "[INFORMATION] ERROR in file: No classes for stork found",filename
        outfile.write(stringaAll)
    outfile.close()
    
    return classesOK




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'python script_bellotas.py image_path'
    else:        
        import os
        header = False
        listTime = []
        for directory in os.listdir(sys.argv[1]):
            if os.path.isdir(directory) and "input" in directory:
                photos = os.listdir(directory)
                print '[INFORMATION] Directory',directory,'have',len(photos),'images'
                # 1) TAKE THE IMAGE
                for photo in photos:
                    start_time = time.time()
                    if len(listTime) > 0:
                        meantime = reduce(lambda x,y:x+y,listTime)/len(listTime)
                        #print "[INFORMATION] Mean time execution per image: ", int(meantime), "seconds"
                        timeleft = (len(photos) - len(listTime)) * meantime
                        timeleftP = timeleft
                        if timeleft >=60:
                            timeleft = timeleft / 60                            
                            if timeleft > 60:
                                timeleft = timeleft / 60
                                timeleft = str(int(timeleft))+" hours"
                            else:
                                timeleft = str(int(timeleft))+" mins"
                        else:
                            timeleft = str(int(timeleft))+" seconds"
                        #print '[INFORMATION] Time Left time execution per image: ',timeleft
                        bar = 0
                        step = timeleftP / 100
                        timeP = (meantime * len(listTime))
                        maxStep = timeleftP - timeP
                        #progressBar = "["
                        #while(bar < timeP):
                            #progressBar += "|"
                            #bar += step
                        #while(bar < maxStep):
                            #progressBar += " "
                            #bar += step
                        #progressBar += "]"
                        print "[INFORMATION] Mean time:",int(meantime), "seconds and Left Time:",timeleft
                        #print progressBar
                            
                    filename = directory+"/"+photo
                    img = cv2.imread(directory+"/"+photo)
                    
                    # 2) RESIZE THE IMAGE
                    width,height = img.shape[:2]  # Take only the two first params: width and height
                    scale = 4
                    scaled_img = cv2.resize(img, (height/scale,width/scale), interpolation = cv2.INTER_CUBIC)
                    print '[INFORMATION] Your',photo,'image is',width,'x',height,'but it is scaled to',width/3,'x',height/3
            
                    #showHistogram3D(scaled_img)
                    
                    # 3) FROM BGR TO YCR_CB AND DELETE FIRST DATA PIXEL:
                    imgYCC = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2YCR_CB)
                    clear_imgYCC = deleteYcomponent(imgYCC)
                    #clear_imgYCC = deleteYcomponentandCrCb(imgYCC)        
                    #cvShowManyImages(scaled_img, clear_imgYCC)
                    
                    # 4) CLUSTERING
                    z = clear_imgYCC.reshape((-1,3))  # Gives a new shape to an array without changing its data.
                    z = np.float32(z)  # Convert to float
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # criteria max_iter = 10 , epsilon accuracy = 1.0
                    k = 6  # classes
                    #   samples,clusters,criteria,attempts,flags
                    ret,label,center = cv2.kmeans(      z,       k,criteria,      10,cv2.KMEANS_RANDOM_CENTERS)  # KMEANS
                    
                    if header == False:
                        outfile = open('data/'+directory+'.csv', 'w')
                        outfile.write("FILENAME,CLASSS,MEAN,MIN,MAX,DESVT\n")
                        outfile.close()
                        header = True
                    clasesPossibles = getStatistics(filename,clear_imgYCC,k,label)
                    
                    listTime.append(time.time() - start_time)
            
                    ## 5) CLASSIFICATION
                    if clasesPossibles:
                        for nclass in clasesPossibles:
                            colors = []
                            for i in range(k):
                                colors.append(255/(i+1))
                            print '[INFORMATION] Showing the class number', nclass+1,'for',k,'classes'
                            print '[IMFORMATION]', clear_imgYCC.shape
                            filas, columnas = clear_imgYCC.shape[:2]

                            clustered = np.zeros((filas,columnas))
                            for i in range (filas*columnas):
                                clase = label[i]
                                color = colors[clase]
                                if clase == nclass:
                                    clustered[i/columnas, i%columnas] = 255
                                else:
                                    clustered[i/columnas, i%columnas] = 0
                            clustered = np.uint8(clustered)
                            
                            out = clustered.copy()
                            # remove noise
                            se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                            se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                            mask = cv2.morphologyEx(out, cv2.MORPH_OPEN, se1)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)

                            mask = np.dstack([mask, mask, mask]) / 255
                            out = scaled_img * mask
                            
                            imageOutput = scaled_img.copy()
                            img1 = out.copy()
                            imgray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
                            ret,thresh = cv2.threshold(imgray,0,255,0)
                            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                            
                            areas = []
                            i = 0
                            for contour in contours:
                                if cv2.contourArea(contour) > 0:
                                    cv2.drawContours(imageOutput, contours, i, (0,255,0), 1)
                                i += 1
                            cv2.imwrite("exportImages/"+photo.split(".")[0]+"___class"+str(nclass+1)+".png", imageOutput)
                        
                    
                    ##for nclass in range(k):
                    #for nclass in [4]:
                        #colors = []
                        #for i in range(k):
                            #colors.append(255/(i+1))
                        #print '[INFORMATION] Showing the class number', nclass+1,'for',k,'classes'
                        #print '[IMFORMATION]', clear_imgYCC.shape
                        #filas, columnas = clear_imgYCC.shape[:2]

                        #clustered = np.zeros((filas,columnas))

                        #for i in range (filas*columnas):
                            #clase = label[i]
                            #color = colors[clase]
                            #if clase == nclass:
                                #clustered[i/columnas, i%columnas] = 255
                            #else:
                                #clustered[i/columnas, i%columnas] = 0


                        #clustered = np.uint8(clustered)
                        ##cv2.imshow("clustered", clustered);
                

                        #### Now convert back into uint8, and make original image
                        ##center = np.uint8(center)
                        ##res = center[label.flatten()]
                        ##res2 = res.reshape((img.shape))

                        ##cv2.imshow('res2',res2)
                        ##cv2.waitKey(0)
                        ##cv2.destroyAllWindows()
                        

                        #print "a------------_"
                        #out = clustered.copy()
                        #cv2.imshow('contourA', out)
                        #for i in range(0,1000):
                            #out = cv2.erode(clustered,(5,5),iterations = 1)
                            #out = cv2.dilate(out,(2,2),iterations = 1)
                        ##cv2.imshow('contour', out)
                        ##cv2.waitKey(0)
                        ##cv2.destroyAllWindows()    
                        
                        
                        
                        
                        
                        
                        #se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                        #se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                        #mask = cv2.morphologyEx(out, cv2.MORPH_OPEN, se1)
                        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)

                        #mask = np.dstack([mask, mask, mask]) / 255
                        #out = scaled_img * mask
                        ##out=dst
                        #cv2.imshow('contour', out)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        
                        
                        #hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
                        ##lower_green = np.array([20, 33, 33])
                        ##upper_green = np.array([40, 255, 255])
                        #lower_green = np.array([0,33,33])
                        #upper_green = np.array([35, 255, 136])

                        #green_mask = cv2.inRange(hsv, lower_green, upper_green) # I have the Green threshold image.
                        ## Bitwise-AND mask and original image
                        #res = cv2.bitwise_and(out,out, mask= green_mask)


                        ## True prepara para exportar, false para mostrar
                        ##imageRes = changeColors(hsv,True)
                        ##imageResF = changeColors(out,True)
                        
                        
                        ##imageResF = changeColors(edges,True)
                        #print '[IMFORMATION] Detecting border'
                        ## TODO: revisar
                        ##imageCanny = mixImageLine(scaled_img,scaled_img * mask)
                        #print '[IMFORMATION] Show image'
                        ##cv2.imshow('Out', out)
                        ##cv2.imshow('filterHSV', res)
                        #edges = cv2.Canny(out,0,255)            
                        #imageFilterCanny = mixImageLine(scaled_img,edges)
                        #edges = cv2.Canny(res,0,255)            
                        #imageFilterCanny2 = mixImageLine(scaled_img,edges)


                        #img1 = out.copy()
                        #img2 = res.copy()
                        #imgray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
                        ##image,contours,hierarchy = cv2.findContours(thresh, 1, 2)
                        #ret,thresh = cv2.threshold(imgray,0,255,0)
                        #contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)            
                        ################################convexList = []
                        ################################noconvexList = []
                        ################################for c in contours:
                            ################################if cv2.isEllipse(c) == True:
                                ################################convexList.append(c)
                            ################################else:
                                ################################noconvexList.append(c)
                        ################################        
                        ################################cv2.drawContours(img1, convexList, -1, (0,255,0), 1)
                        ################################cv2.drawContours(img2, noconvexList, -1, (0,255,0), 1)
                        
                        #areas = []
                        #i = 0
                        #for contour in contours:
                            #if cv2.contourArea(contour) > 0:
                                #cv2.drawContours(scaled_img, contours, i, (0,255,0), 1)
                            #i += 1
                        
                        ##cv2.drawContours(img2, areas, -1, (0,255,0), 1)
                        
                        ##imageContours = cv2.drawContours(image, contours, -1, (0,255,0), 3)
                        ##cnt = contours[0]
                        ##M = cv2.moments(cnt)
                        ##print M
                        ##cv2.drawContours(imageFilterCanny, contours, -1, (0,255,0), 3)
                        ##print type(imagContour)            
                        ##cv2.imshow('Output', imageFilterCanny)
                        
                        ##epsilon = 0.1*cv2.arcLength(cnt,True)
                        ##approx = cv2.approxPolyDP(cnt,epsilon,True)

                        ##imageContour = mixImageLine(scaled_img,approx)
                        #cv2.imshow('canny', imageFilterCanny)
                        #cv2.imshow('contour', scaled_img)
                        ##cv2.imshow('no contour', img2+scaled_img)
                        ##cv2.imshow('OutputHSV', imageFilterCanny2)
                        ##cv2.imshow('contour',imagContour)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()           

                        ##showHistogram2D(out)
                        ##cont = 0
                        ##for i in out:
                            ##for ia in i:
                                ##if np.mean(ia) != 0:
                                    ##cont += 1
                        ##print "para la clase",nclass+1," --> ",cont


                        

                        ##cv2.imshow('Output', res)
                        ##cv2.imshow('Output', imageRes)
                        ##cv2.waitKey(0)
                        ##cv2.destroyAllWindows()

                        ####################cv2.imwrite('imagesOut/output'+str(nclass+1)+'.png', out)
                        ####################cv2.imwrite('imagesColorOut/output'+str(nclass+1)+'.png', imageRes)
                        ####################cv2.imwrite('imagesFilterOut/output'+str(nclass+1)+'.png', imageResF)

                        ##cvShowManyImages(scaled_img, out)

                    #############################for c in center:
                        #############################c = np.uint8([[[int(round(c[0])),int(round(c[1])),int(round(c[2]))]]])
                        #############################print cv2.cvtColor(c,cv2.COLOR_BGR2HSV)

                    ## Plot the data
                    ########################drawClusters(z,center)

