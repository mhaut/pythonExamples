import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as colors


#######################
pathname = 'foto.jpg'
nclustersList = [1,3,8,30]
scale = 4
#######################

img = cv2.imread(pathname)
width,height = img.shape[:2]

img = cv2.resize(img, (height/scale,width/scale), interpolation = cv2.INTER_CUBIC)
Z = X.reshape(img.shape[0]*img.shape[1], img.shape[2])


result = None
figurePos = 1
for nclusters in [1,3,8,30]:
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,nclusters,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    listLabels = list()
    for i in range(nclusters):
        listLabels.append(Z[label.ravel()==i])
    listColor = np.abs(np.random.rand(nclusters))
    ax1 = plt.subplot(1, 4, figurePos)
    for i,c in zip(range(nclusters),colors.cnames):
        ax1.scatter(listLabels[i][:,0],listLabels[i][:,1],color=c)
    figurePos += 1
    ax1.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    ax1.set_xlim(0,256)
    ax1.set_ylim(0,256)
    ax1.set_xticks([int(0),127.5,int(255)])
    ax1.set_yticks([int(0),127.5,int(255)])
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    if result == None:
        result = res2
    else:
        result = np.hstack((result,res2))


plt.subplots_adjust(wspace=None, hspace=None)
#plt.show()

img2 = cv2.imread('figure_1.png')
cv2.imshow('Simple k-means classification',result)
img2 = cv2.resize(img2, ((height*4+940)/scale,width*2/scale), interpolation = cv2.INTER_CUBIC)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()