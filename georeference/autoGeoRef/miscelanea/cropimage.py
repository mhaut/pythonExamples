
import cv2

image = cv2.imread('images/puerto_sin_SRC.TIF',cv2.IMREAD_GRAYSCALE)

numImagesH = 5
numImagesW = 5
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
    cv2.imwrite('out/trash/im'+str(cont)+'.png',image)
    cont += 1