import cv2
from PIL import ImageChops

image = cv2.imread("magro.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale

_,thresh = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV) # threshold

#print thresh
cv2.imwrite("threshMagro.jpg", thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    print contour
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>300 and w>300:
        continue

    # discard areas that are too small
    if h<40 or w<40:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

# write original image with added contours to disk  
cv2.imwrite("contoured.jpg", image) 
