import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


###Documentation:
###http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
###http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
###http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf


## initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture(1)


while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # resize image size
    (h, w) = image.shape[:2]
    width = min(400, image.shape[1])
    r = width / float(w)
    dim = (width, int(h * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05, useMeanshiftGrouping=True)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.55)
    print type(pick), type(weights)
    print weights, pick
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    cv2.imshow("Window", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break