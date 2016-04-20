import cv2
import numpy as np

pathInput = ""
pathOutput = ""
image = cv2.imread(pathInput, -1)
image[image == 0] = 0
image[image != 0] = 255

cv2.imwrite(pathOutput, image)