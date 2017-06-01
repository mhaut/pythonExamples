import numpy as np
import time
from PyQt4 import QtCore
from proba import Manyfoxcam
import cv2
import matplotlib.pyplot as plt

class Camera(object):
	urls = []
	#urls.append("IP:PORT/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr=X&pwd=Y&.mjpg")
	alltheads = [None] * len(urls)
	for url, i in zip(urls, range(len(urls))):
		alltheads[i] = Manyfoxcam(url)

	for i in range(len(urls)):
		if alltheads[i].isRunning() == False:
			print "starting camera", i
			alltheads[i].start()

	while True:
		plt.ion()
		plt.axis("off")
		fcam = []
		for i in range(len(urls)):
			if alltheads[i].frame is not None:
				fcam.append(cv2.cvtColor(alltheads[i].frame, cv2.COLOR_BGR2RGB))
		if len(fcam) == 2:
			plt.imshow(np.hstack((fcam[0],fcam[1])))
			plt.pause(0.1)


					
a = Camera()