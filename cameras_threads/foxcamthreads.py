import sys, os, traceback, time
import numpy as np
import cv2
import time
from PyQt4 import QtCore
from PyQt4.Qt import *

class Manyfoxcam(QtCore.QThread):
	def __init__(self, url):
		QtCore.QThread.__init__(self, None)
		self.url = url
		self.frame = None
		self.app = QApplication(sys.argv)

	def __del__(self):
		self.wait()

	def run(self):
		url = self.url
		mirror=False
		while(True):
			try:
				vidFile = cv2.VideoCapture(url)
			except:
				print "problem opening input stream"
				#sys.exit(1)
				return None
			ret, frame = vidFile.read()
			if ret:
				if mirror:
					frame = cv2.flip(frame, 1)
				self.frame = frame
			else:
				self.frame = None
		vidFile.release()
		cv2.destroyAllWindows()