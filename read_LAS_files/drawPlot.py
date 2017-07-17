# -*- encoding: utf-8 -*-
import sys
import pylab
import ctypes
import numpy as np
from time import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# sudo apt-get install python-liblas
from liblas import file

start_time = time()

listX = []
listY = []
listZ = []
listIntensity = []
listReturnNumber = []
listNumberReturns = []
listClassification = []
allList = []
f = file.File(sys.argv[1],mode='r') # f is a

print "read",len(f),"lines of datafile."
dataReadNumber = 0
last_time = start_time
for p in f:
    if time() - last_time > 1:
        print("%0.2f lines per seconds. Time:%0.2f" % (dataReadNumber / float(time() - last_time), abs(last_time - time())))
        last_time = time()
        dataReadNumber = 0
    listX.append(p.get_x())
    listY.append(p.get_y())
    listZ.append(p.get_z())
    listIntensity.append(p.get_intensity())
    listReturnNumber.append(p.get_return_number())
    listNumberReturns.append(p.get_number_of_returns())
    listClassification.append(p.get_classification())
    dataReadNumber += 1

list2DName = {"listIntensity":listIntensity,"listReturnNumber":listReturnNumber,
              "listNumberReturns":listNumberReturns,"listClassification":listClassification}

print("Load Data in: %0.10f seconds." % (time() - start_time))
print "Plot"

# 3d
print "plot listPoints"
fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(listX, listY, listZ)
ax.set_title("Points")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.savefig("plots/listPoints.png")
plt.close(fig)

# 2d
for currentList, currentName in zip(list2DName.values(),list2DName.keys()):
    print "plot",currentName
    x = np.arange(len(currentList))
    plt.xlabel("item")
    plt.ylabel("value")
    plt.title(currentName.replace("list",""))
    # all symbols: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
    plt.plot(x,currentList,'.')
    plt.plot(x,np.full(len(currentList),np.mean(currentList)),'-',linewidth=4,c='r')
    plt.savefig("plots/"+currentName+".png")
    plt.close()

print("Finish execution at: %0.10f seconds." % (time() - start_time))
print "Bye"