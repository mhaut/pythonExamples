
from rtree import Rtree
import random
import pylab


points = {}

dist = 500
index = Rtree()

# create some random points and put them in an index.
for i in range(30000):
    x = random.random() * 10000
    y = random.random() * 10000
    pt = (x, y)
    points[i] =  pt
    index.add(i, pt)

print "index created..."

groups = {}
while len(points.values()):
    pt = random.choice(points.values())
    print pt
    bbox = (pt[0] - dist, pt[1] - dist, pt[0] + dist, pt[1] + dist)

    idxs = index.intersection(bbox)
    # add actual distance here, to get those within dist.

    groups[pt] = []
    for idx in sorted(idxs, reverse=True):
        delpt = points[idx]
        groups[pt].append(delpt)
        index.delete(idx, delpt)
        del points[idx]

# groups contains keys where no key is within dist of any other pt
# the values for a given key are all points within dist of that point.
#print groups

# keys plotted in red.
for pt, subpts in groups.iteritems():
    subpts = pylab.array(subpts)
    pylab.plot(subpts[:,0], subpts[:,1], 'k.')
    pylab.plot([pt[0]], [pt[1]], 'ro')

pylab.show()
