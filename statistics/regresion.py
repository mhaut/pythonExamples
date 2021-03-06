#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy as sp

pointsX = [21.65, 22.04, 20.53, 17.26, 15.33, 12.44, 8.64,        21.13, 24.13, 21.82, 17.84, 15.92, 13.42, 8.5,        19.42, 20.45, 17.79, 15.3, 13.7, 12, 7.16,        20.77, 21.75, 19.8, 16.87, 15.8, 13.98, 9.61,        16.33, 16.57, 15.6, 12.35, 10.69, 11.34, 6.62,        15.71, 17.93, 16.16, 12.99, 11.9, 10.83, 6.83,        18.16, 20.04, 20.57, 17.21, 14.15, 12.65, 7.89,        20.17, 21.38, 18.67, 17.08, 14.96, 13.05, 7.29,        20.33, 23.12, 22.52, 19.16, 17.66, 16.22, 8.89,        29.79, 33.87, 33.08, 25.08, 22.97, 20.73, 15.35,        18.75, 19.76, 18.53, 16.34, 15.84, 13.86, 8.6,        29,    29.97, 28.59, 23.08, 21.22, 18.87, 11.66,        25.81, 28.05, 27.19, 23.99, 22.86, 20.76, 11.98,        20.14, 22.34, 23.26, 21.95, 20.18, 17.97, 10.19,        34.78, 36.22, 34.35, 30.13, 27.77, 25.25, 17.7,        26.59, 28.98, 27.62, 24.99, 22.86, 20.33, 12.43,        32.42, 33.73, 32.58, 29.28, 28.6, 26, 17.24,        31.84, 34.79, 37.02, 27.75, 18.63, 18.63, 17.45,        28.4, 32.41, 26.86, 22.41, 23.47, 23.47, 19.89]

pointsY = [-1952, -151, 768, 712, 2182, 4519, 2690,        -1455, -882, -855, 150, -78, 680, 1642,        -495, -214, -567, 63, 262, 8209, 12966,        -6698, -7974, -7623, -5248, -1995, -2589, -3744,        3398, 2995, 3243, 3375, 1296, 812, -1095,        680, -75, 632, 987, 1608, 1742, 2094,        -324, -270, -310, -443, -382, -409, 467,        -635, -322, -549, -1013, -1632, -1291, 822,        2720, -1455, -3100, -1048, -1103, -3815, -5155,        -2431, -1074, -1390, -253, 637, 1808, - 77,        15641, 14830, 13520, 7729, - 985, 4179, - 10013,        -8673, -9386, -6978, 445, 6318, 8209, 12996,        -170, 357, -1229, -2524, - 3394, - 4790, 1188,        3658, 5062, 4395, 257, - 1193, 315, 3056,        -5290, - 6627, - 6153, - 2976, 1527, 1780, - 1069,        271, 1561, 959, - 468, - 41, 1338, 304,        2140, 2824, 3777, - 1106, - 4520, - 6359, - 5502,        - 499, 269, 38, 261, 398, 738, 27,        113, 534, 1422, 1099, 1042, 931, 201]


# draw points
plt.plot(pointsX,pointsY,'o')


fp = sp.polyfit(pointsX,pointsY, 1)
f1 = sp.poly1d(fp)
# this is equation
print f1
# draw rect
plt.plot(pointsX, f1(pointsX), linewidth=4)


fp = sp.polyfit(pointsY,pointsX, 1)
f1 = sp.poly1d(fp)
# this is equation
print f1
# draw rect
plt.plot(f1(pointsY), pointsY, linewidth=4)

plt.show()

