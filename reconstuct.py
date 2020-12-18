import cv2
import numpy as np
import math

path = './image/object/image1/'
path1 = './image/image1/'
freq = [80,75,71]

#y = np.loadtxt("./image/calculation/ph123_test.txt")
y = np.loadtxt('./image/object/image1/phase1_up_org.txt')
y1 = np.loadtxt('./image/surface/image1/phase1_up_org.txt')
h, w = len(y), len(y[0])
recon_3d = np.zeros((w*h,3))
recon_3d2 = np.zeros((w*h,3))
# h = lt*fi/(2pid+tfi)
L=50
d = 10
T = 10
hight = 500
width = 700
for i in range(h):
    for j in range(w):
        dlta = y[i,j]
        dlta2 =y1[i,j]-y1[i,j]
        hi = L*T*dlta/(2*math.pi*d+ T*dlta)
        hi2 = L*T*dlta2/(2*math.pi*d+ T*dlta2)

        if abs(hi)>100: hi =0
        if abs(hi2)>100: hi2 = 0
        recon_3d[i*w+j] = [i/h*hight,j/w*width, hi]
        recon_3d2[i * w + j] = [i / h * hight, j / w * width, hi2]

np.savetxt("./image/calculation/"+'out_obj_new.txt', recon_3d, fmt="%.3f", delimiter=' ')
np.savetxt("./image/calculation/"+'result_obj_sur.txt', recon_3d2, fmt="%.3f", delimiter=' ')

print ("finished save txt file")