import cv2
import numpy as np
path0 = './image/surface/image1/'
path = './image/image_test/'
y = np.loadtxt(path+'phase1_up_new.txt')/1000
cv2.imshow("phase1", y)
cv2.waitKey(0)
cv2.destroyAllWindows()