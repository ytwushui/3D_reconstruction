import numpy as np
import cv2
import math

a=[0, 10000000, -1,-0.5]

b = np.arctan(a)
for x in a:
    print(math.atan(x))
print(b)
