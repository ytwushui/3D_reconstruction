import cv2
import numpy as np
import math
fre = [6,5]
height, width= 100, 1280
path = './double/'
def create_fig(freq, h, w):
    img = np.zeros((h,w))
    for i in range(2):
        for j in range(4):
            for k in range(w):
                img[:, k] = 127.5+127.5*math.cos(2 * math.pi * k * freq[i] / w + j * math.pi / 2)
                if k == 0:
                    print(img[:,k])
            cv2.imwrite('./double/{}_{}.jpg'.format(i+1,j+1), img)
def relativePhase(ip, h, w):
    I1=np.array(cv2.imread(path+'{}_1.jpg'.format(ip),0)).astype(np.float32)
    I2 = np.array(cv2.imread(path+'{}_2.jpg'.format(ip),0)).astype(np.float32)
    I3 = np.array(cv2.imread(path+'{}_3.jpg'.format(ip),0)).astype(np.float32)
    I4 = np.array(cv2.imread(path+'{}_4.jpg'.format(ip),0)).astype(np.float32)
    nominator, denominator = np.subtract(I4,I2),np.subtract(I1,I3)
    r_phase = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            x = I4[i,j] - I2[i,j]
            y = I1[i,j]- I3[i,j]
            if I4[i,j] == I2[i,j] and I1[i,j] > I3[i,j]:
                r_phase[i,j]=0
            elif I4[i,j] == I2[i,j] and I1[i,j] < I3[i,j]:
                r_phase[i][j] = math.pi
            elif I1[i,j] == I3[i,j] and I4[i,j] > I2[i,j]:
                r_phase[i][j]= math.pi/2
            elif I1[i,j] == I3[i,j] and I4[i,j] < I2[i,j]:
                r_phase[i,j] = 3*math.pi/2
            elif I1[i,j] < I3[i,j]:  #二三象限
                r_phase[i,j]= math.atan((I4[i,j] - I2[i,j])/(I1[i,j] - I3[i,j])) + math.pi
            elif I1[i][j] > I3[i][j] and I4[i][j] > I2[i][j] :# % 第一象限
                r_phase[i,j]= math.atan((I4[i,j]- I2[i,j])/ (I1[i,j] - I3[i,j]))
            elif I1[i,j] > I3[i,j] and I4[i,j] < I2[i,j]: #% 第四象限
                r_phase[i,j]= math.atan((I4[i,j] - I2[i,j])/ (I1[i,j]- I3[i,j])) + 2 * math.pi
            if r_phase[i,j] >=7:
                print (r_phase[i,j], I1[i,j], I3[i,j],I4[i,j] ,I2[i,j],
                       x,y, 'I4',I4[i,j], 'I2',I2[i,j], I4[i,j]-I2[i,j] )
    print("finished relative Phase Calculation")

    return r_phase

def exter_differ(p1, p2):
    p12 = np.zeros((height,width))
    n = 0
    for i in range(height):
        for j in range(width):
            if p1[i,j] > p2[i,j]:
                p12[i,j] = p1[i,j]-p2[i,j]
            else:
                p12[i,j] = p1[i,j]-p2[i,j]+ 2*math.pi
    return p12
def unwrap_phase1():
    p1, p2, =1/6, 1/5

    N1=np.floor(p2/(p2-p1)*ph12/2/math.pi)
    np.savetxt('./double/N1.txt', N1)
    print(N1)
    ph1_abs = 2*math.pi*N1+phase1
    return ph1_abs

def relativePhase2(ip, h, w):
    I1 = np.array(cv2.imread(path + '{}_1.jpg'.format(ip), 0))

    np.savetxt('./double/I1.txt', I1)
    print(I1[0, 1:100])
    cv2.imshow('I1', I1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    I2 = cv2.imread(path + '{}_2.jpg'.format(ip), 0)
    I3 = cv2.imread(path + '{}_3.jpg'.format(ip), 0)
    I4 = cv2.imread(path + '{}_4.jpg'.format(ip), 0)
    nominator, denominator = np.subtract(I4, I2), np.subtract(I1, I3)
    r_phase = np.zeros((h, w))
    for i in range(h):
        for j in range(w):

            y = I4[i,j] - I2[i,j]
            x = I1[i,j] - I3[i,j]
            if x>0: r_phase[i,j] = math.atan(y/x)+math.pi/2
            elif x<0 and y>=0: r_phase[i,j] = math.atan(y/x)+3*math.pi/2
            elif x<0 and y<0: r_phase[i,j] = math.atan(y/x)-math.pi/2
            elif x==0 and y>0: r_phase[i,j] = math.pi*2
            elif x==0 and y<0: r_phase[i,j] = 0

    print("finished relative Phase Calculation")

    return r_phase

phase1 = relativePhase(1,height,width)

np.savetxt(path + 'phase1_r.txt', phase1, fmt="%.3f", delimiter=' ')
cv2.imshow('phase1',phase1/6.3)
cv2.waitKey(0)
cv2.destroyAllWindows()

phase2 = relativePhase(2,height,width)
cv2.imshow('phase2',phase2/6.3)
cv2.waitKey(0)
cv2.destroyAllWindows()

ph12 = exter_differ(phase1,phase2)
cv2.imshow('ph12',ph12/10)
cv2.waitKey(0)
cv2.destroyAllWindows()

p1 = unwrap_phase1()
cv2.imshow('ph1',p1/50)
cv2.waitKey(0)
cv2.destroyAllWindows()
np.savetxt('./image/image_test/small1.txt', p1)
