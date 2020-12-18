import cv2
import numpy as np
import math

path = './image/object/image1/'
path1 = './image/image_surface/image1/'
path = './image/image_test/'

#freq = [80,75,71]
freq = [80,74,69]
#freq = [86,81,77]
f12 =abs((freq[0]*freq[1])/(freq[0]-freq[1]))
f23 = abs((freq[1]*freq[2])/(freq[1]-freq[2]))
f13 = abs((freq[0]*freq[2])/(freq[0]-freq[2]))
f123 = abs((f12*f23)/f12-f23)
img1 = cv2.imread(path+'1_1.jpg', -1)
I = np.array(img1)
height, width = len(I), len(I[0])
print(height,width)
def add_image():
    img1 = cv2.imread(path + '1_1.jpg', -1)
    img2 = cv2.imread(path + '2_1.jpg', -1)
    img3 = cv2.imread(path + '3_1.jpg', -1)

    img12 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    img23 = cv2.addWeighted(img2, 0.5, img3, 0.5, 0)
    img123 = cv2.addWeighted(img12, 0.5, img23, 0.5, 0)
def create_fig(fre, h, w):
    img = np.zeros((h,w))
    for i in range(3):
        for j in range(4):
            for k in range(w):
                img[:, k] = 127.5 + 127.5 * math.cos(2 * math.pi * k * freq[i] / w + j * math.pi / 2)
                if k == 0:
                    print(img[:,k])
            cv2.imwrite('./image/image_test2/{}_{}.jpg'.format(i+1,j+1), img)
def relativePhase(ip, h, w):
    I1= np.array(cv2.imread(path+'{}_1.jpg'.format(ip),-1)).astype(np.float32)
    I2 = np.array(cv2.imread(path+'{}_2.jpg'.format(ip),-1)).astype(np.float32)
    I3 = np.array(cv2.imread(path+'{}_3.jpg'.format(ip),-1)).astype(np.float32)
    I4 = np.array(cv2.imread(path+'{}_4.jpg'.format(ip),-1)).astype(np.float32)
    nominator, denominator = np.subtract(I4,I2),np.subtract(I1,I3)
    r_phase = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
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
    #uw_phase = unwrapp(r_phase)
    print("finished relative Phase Calculation")

    return r_phase
def new_relativePhase(ip, h, w):
    I1 = np.array(cv2.imread(path+'{}_1.jpg'.format(ip),-1))
    I2 = np.array(cv2.imread(path+'{}_2.jpg'.format(ip),-1))
    I3 = np.array(cv2.imread(path+'{}_3.jpg'.format(ip),-1))
    I4 = np.array(cv2.imread(path+'{}_4.jpg'.format(ip),-1))
    nominator, denominator = np.subtract(I4,I2),np.subtract(I1,I3)

    r_phase = np.zeros((h,w))
    s_phase = np.zeros((h,w))
    ss = math.pi*3/2
    for i in range(h):
        for j in range(w):

            if I4[i,j] == I2[i,j] and I1[i,j] > I3[i,j]:
                r_phase[i,j]=0
            elif I4[i,j] == I2[i,j] and I1[i,j] < I3[i,j]:
                r_phase[i][j] = math.pi
            elif I1[i,j] == I3[i,j] and I4[i,j] > I2[i,j]:
                r_phase[i][j]=math.pi/2
            elif I1[i,j] == I3[i,j] and I4[i,j] < I2[i,j]:
                r_phase[i,j] = 3*math.pi/2
            elif I1[i,j] < I3[i,j]:  #二三象限
                r_phase[i,j]= math.atan((I4[i,j] - I2[i,j])/(I1[i,j] - I3[i,j])) + math.pi
            elif I1[i][j] > I3[i][j] and I4[i][j] > I2[i][j] :# % 第一象限
                r_phase[i,j]= math.atan((I4[i,j]- I2[i,j])/ (I1[i,j] - I3[i,j]))
            elif I1[i,j] > I3[i,j] and I4[i,j] < I2[i,j]: #% 第四象限
                r_phase[i,j]= math.atan((I4[i,j] - I2[i,j])/ (I1[i,j]- I3[i,j])) + 2 * math.pi
            s_phase[i,j] = r_phase[i,j] -ss
            ss += 2*math.pi*freq[ip-1]/w
            if ss >= 2*math.pi:
                ss-=2*math.pi
    return s_phase
def others():
    img2 = cv2.imread('./image/image1/2_1.jpg')
    img3 = cv2.imread('./image/image1/3_1.jpg')
    image_arr = np.array(img1)
    print(len(image_arr), len(image_arr[0]))
    img12 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    img23 = cv2.addWeighted(img2, 0.5, img3, 0.5, 0)
    img123 = cv2.addWeighted(img12, 0.5, img23, 0.5, 0)
    cv2.imshow('Image', img123)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def exter_differ(p1, p2):
    p12 = np.zeros((height,width))
    n = 0
    for i in range(height):
        for j in range(width):
            if p1[i,j] >= p2[i,j]:
                p12[i,j] = p1[i,j]-p2[i,j]
            else:
                p12[i,j] = p1[i,j]-p2[i,j]+ 2*math.pi
    return p12
def unwrapp(phase):
    n = np.zeros((height,width))
    for i in range(1, width):
        if abs(phase[0, i] - phase[0, i - 1]) < math.pi:
            n[0, i] = n[0, i - 1]
        elif phase[0, i] - phase[0, i - 1] <= -math.pi:
            n[0, i] = n[0, i - 1]+ 1;
        elif phase[0, i] - phase[0, i - 1] >= math.pi:
            n[0, i] = n[0, i - 1] - 1;


    for i in range(1,height):
        for j in range(1,width):
            if abs(phase[i, j] - phase[i - 1, j]) < math.pi:
                n[i, j] = n[i - 1, j]
            elif phase[i, j] - phase[i - 1, j] <= -math.pi:
                n[i, j] = n[i - 1, j] + 1;
            elif phase[i, j] - phase[i - 1, j] >= math.pi:
                n[i, j] = n[i - 1, j] - 1;

    pphase = phase + 2 * math.pi * n
    return pphase
def unwrapp_phase(PH1,i):
    UPH1 =np.zeros((height, width))

    for g in range(height):
        for k in range (width):
            UPH1[g, k] = PH1[g, k] + 2 * math.pi * math.ceil((freq[i-1] * ph123[g, k] - PH1[g, k]) / (2 * math.pi))
    return UPH1
def phase_to_img_show(phase):
    uv = np.cos(phase) * 127+ 127

    cv2.imshow('Image', uv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def unwrap_phase1():

    p1, p2, p3 = 1/freq[0], 1/freq[1], 1/freq[2]
    p12, p23 = 1/(freq[0]-freq[1]), 1/(freq[1]-freq[2])
    n12 = p23/(p23-p12)*(ph123)/2/math.pi
    N12 = np.floor(n12)
    print(N12)
    N1=np.floor(p2/(p2-p1)*(N12+ph12/2/math.pi))
    print(N1)
    ph1_abs = 2*math.pi*N1+phase1
    return ph1_abs
runit = 0
runit2 = 1  # for generate result
runit3 = 0
check = 0
def mynorm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if check:
    # normalize
    ph1 = np.loadtxt(path+'phase1.txt')
    ph2 = np.loadtxt(path + 'phase2.txt')
    ph3 = np.loadtxt(path + 'phase3.txt')
    ph12 = exter_differ(ph1, ph2)
    ph23 = exter_differ(ph2, ph3)
    ## normalization
    ph12_norm = mynorm(ph12)*2*math.pi
    ph23_norm = mynorm(ph23)*2*math.pi
    ph123 = exter_differ(ph12_norm, ph23_norm)
    ph123_norm = mynorm(ph123)*2*math.pi
    np.savetxt(path + 'ph123_norm.txt', ph123_norm, fmt="%.3f", delimiter=' ')
    print ("show image")
    img = np.cos(ph123_norm)
    np.savetxt(path + 'ph123_img.txt', img, fmt="%.3f", delimiter=' ')
    ph123 = ph123_norm
    ph1up_norm = unwrapp_phase(ph1,1)
    np.savetxt(path + 'ph1up_norm.txt', ph1up_norm, fmt="%.3f", delimiter=' ')


if runit:
    ph123 = np.loadtxt(path+'ph123.txt')
    phase1 = np.loadtxt(path+'phase1.txt')
    uph1 = unwrapp_phase(phase1, 1)
    np.savetxt(path + 'up_phase1.txt', uph1, fmt="%.3f", delimiter=' ')

    cv2.imshow('Image', uph1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if runit2:
    phase1 = relativePhase(1,height,width)
    np.savetxt(path + 'phase1_r.txt', phase1, fmt="%.3f", delimiter=' ')
    cv2.imshow("phase1", phase1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    phase2 = relativePhase(2,height,width)
    np.savetxt(path + 'phase2_r.txt', phase2, fmt="%.3f", delimiter=' ')
    phase3 = relativePhase(3, height, width)
    np.savetxt(path + 'phase3_r.txt', phase3, fmt="%.3f", delimiter=' ')
    ph12 = exter_differ(phase1, phase2)
    cv2.imshow("phase12", ph12/10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ph23 = exter_differ(phase2, phase3)
    cv2.imshow("ph23", ph23/10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ph13 = exter_differ(phase1, phase3)
    cv2.imshow("phase13", ph13 / 10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ph123 = exter_differ(ph12, ph23)
    cv2.imshow("phase123", ph123/10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ## change name
    np.savetxt(path + 'ph123_.txt', ph123, fmt="%.3f", delimiter=' ')
    #np.savetxt("./image/calculation/" + 'ph123_sur_r.txt', ph123, fmt="%.3f", delimiter=' ')
    print("saved ph123")
    uph1 = unwrap_phase1()
    np.savetxt(path + 'phase1_up_new.txt', uph1, fmt="%.3f", delimiter=' ')
    print("saved ph1 and show phase")
    cv2.imshow("ph1_up", uph1/1000)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    uph2 = unwrapp_phase(phase1,1)
    cv2.imshow("phase1_2", uph2 / 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.savetxt(path + 'phase1_up_org.txt', uph2, fmt="%.3f", delimiter=' ')
if runit3:
    phase1 = relativePhase(1, height, width)
    phase_uw = unwrapp(phase1)
    uv = np.cos(phase_uw)*127
    cv2.imshow('Image', uv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

