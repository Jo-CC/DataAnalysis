import numpy as np
from PIL import Image
import cv2
import pandas as pd
import csv
import math
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import math
# Set path to the image file
img_dir = r'C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/AIDA/Dataset/img_'
df33 = np.load('C:/Users/choo-chuan.jovin-ooi.VITROX/Downloads/label410.npy')
df = pd.read_csv('C:/Users/choo-chuan.jovin-ooi.VITROX/Downloads/diaxs.csv')
df2 = pd.read_csv('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/AIDA/XraySimulator/simulator/output.csv')
df4 = pd.read_csv('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/AIDA/XraySimulator/simulator/output_2.csv')

param = input("parameter to test: ")
param2 = input("(1) Top Right (2) Top Left (3) Bottom Right (4) Bottom Left: ")
# Load the image and convert to grayscale
y1 = df2[param]
y2 = df4.loc[0:49, param]
y = pd.concat([y1,y2],ignore_index=True)
print(y)
Aa = []
Bb = []
Cc = []
Dd = []
Ee = []
Ff = []

if param2 == '1':
    a, b, c, d, e, f = 9799, 9899, 9898, 9999, 9998, 9997
elif param2 == '2':
    a, b, c, d, e, f = 9700, 9800, 9801, 9900, 9901, 9902
elif param2 == '3':
    a, b, c, d, e, f = 97, 98, 99, 198, 199, 299
elif param2 == '4':
    a, b, c, d, e, f = 0, 1, 2, 100, 101, 200

#a,b,c,d,e,f = 97,98,99,198,199,299

for img_no in range(150):

    if img_no < 100:
        img_c0 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_35.png',cv2.IMREAD_GRAYSCALE)
        img_c1 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_40.png',cv2.IMREAD_GRAYSCALE)
        img_c2 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_45.png',cv2.IMREAD_GRAYSCALE)
        img_c3 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_50.png',cv2.IMREAD_GRAYSCALE)
        img_c4 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_55.png',cv2.IMREAD_GRAYSCALE)
        img_c5 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_60.png',cv2.IMREAD_GRAYSCALE)
        img_c6 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP/data_'+str(img_no)+'/reconstruction_plane_65.png',cv2.IMREAD_GRAYSCALE)

    else:
        img_no = img_no - 100
        img_c0 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_35.png',cv2.IMREAD_GRAYSCALE)
        img_c1 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_40.png',cv2.IMREAD_GRAYSCALE)
        img_c2 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_45.png',cv2.IMREAD_GRAYSCALE)
        img_c3 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_50.png',cv2.IMREAD_GRAYSCALE)
        img_c4 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_55.png',cv2.IMREAD_GRAYSCALE)
        img_c5 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_60.png',cv2.IMREAD_GRAYSCALE)
        img_c6 = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(img_no) + '/reconstruction_plane_65.png',cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(img_dir+str(img_no)+'.png')
    #r,g,b = cv2.split(img)

    img_array_c0 = np.array(img_c0)
    img_array_c1 = np.array(img_c1)
    img_array_c2 = np.array(img_c2)
    img_array_c3 = np.array(img_c3)
    img_array_c4 = np.array(img_c4)
    img_array_c5 = np.array(img_c5)
    img_array_c6 = np.array(img_c6)

    dft_c0 = np.fft.fft2(img_array_c0)
    dft_c1 = np.fft.fft2(img_array_c1)
    dft_c2 = np.fft.fft2(img_array_c2)
    dft_c3 = np.fft.fft2(img_array_c3)
    dft_c4 = np.fft.fft2(img_array_c4)
    dft_c5 = np.fft.fft2(img_array_c5)
    dft_c6 = np.fft.fft2(img_array_c6)

    dft_real_c0 = np.real(dft_c0)
    dft_imag_c0 = np.imag(dft_c0)
    dft_real_c1 = np.real(dft_c1)
    dft_imag_c1 = np.imag(dft_c1)
    dft_real_c2 = np.real(dft_c2)
    dft_imag_c2 = np.imag(dft_c2)
    dft_real_c3 = np.real(dft_c3)
    dft_imag_c3 = np.imag(dft_c3)
    dft_real_c4 = np.real(dft_c4)
    dft_imag_c4 = np.imag(dft_c4)
    dft_real_c5 = np.real(dft_c5)
    dft_imag_c5 = np.imag(dft_c5)
    dft_real_c6 = np.real(dft_c6)
    dft_imag_c6 = np.imag(dft_c6)

    dft_real_c0 = dft_real_c0.flatten()
    dft_imag_c0 = dft_imag_c0.flatten()
    dft_real_c1 = dft_real_c1.flatten()
    dft_imag_c1 = dft_imag_c1.flatten()
    dft_real_c2 = dft_real_c2.flatten()
    dft_imag_c2 = dft_imag_c2.flatten()
    dft_real_c3 = dft_real_c3.flatten()
    dft_imag_c3 = dft_imag_c3.flatten()
    dft_real_c4 = dft_real_c4.flatten()
    dft_imag_c4 = dft_imag_c4.flatten()
    dft_real_c5 = dft_real_c5.flatten()
    dft_imag_c5 = dft_imag_c5.flatten()
    dft_real_c6 = dft_real_c6.flatten()
    dft_imag_c6 = dft_imag_c6.flatten()

    A=(dft_real_c0[a],dft_imag_c0[a],dft_real_c1[a],dft_imag_c1[a],dft_real_c2[a],dft_imag_c2[a],dft_real_c3[a],dft_imag_c3[a],
       dft_real_c4[a],dft_imag_c4[a],dft_real_c5[a],dft_imag_c5[a],dft_real_c6[a],dft_imag_c6[a])
    B=(dft_real_c0[b],dft_imag_c0[b],dft_real_c1[b],dft_imag_c1[b],dft_real_c2[b],dft_imag_c2[b],dft_real_c3[b],dft_imag_c3[b],
       dft_real_c4[b],dft_imag_c4[b],dft_real_c5[b],dft_imag_c5[b],dft_real_c6[b],dft_imag_c6[b])
    C=(dft_real_c0[c],dft_imag_c0[c],dft_real_c1[c],dft_imag_c1[c],dft_real_c2[c],dft_imag_c2[c],dft_real_c3[c],dft_imag_c3[c],
       dft_real_c4[c],dft_imag_c4[c],dft_real_c5[c],dft_imag_c5[c],dft_real_c6[c],dft_imag_c6[c])

    D=(dft_real_c0[d],dft_imag_c0[d],dft_real_c1[d],dft_imag_c1[d],dft_real_c2[d],dft_imag_c2[d],dft_real_c3[d],dft_imag_c3[d],
       dft_real_c4[d],dft_imag_c4[d],dft_real_c5[d],dft_imag_c5[d],dft_real_c6[d],dft_imag_c6[d])
    E=(dft_real_c0[e],dft_imag_c0[e],dft_real_c1[e],dft_imag_c1[e],dft_real_c2[e],dft_imag_c2[e],dft_real_c3[e],dft_imag_c3[e],
       dft_real_c4[e],dft_imag_c4[e],dft_real_c5[e],dft_imag_c5[e],dft_real_c6[e],dft_imag_c6[e])
    F=(dft_real_c0[f],dft_imag_c0[f],dft_real_c1[f],dft_imag_c1[f],dft_real_c2[f],dft_imag_c2[f],dft_real_c3[f],dft_imag_c3[f],
       dft_real_c4[f],dft_imag_c4[f],dft_real_c5[f],dft_imag_c5[f],dft_real_c6[f],dft_imag_c6[f])

    Aa.append(A)
    Bb.append(B)
    Cc.append(C)
    Dd.append(D)
    Ee.append(E)
    Ff.append(F)

all_con = np.concatenate((Aa,Bb,Cc,Dd,Ee,Ff), axis=1)
print(all_con)

lr = lm.LinearRegression().fit(all_con,y)
print(lr.score(all_con,y))

Aa2 = []
Bb2 = []
Cc2 = []
Dd2 = []
Ee2 = []
Ff2 = []

for imgx in range(50,100):

    img_c0b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_35.png',cv2.IMREAD_GRAYSCALE)
    img_c1b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_40.png',cv2.IMREAD_GRAYSCALE)
    img_c2b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_45.png',cv2.IMREAD_GRAYSCALE)
    img_c3b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_50.png',cv2.IMREAD_GRAYSCALE)
    img_c4b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_55.png',cv2.IMREAD_GRAYSCALE)
    img_c5b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_60.png',cv2.IMREAD_GRAYSCALE)
    img_c6b = cv2.imread('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/PPP2/data_' + str(imgx) + '/reconstruction_plane_65.png',cv2.IMREAD_GRAYSCALE)

    img_array_c0b = np.array(img_c0b)
    img_array_c1b = np.array(img_c1b)
    img_array_c2b = np.array(img_c2b)
    img_array_c3b = np.array(img_c3b)
    img_array_c4b = np.array(img_c4b)
    img_array_c5b = np.array(img_c5b)
    img_array_c6b = np.array(img_c6b)

    dft_c0b = np.fft.fft2(img_array_c0b)
    dft_c1b = np.fft.fft2(img_array_c1b)
    dft_c2b = np.fft.fft2(img_array_c2b)
    dft_c3b = np.fft.fft2(img_array_c3b)
    dft_c4b = np.fft.fft2(img_array_c4b)
    dft_c5b = np.fft.fft2(img_array_c5b)
    dft_c6b = np.fft.fft2(img_array_c6b)

    dft_real_c0b = np.real(dft_c0b)
    dft_imag_c0b = np.imag(dft_c0b)
    dft_real_c1b = np.real(dft_c1b)
    dft_imag_c1b = np.imag(dft_c1b)
    dft_real_c2b = np.real(dft_c2b)
    dft_imag_c2b = np.imag(dft_c2b)
    dft_real_c3b = np.real(dft_c3b)
    dft_imag_c3b = np.imag(dft_c3b)
    dft_real_c4b = np.real(dft_c4b)
    dft_imag_c4b = np.imag(dft_c4b)
    dft_real_c5b = np.real(dft_c5b)
    dft_imag_c5b = np.imag(dft_c5b)
    dft_real_c6b = np.real(dft_c6b)
    dft_imag_c6b = np.imag(dft_c6b)

    dft_real_c0b = dft_real_c0b.flatten()
    dft_imag_c0b = dft_imag_c0b.flatten()
    dft_real_c1b = dft_real_c1b.flatten()
    dft_imag_c1b = dft_imag_c1b.flatten()
    dft_real_c2b = dft_real_c2b.flatten()
    dft_imag_c2b = dft_imag_c2b.flatten()
    dft_real_c3b = dft_real_c3b.flatten()
    dft_imag_c3b = dft_imag_c3b.flatten()
    dft_real_c4b = dft_real_c4b.flatten()
    dft_imag_c4b = dft_imag_c4b.flatten()
    dft_real_c5b = dft_real_c5b.flatten()
    dft_imag_c5b = dft_imag_c5b.flatten()
    dft_real_c6b = dft_real_c6b.flatten()
    dft_imag_c6b = dft_imag_c6b.flatten()

    A2 = (dft_real_c0b[a], dft_imag_c0b[a], dft_real_c1b[a], dft_imag_c1b[a], dft_real_c2b[a], dft_imag_c2b[a], dft_real_c3b[a],dft_imag_c3b[a],
         dft_real_c4b[a], dft_imag_c4b[a], dft_real_c5b[a], dft_imag_c5b[a], dft_real_c6b[a], dft_imag_c6b[a])

    B2 = (dft_real_c0b[b], dft_imag_c0b[b], dft_real_c1b[b], dft_imag_c1b[b], dft_real_c2b[b], dft_imag_c2b[b], dft_real_c3b[b],dft_imag_c3b[b],
         dft_real_c4b[b], dft_imag_c4b[b], dft_real_c5b[b], dft_imag_c5b[b], dft_real_c6b[b], dft_imag_c6b[b])

    C2 = (dft_real_c0b[c], dft_imag_c0b[c], dft_real_c1b[c], dft_imag_c1b[c], dft_real_c2b[c], dft_imag_c2b[c],dft_real_c3b[c], dft_imag_c3b[c],
         dft_real_c4b[c], dft_imag_c4b[c], dft_real_c5b[c], dft_imag_c5b[c], dft_real_c6b[c], dft_imag_c6b[c])

    D2 = (dft_real_c0b[d], dft_imag_c0b[d], dft_real_c1b[d], dft_imag_c1b[d], dft_real_c2b[d], dft_imag_c2b[d],dft_real_c3b[d], dft_imag_c3b[d],
         dft_real_c4b[d], dft_imag_c4b[d], dft_real_c5b[d], dft_imag_c5b[d], dft_real_c6b[d], dft_imag_c6b[d])

    E2 = (dft_real_c0b[e], dft_imag_c0b[e], dft_real_c1b[e], dft_imag_c1b[e], dft_real_c2b[e], dft_imag_c2b[e],dft_real_c3b[e], dft_imag_c3b[e],
         dft_real_c4b[e], dft_imag_c4b[e], dft_real_c5b[e], dft_imag_c5b[e], dft_real_c6b[e], dft_imag_c6b[e])

    F2 = (dft_real_c0b[f], dft_imag_c0b[f], dft_real_c1b[f], dft_imag_c1b[f], dft_real_c2b[f], dft_imag_c2b[f],dft_real_c3b[f], dft_imag_c3b[f],
         dft_real_c4b[f], dft_imag_c4b[f], dft_real_c5b[f], dft_imag_c5b[f], dft_real_c6b[f], dft_imag_c6b[f])

    Aa2.append(A2)
    Bb2.append(B2)
    Cc2.append(C2)
    Dd2.append(D2)
    Ee2.append(E2)
    Ff2.append(F2)

all_con2 = np.concatenate((Aa2,Bb2,Cc2,Dd2,Ee2,Ff2), axis=1)
y_predd = []
f = []
for test in all_con2:

    y_pred = lr.predict(test.reshape(1, -1))
    print("YPRED:", y_pred)
    y_predd.append(y_pred[0])
    f.append(y_pred[0] ** 2)

ytruee = []
y_true = df4.loc[50:100, param]
for aysd in y_true:
    ytruee.append(aysd)
df = pd.DataFrame(list(zip(y_true, y_predd)), columns=['ytrue', 'ypred'])
print(lr.score(all_con2,ytruee))
#plt.scatter(y_true,y_predd)

plt.plot(ytruee,'o', label='Y-True')
plt.plot(y_predd,'o',label='Y-Pred')
plt.xlabel('Imageset Number')
plt.ylabel(param)
xticks = np.arange(-1, 50, 1)
plt.xticks(xticks)
plt.grid(axis='x')
titles = "Predicted vs. True Values for " + param
# Add a title and legend
plt.title(titles)
plt.legend()

# Show the plot
plt.show()
df.to_csv('C:/Users/choo-chuan.jovin-ooi.VITROX/Desktop/AIDA/example12.csv')

plt.plot(ytruee, y_predd ,'o')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.show()