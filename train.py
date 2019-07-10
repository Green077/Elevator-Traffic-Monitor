import numpy as np
import cv2
from os.path import dirname, join, basename
import sys
from glob import glob


bin_n = 16*16 # Number of bins

def hog(img):
    x_pixel=194
    y_pixel=259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:int(x_pixel/2),:int(y_pixel/2)], bins[int(x_pixel/2):,:int(y_pixel/2)], bins[:int(x_pixel/2),int(y_pixel/2):], bins[int(x_pixel/2):,int(y_pixel/2):]
    mag_cells = mag[:int(x_pixel/2),:int(y_pixel/2)], mag[int(x_pixel/2):,:int(y_pixel/2)], mag[:int(x_pixel/2),int(y_pixel/2):], mag[int(x_pixel/2):,int(y_pixel/2):]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
img={}
num=0
for fn in glob(join(dirname(__file__)+'/train2', '*.jpg')):
    img[num] = cv2.imread(fn,0)
    num=num+1
positive=num

trainpic=[]
for i in img:

    trainpic.append(img[i])

hogdata = map(hog,trainpic)
trainData = np.float32(list(hogdata)).reshape(-1,bin_n*4)
responses = np.float32(np.repeat(1.0,trainData.shape[0])[:,np.newaxis])
j = 0
while( j < trainData.shape[0]):
    responses[j] = j + 10
    j += 1
responses = responses.astype(int)

svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setGamma(5.383)

svm.setC(2.67)
svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
result = svm.predict(trainData)
svm.save('trained_model.xml')