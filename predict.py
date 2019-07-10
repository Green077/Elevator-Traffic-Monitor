import numpy as np
import cv2
from os.path import dirname, join, basename
import sys
from glob import glob

bin_n = 16*16

def hog(img):
    x_pixel,y_pixel=194,259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    
    bin_cells = bins[:int(x_pixel/2),:int(y_pixel/2)], bins[int(x_pixel/2):,:int(y_pixel/2)], bins[:int(x_pixel/2),int(y_pixel/2):], bins[int(x_pixel/2):,int(y_pixel/2):]
    mag_cells = mag[:int(x_pixel/2),:int(y_pixel/2)], mag[int(x_pixel/2):,:int(y_pixel/2)], mag[:int(x_pixel/2),int(y_pixel/2):], mag[int(x_pixel/2):,int(y_pixel/2):]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  
    return hist

svm = cv2.ml.SVM_load('/a092017/iot/final_project/trained_model.xml')

test_temp=[]
for fn in glob(join(dirname(__file__)+'/predict2', '*.jpg')):
    img=cv2.imread(fn,0)
    test_temp.append(img)

hogdata = map(hog,test_temp)
testData = np.float32(list(hogdata)).reshape(-1,bin_n*4)
result = svm.predict(testData)
print (result)




