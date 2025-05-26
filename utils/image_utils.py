import cv2
import numpy as np


def copyImage(img):
 
    return  img.copy()


def getBackground(img):

    hist = cv2.calcHist([img], [0], None, [256], [0,256])
   
    return int(np.argmax(hist))

def imageToGray(img):

    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def invertImage(img):
    
    return cv2.bitwise_not(img)

def readImage(path):

    return cv2.imread(path)


def resizeImage(img,size=(24,24)):

    #size will be a tupple like this (32,32)
  
    return cv2.resize(img, (1280,128))


def show(img): 

    cv2.imshow("my img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





