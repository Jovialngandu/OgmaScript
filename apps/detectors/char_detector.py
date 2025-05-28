import cv2
import numpy as np
import utils.image_utils as image_utils
from ..app import App

class charDetector(App):

    def __init__(self,word_img,pre_name):
        super().__init__()
        self.word_img=word_img
        self.pre_name=pre_name

    def process(self):

        img = self.word_img
        img_copy=image_utils.copyImage(img)
        img_gray=image_utils.imageToGray(img_copy)
        back_color=image_utils.getBackground(img_gray)
        kernel=np.ones((40,1),np.uint8)
        chars_regions=[]
        chars=[]
        rois=[]

        if back_color > 28 and back_color <200:
            _, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
        elif back_color>200: 
            img_gray=cv2.bitwise_not(img_gray)
            _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
        else:
            _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

        
        img_dilated=cv2.dilate(thresh,kernel,iterations=1)

        contours,_=cv2.findContours(img_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            chars_regions.append(cv2.boundingRect(contour))
    

        chars_regions_sorted=sorted(chars_regions,key=lambda x:x[0])
       
        num=0
 
        for char in chars_regions_sorted:

            x,y,w,h=char
            if h>=20 and w>0.5:
                chars.append(char)
                num=num+1
                roi=img_gray[y:y+h,x:x+w]
                cv2.imwrite(f"tmp/chars/{self.pre_name}char{num}.png",roi)
                rois.append({"roi":roi,"name":f"{self.pre_name}char{num}"})
                cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,0),1)
        
        #image_utils.show(image_utils.resizeImage(img_copy,(1280,128)))


        return {"chars":chars,"regions":chars_regions_sorted,"rois":rois}

