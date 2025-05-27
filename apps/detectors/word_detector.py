import cv2
import numpy as np
import utils.image_utils as image_utils
from ..app import App

class WordDetector(App):
    def __init__(self,line_img=None,number=None):
        super().__init__()
        self.line_img=line_img
        self.number=number

    def process(self):

        length_result=self.getLengthSeparator()
        
        word_sep_length=length_result["word"]

        if not word_sep_length or word_sep_length >= 8:
            
            return self.secondtDetector(self.number)
           
        else:

             #return self.secondtDetector(self.number)
             return self.firstDetector(self.number)
          
    
    def getLengthSeparator(self,img=None):
        
        if not img:
            img=self.line_img

        kernel=np.ones((10,2),np.uint8)
        img_copy=image_utils.copyImage(img)
        img_resized=image_utils.resizeImage(img_copy, (640,64))
        img_gray=image_utils.imageToGray(img_resized)
        #print("affichage imgray")
        #image_utils.show(img_gray)

        _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        #image_utils.show(thresh)
        
        img_dilated=cv2.dilate(thresh,kernel,iterations=1)
        #image_utils.show(img_dilated)
        _, th = cv2.threshold(img_dilated, 127, 255, cv2.THRESH_BINARY)
        #image_utils.show(th)
        cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        b=[]
        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            if h>=20 and w>1.5:
                b.append((x,w))
                #cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,0),1)
        #image_utils.show(img_copy)
        
        space=[]
        nbr=len(b)
        b=sorted(b,key=lambda x:x[0])
        for i in range (nbr-1):
            l1=b[i][0]+b[i][1]
            l2=b[i+1][0]
            space.append(l2-l1)

        seuil=int(np.ceil((5*nbr)/100))
        #print(seuil)
        filt=list(filter(lambda x:x>0 and x < 10,space))
        mins=sorted(filt)[:seuil] 

        if len(filt)>0 :

            Length_char=max(set(mins),key=mins.count)            
            maxs= sorted(filt,reverse=True)[:seuil]
            Length_word= np.mean(maxs) 

            return {"word":Length_word,"char":Length_char}
        else:
            return {"word":None,"char":None}


    def setLine(self,line_img):

        self.line_img=line_img



    def firstDetector(self,number):
         
        words_regions=[]
        words=[]
        img=image_utils.readImage(f"tmp/ligne/ligne{number}.png")
        img2=image_utils.readImage(f"tmp/lignes/ligne{number}.png")
        
        img_copy=image_utils.copyImage(img)
        img_resized=image_utils.resizeImage(img_copy, (640,64))
        img_gray=image_utils.imageToGray(img_resized)

        img_copy2=image_utils.copyImage(img2)
        img_resized2=image_utils.resizeImage(img_copy2, (640,64))
        img_gray2=image_utils.imageToGray(img_resized2)

        kernel=np.ones((18,3),np.uint8)
        ###j'ai add 2 iterations
        img_dilated=cv2.dilate(img_gray,kernel,iterations=1)
        #image_utils.show(img_dilated)

        contours,_=cv2.findContours(img_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                

        for contour in contours:
            words_regions.append(cv2.boundingRect(contour))
        
        words_regions_sorted=sorted(words_regions,key=lambda x:x[0])
                
        
        num=0
        for region in words_regions_sorted:
            x,y,w,h=region

            if h>=20:
                num=num+1
                roi=img_gray2[y:y+h,x:x+w]
                words.append(roi)
                cv2.imwrite(f"tmp/words/l{number}word{num}.png",roi)
                #cv2.rectangle(img_resized2,(x,y),(x+w,y+h),(255,0,0),1)
        #image_utils.show(img_resized2)
        return {"words":words,"regions":words_regions}

    def secondtDetector(self,number):

        words_regions=[]
        words=[]
        img=image_utils.readImage(f"tmp/ligne/ligne{number}.png")
        img2=image_utils.readImage(f"tmp/lignes/ligne{number}.png")
        
        img_copy=image_utils.copyImage(img)
        img_resized=image_utils.resizeImage(img_copy, (640,64))
        img_gray=image_utils.imageToGray(img_resized)

        img_copy2=image_utils.copyImage(img2)
        img_resized2=image_utils.resizeImage(img_copy2, (640,64))

        kernel=np.ones((18,6),np.uint8)
        ###j'ai add 2 iterations
        img_dilated=cv2.dilate(img_gray,kernel,iterations=1)

        contours,_=cv2.findContours(img_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                

        for contour in contours:
            words_regions.append(cv2.boundingRect(contour))
        
        words_regions_sorted=sorted(words_regions,key=lambda x:x[0])
                
        
        num=0
        for region in words_regions_sorted:
            x,y,w,h=region

            if h>=20:
                num=num+1
                roi=img_resized2[y:y+h,x:x+w]
                words.append(roi)
                cv2.imwrite(f"tmp/words/l{number}word{num}.png",roi)
                #cv2.rectangle(img_resized2,(x,y),(x+w,y+h),(255,0,0),1)
        #image_utils.show(img_resized2)
        return {"words":words,"regions":words_regions}
        