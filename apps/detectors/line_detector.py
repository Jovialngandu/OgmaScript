import cv2
import numpy as np
import utils.image_utils as image_utils
from ..app import App


class LineDetector(App):

    def __init__(self,img):

        super().__init__()
        self.img=img



    def process(self):

        image_gray=image_utils.imageToGray(self.img)
        background_color=image_utils.getBackground(self.img)
        kernel=np.ones((1,30),np.uint8)
        lignes=[]
        result=[]
       
        if background_color < 128:
            #image_gray= image_utils.invertImage(image_gray)
            _, thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)

        else :
            _, thresh = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV)

        
        image_dilated=cv2.dilate(thresh,kernel,iterations=50)

        contours,_=cv2.findContours(image_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            lignes.append(cv2.boundingRect(contour))
        
        lignes_sorted=sorted(lignes,key=lambda x:x[1])

        num=0
        for ligne in lignes_sorted :
             
             x,y,w,h=ligne
            
             if h>=2:
                num=num+1
                roi=image_gray[y:y+h,x:x+w]
                roi2=thresh[y:y+h,x:x+w]
                result.append(roi)
                cv2.imwrite(f"tmp/lignes/ligne{num}.png",roi)
                cv2.imwrite(f"tmp/ligne/ligne{num}.png",roi2)
             
                cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),1)

        
        
        return {"lignes":lignes_sorted,"result":result}
