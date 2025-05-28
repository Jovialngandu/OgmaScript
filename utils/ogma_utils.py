import cv2
import numpy as np
import utils.image_utils as image_utils


def  ogmaCharPreprocessing(img,char_name):

    img=cv2.resize(img,(28,28))
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray =cv2.bitwise_not(img)
    thresh = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)  # Augmente le (3, 3) pour épaissir encore plus
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Trouver le plus grand contour (bloc de texte)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = gray[y:y+h, x:x+w]
    cropped=cv2.resize(cropped, (90, 90))
    #image_utils.show(cropped )
    padded = cv2.copyMakeBorder(cropped, 20, 20, 50, 50, cv2.BORDER_CONSTANT, value=255)
    #image_utils.show(padded)

    resized = cv2.resize(padded, (256, 256))  # à adapter selon ton input

    smoothed = cv2.GaussianBlur(resized , (3, 3), 0)

    kernel = np.ones((1, 1), np.uint8)  # Ajuste la taille si nécessaire
    dilated = cv2.dilate(smoothed , kernel, iterations=2)

    laplacian = cv2.Laplacian(dilated, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(dilated - 0.7 * laplacian)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharpened)


    seuil = 200 
    _, mask = cv2.threshold(enhanced, seuil, 255, cv2.THRESH_BINARY)
    #image_utils.show(mask)

    cv2.imwrite(f'tmp/chars_final/{char_name}.png',mask)

    return mask




