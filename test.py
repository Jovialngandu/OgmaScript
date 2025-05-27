import utils.image_utils as image_utils
from apps.detectors.line_detector  import LineDetector
from apps.detectors.word_detector import WordDetector
import cv2
a=image_utils.readImage("tmp/5.png")
b=image_utils.copyImage(a)


#image_utils.show(b)

c=LineDetector(b)

_,result_line=c.process().values()

for i in range (len(result_line)):
    #print(f"tmp/ligne/ligne{i+1}.png")
    d=WordDetector(image_utils.readImage(f"tmp/ligne/ligne{i+1}.png"),number=i+1)
    d.process()


#image_utils.show(b)


#print(cv2.version.opencv_version)