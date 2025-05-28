import utils.image_utils as image_utils
import utils.ogma_utils as ogma_utils
from apps.detectors.line_detector  import LineDetector
from apps.detectors.word_detector import WordDetector
from apps.detectors.char_detector import charDetector

import cv2
a=image_utils.readImage("tmp/1.png")
b=image_utils.copyImage(a)


#image_utils.show(b)

c=LineDetector(b)

_,result_line=c.process().values()

for i in range (len(result_line)):
    #print(f"tmp/ligne/ligne{i+1}.png")
    d=WordDetector(image_utils.readImage(f"tmp/ligne/ligne{i+1}.png"),number=i+1)
    words=d.process()["words"]
    for j in range (len(words)):
        c=charDetector(word_img=image_utils.readImage(f"tmp/words/l{i+1}word{j+1}.png"),pre_name=f"l{i+1}word{j+1}")
        f=c.process()
        rois=f["rois"]

        for  k in range(len(rois)):
            #image_utils.show(rois[k]["roi"])
            ogma_utils.ogmaCharPreprocessing(img=rois[k]["roi"],char_name=rois[k]["name"])

#image_utils.show(b)


#print(cv2.version.opencv_version)