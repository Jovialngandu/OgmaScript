import utils.image_utils as image_utils
from apps.detectors.line_detector  import LineDetector

a=image_utils.readImage("tmp/1.png")
b=image_utils.copyImage(a)


#img_utils.show(img_utils.invertImage(b))

a=LineDetector(b)
a.process()


image_utils.show(b)


