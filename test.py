import utils.image_utils as img_utils

a=img_utils.readImage("tmp/1.png")
b=img_utils.copyImage(a)
b=img_utils.imageToGray(b)


print(b)


#img_utils.show(img_utils.invertImage(b))





