import cv2
import numpy as np
def Image_pass(image):
    image_grey = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)).astype(float)/255
    image_padded=np.pad(image_grey,(400,300),'symmetric')
    #cv2.imshow("Image for Padding " ,image_padded)
    cv2.imshow("im",image_padded)
    cv2.waitKey(-1)


im=cv2.imread('padding.jpeg')
Image_pass(im)
