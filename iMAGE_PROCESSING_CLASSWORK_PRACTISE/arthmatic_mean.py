import cv2
import numpy as np
def AM_mean(image,wind_size):
  i,j=wind_size
  div=i*j
  sum=0
  gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)/255
  row,col=np.shape(gray_image)
  copy_image = np.zeros((row, col))
  image_padded = np.pad(gray_image,(1,1),'symmetric')
  patch_curr=[]
  for a in range(row):
    for b in range(col):
       patch_curr = image_padded[a:a + i, b:b + j]
       val = np.sum(patch_curr)/9
       copy_image[a,b]=((val))
  cv2.imshow("copy",copy_image)
  cv2.imshow("oringal",gray_image)
  cv2.waitKey(-2)

im = cv2.imread('noisee.jpg')
#im = cv2.imread('download.jpg')
AM_mean(im,(3,3))