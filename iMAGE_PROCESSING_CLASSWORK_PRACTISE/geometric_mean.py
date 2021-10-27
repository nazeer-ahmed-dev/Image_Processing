import cv2
import numpy as np
def GM_mean(image,wind_size):
  i,j=wind_size
  div=i*j
  sum=0
  gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  row,col=np.shape(gray_image)
  copy_image = np.zeros((row, col))
  image_padded = np.pad(gray_image,(1,1),'symmetric').astype(float)/255
  patch_curr=[]
  for a in range(row):
    for b in range(col):  
       patch_curr = image_padded[a:a + i, b:b + j]
       q,w = np.shape(patch_curr)
       m=1
       for s in range(w):
         for d in range(s):
           m=m*patch_curr[s,d]
           power1 = np.power(m,(1/div))
           copy_image[a,b]=power1
  cv2.imshow("copy",copy_image)
  cv2.imshow("oringal",gray_image)
  cv2.waitKey(-2)

im = cv2.imread(a)
GM_mean(im,(3,3))