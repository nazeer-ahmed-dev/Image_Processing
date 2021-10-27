import cv2
import numpy as np
def AM_mean(image,wind_size):
  size_pad = np.floor(wind_size/2)
  image_padded=np.pad(image,(int(size_pad),int(size_pad)),'reflect')
  image_output = image
  R,C = image.shape
  #print(i)
  #R,C = np.shape(image)
  for a in range(R):
    for b in range(C):
       patch_curr = image_padded[a:a + wind_size, b:b + wind_size]
       array1 = np.array(patch_curr)
       k = array1.flatten()
       c=np.sort(k)
       d=c[2:-2]
       val = np.mean(d)
       image_output[a,b]=val
       #val = np.mean(c)
       # copy_image[a][b]=((val))
  cv2.imshow("copy",image_output)
  print(b)
  print(c)
  print(d)
  print(val)
  cv2.imshow("copy",image_output)
  cv2.waitKey(-2)

im = cv2.imread('noisee.jpg')
grey = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
AM_mean(grey,3)