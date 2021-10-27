import cv2
import matplotlib.pylab as plt
import  numpy as np
import copy

def myAdapMedia(image_input, max_wind):
    pad_max = np.floor(max_wind/2)#5
    cent_max = int(pad_max + 1) #6
    image_padded = np.pad(image_input, (int(pad_max), int(pad_max)), 'reflect')
    R,C = image_input.shape
    image_output = copy.copy(image_input)
    for i in range(R):
        for j in range(C):
            patch_curr = image_padded[i:i+max_wind, j:j+max_wind]
            k = 3;
            if image_input[i,j] ==0 or image_input[i,j] ==255:
                while k<=max_wind:
                    patch_currCrop = image_padded[cent_max - int(np.floor(k/2)):cent_max + int(np.floor(k/2))+1,cent_max - int(np.floor(k/2)):cent_max + int(np.floor(k/2)+1)]
                   # print(patch_currCrop)
                    median_val = np.median(patch_currCrop)
                    if median_val!=0 and median_val!=255 or k == max_wind:
                        image_output[i,j] = median_val
                        break
                    else:
                        k=k+1
                        print(k)

    # cv2.imshow("IM",cv2.hconcat([image_input,image_output]))
    # cv2.waitKey(-2)

im =cv2.imread('download.jpg')
grey_image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
myAdapMedia(grey_image,11)