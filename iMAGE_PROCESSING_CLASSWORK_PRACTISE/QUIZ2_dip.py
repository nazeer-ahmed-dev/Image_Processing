import cv2
import numpy as np
#import matplotlib.pylab as plt
def webam():
    CameraObject = cv2.VideoCapture(0)
    fps = CameraObject.get(cv2.CAP_PROP_FPS)
    size = ((CameraObject.get(cv2.CAP_PROP_FRAME_WIDTH),(cv2.CAP_PROP_FRAME_HEIGHT)))
    success , frame = CameraObject.read()
    num_of_Frames = fps*1500-1
    while success and num_of_Frames > 0:
       #cv2.imshow("vedio",frame)
       concatination(frame)
       # grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       success,frame=CameraObject.read()
       cv2.waitKey(1)
       num_of_Frames-=1
def SobelEdgeDetection1(img):
    xsobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ysoble = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filtImage_x = cv2.filter2D(img,-1,xsobel)
    filtImage_y = cv2.filter2D(img, -1, ysoble)
    return filtImage_y
    # cv2.imshow("output",filtImage_x)
    # cv2.waitKey(-1)
#grey_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#SobelEdgeDetection(grey_image)
def SobelEdgeDetection2(img):
    xsobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ysoble = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filtImage_x = cv2.filter2D(img,-1,xsobel)
    filtImage_y = cv2.filter2D(img, -1, ysoble)
    return filtImage_x
def laplacianEdgeDetecotor(img):
    filt_image = cv2.Laplacian(img,ksize=1,scale=None,ddepth=cv2.CV_16S)
    filt_image = cv2.convertScaleAbs(filt_image)
    return  filt_image
    # cv2.imshow("laplace",filt_image)
    # cv2.waitKey(-1)
#laplacianEdgeDetecotor(grey_image)
def cannydetectro(img):
    filt_image = cv2.Canny(img,threshold1=10,threshold2=200)
    return  filt_image
#cannydetectro(grey_image)
def concatination(im):
    grey_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im1=SobelEdgeDetection1(grey_image)
    im3=laplacianEdgeDetecotor(grey_image)
    im2=SobelEdgeDetection1(grey_image)
    hconcat_img1 = cv2.hconcat([grey_image,im1])
    hconcat_img2 = cv2.hconcat([im2,im3])
    vconcat = cv2.vconcat([hconcat_img1,hconcat_img2])
    cv2.imshow("output",vconcat)
    cv2.waitKey(1)
image = cv2.imread("padding.tif")
#concatination(image)
webam()
