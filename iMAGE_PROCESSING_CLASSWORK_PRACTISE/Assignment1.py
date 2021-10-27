import cv2
import numpy as np
import matplotlib.pylab as plt
import os
from PIL import Image, ImageOps

#1.	Read the image named “Example1.png”, subtract 50 from all the intensities using for loop, write it with the name imageSub.jpg.
def task1(image):
    copy_image=np.array(image)
    i,j = np.shape(image)
    #print(i,j)
    for k in range(i):
        for l in range(j):
            copy_image[k][l]=image[k][l]-50

    cv2.imshow("image",copy_image)
    cv2.waitKey(-3)


image = cv2.imread("Example1.png")
grey_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#task1(grey_image)

# 2.Read an image, subtract 50 from those regions that have rows and columns in the ranges of 1-50 and 87-133 respectively. Write the image with the name of imageSubSel.jpg.

def task2(image):
    copy_image = np.array(image)
    i, j = np.shape(image)
    # print(i,j)
    for k in range(1,50):
        for l in range(87,133):
            copy_image[k][l] = image[k][l] - 50

    cv2.imshow("image", copy_image)
    cv2.waitKey(-3)

#task2(grey_image)

# 3. Read an image and subtract 50 from those pixels that have intensities greater than 230.
def task3(image):
    copy_image = np.array(image)
    i, j = np.shape(image)
    # print(i,j)
    for k in range(1,50):
        for l in range(87,133):
            if(copy_image[k][l]>230):
                copy_image[k][l] = image[k][l] - 50

    cv2.imshow("image", copy_image)
    cv2.waitKey(-3)

#task3(grey_image)

# 4.Create a function that takes an image as input and produce mirror effect such as shown in figure below
def mirror_effect(img):
    mirror =img.transpose(Image.FLIP_LEFT_RIGHT)
    dst = Image.new('RGB', (img.width + mirror.width, img.height))
    dst.paste(img, (0, 0))
    dst.paste(mirror, (img.width, 0))
    dst.show()
image_for_mirror=Image.open("waseem.png")
#mirror_effect(image_for_mirror)

#Histogram Processing

#1.	Write a function named myImHist that is similar to imhist. It should take a grayscale image as input, count the frequencies of each intensity, calculate probability distributive function(pdf), plot pdf vs each intensity and return values of pdf.

def myImHist(image):
    #image1=[[1,2,3],[2,1,3],[1,3,3]]
    min_value = np.min(image)
    max_value = np.max(image)
    j,k = np.shape(image)
    frequency = []
    pdf=[]
    count=0
    total_pixel=0
    for i in range(min_value,max_value+1):
        for a in range(j):
            for b in range(k):
                if image[a][b]==i:
                    print(i)
                    count=count+1
                    print(count)
        frequency.append((i,count))
        total_pixel=total_pixel+count
        count=0
    for intensity , no_of_repeat in frequency:
        pdf.append(no_of_repeat/total_pixel)

    plt.stem(pdf)
    plt.show()
image1 = cv2.imread("download.jpg")
grey_image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#myImHist(grey_image1)

# 2. Perform histogram equalization iteratively three times and see if there is any difference in the result or not. Analyze the outcome with the help of some examples.2.	Perform histogram equalization iteratively three times and see if there is any difference in the result or not. Analyze the outcome with the help of some examples.

def Histogram_Equalization(image):
    # image1=[[1,2,3],[2,1,3],[1,3,3]]
    min_value = np.min(image)
    max_value = np.max(image)
    j, k = np.shape(image)
    frequency = []
    pdf = []
    cdf = []
    count = 0
    total_pixel = 0
    sum1=0
    for i in range(min_value, max_value + 1):
        for a in range(j):
            for b in range(k):
                if image[a][b] == i:
                    print(i)
                    count = count + 1
                    print(count)
        frequency.append((i, count))
        total_pixel = total_pixel + count
        count = 0
    for intensity, no_of_repeat in frequency:
        pdf.append(no_of_repeat / total_pixel)
    for value in pdf:
        sum1 = sum1 + value
        cdf.append(sum1)
    plt.stem(cdf)
    plt.show()
image1 = cv2.imread("download.jpg")
grey_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#Histogram_Equalization(grey_image1)

# 3.Find a pre-defined python function that perform histogram matching. Test a few examples for histogram matching using that function.

def predefind_hist_equ(image):
    equ = cv2.equalizeHist(image)
    cv2.imshow("df",equ)
    cv2.waitKey(-1)
image1 = cv2.imread("download.jpg")
grey_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#predefind_hist_equ(grey_image1)

# 4.Create your own function of histogram matching and name it as myHistMatch.
def myHistMatch(input_hist,refernce_hist):
    frequency_a = []
    frequency_b = []
    pdf_a =[]
    pdf_b=[]
    cdf_a=[]
    cdf_b=[]
    cdf_a_X_max_val=[]
    cdf_b_X_max_val=[]
    Round_of_a=[]
    Round_of_b=[]
    New_values_a=[]
    total_pixel_a=0
    total_pixel_b=0
    count_a=0
    count_b=0
    sum_a=0
    sum_b=0
    min_a=np.min(input_hist)
    max_a=np.max(input_hist)
    min_b=np.min(refernce_hist)
    max_b=np.max(refernce_hist)
    i_a,j_a=np.shape(input_hist)
    i_b,j_b=np.shape(refernce_hist)

   #finding frequency of Hist a
    for i in range(min_a,max_a):
        for a in range(i_a):
            for b in range(j_a):
                if input_hist[a][b] == i:
                    count_a = count_a + 1
        frequency_a.append((i,count_a))
        total_pixel_a = total_pixel_a + count_a
        count_a = 0
    print(frequency_a)

    # finding frequency of Hist b
    for i in range(min_b,max_b):
        for a in range(i_b):
            for b in range(j_b):
                if refernce_hist[a][b] == i:
                    count_b = count_b + 1
        frequency_b.append((i,count_b))
        total_pixel_b = total_pixel_b + count_b
        count_b = 0
    #finding pdf of a
    for intensity, no_of_repeat in frequency_a:
        pdf_a.append(no_of_repeat / total_pixel_a)
    # finding pdf of b
    for intensity, no_of_repeat in frequency_b:
        pdf_b.append(no_of_repeat / total_pixel_b)
    #finding cdf of a
    for value in pdf_a:
        sum_a = sum_a + value
        cdf_a.append(sum_a)
    # finding cdf of b
    for value in pdf_b:
        sum_b = sum_b + value
        cdf_b.append(sum_b)

    cdf_a_X_max_val=max_a*cdf_a
    cdf_b_X_max_val = max_b * cdf_b

    Round_of_a=np.round(cdf_a_X_max_val)
    Round_of_b=np.round(cdf_b_X_max_val)

    print(Round_of_a)


hist_a=cv2.imread('Example1.png',cv2.IMREAD_GRAYSCALE)
hist_b=cv2.imread('waseem.png',cv2.IMREAD_GRAYSCALE)
#myHistMatch(hist_a,hist_b)
#Spatial and geometric filtering

#3.	Open the Example.png image and display it in the notebook. Make sure to correct for the RGB order.
#a.	Flip the image upside down and display it in the notebook.
def flipImage(image):
    flippedimage = cv2.flip(image,0)
    cv2.imshow('vertically Flipped Image', flippedimage)
    cv2.waitKey(-1)
flip_input = cv2.imread("Example1.png")
#flipImage(flip_input)
#b.	Rotate the image 45 and 90 degrees
def rotating_image(imgae):
    rotate_image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("Rotated Image ",rotate_image)
    cv2.waitKey(-2)
#rotating_image(cv2.imread("Example1.png"))
#c.	Resize the image by using scaling factor s = 0.5, 1.5, and
def scaling_image(im):
    scale_precent = 2
    width = int(im.shape[1]+0.5)
    height  = int(im.shape[0]+1.5)
    output=cv2.resize(im,(width,height))
    cv2.imwrite("scaled_image.png",output)
    cv2.imshow("scaled_image",output)
    cv2.waitKey(-2)
scaling_image(cv2.imread('Example1.png',cv2.IMREAD_GRAYSCALE))

#4.	Open the Example1.png image as grayscale and display it in the notebook. Create a

#a.Salt and Paper Noise (save it as "Exam1_SPN.png")

def salt_and_paper_nosise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.3
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0

    cv2.imshow("salt_and_paper_Noise_added",out)
    cv2.waitKey(-1)

    cv2.imwrite("Exam1_SPN.png",out)

Image_for_noise = cv2.imread("Example1.png")
#salt_and_paper_nosise(Image_for_noise)

#b.	Gaussian Noise (save it as "Exam1_Gaus.png")
def Gaussian_Noise_adding(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss

    cv2.imshow("Gaus_Noise_added", noisy)
    cv2.waitKey(-1)

    cv2.imwrite("Exam1_Gaus.png", noisy)

#Gaussian_Noise_adding(Image_for_noise)

#c.	Speckle Noise (save it as "Exam1_Spek.png")
def Speckle_noise_added(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + image * gauss

    cv2.imshow("Speckle_Noise_added", noisy)
    cv2.waitKey(-1)

    cv2.imwrite("Exam1_Spek.png", noisy)

#Speckle_noise_added(Image_for_noise)

#d.poison Noise (save it as "Exam1_Pois.png")
def poison_noise_added(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    cv2.imshow("poison_Noise_added", noisy)
    cv2.waitKey(-1)

    cv2.imwrite("Exam1_Pois.png", noisy)

#poison_noise_added(Image_for_noise)

#e.Plot all of these images together in Matplotlib (use subplot)
def ploting_noisy_images(im1,im2,im3,im4):
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.title("Gaus_Noise")

    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.title("Pois_Noise")

    plt.subplot(2, 2, 3)
    plt.imshow(im3)
    plt.title("Pois_Noise")

    plt.subplot(2, 2, 4)
    plt.imshow(im4)
    plt.title("SPN_Noise")

    plt.show()


im1=cv2.imread("Exam1_Gaus.png",cv2.IMREAD_GRAYSCALE)
im2=cv2.imread("Exam1_Pois.png",cv2.IMREAD_GRAYSCALE)
im3=cv2.imread("Exam1_Spek.png",cv2.IMREAD_GRAYSCALE)
im4=cv2.imread("Exam1_SPN.png",cv2.IMREAD_GRAYSCALE)
#ploting_noisy_images(im1,im2,im3,im4)
#5.	Open the Exam1_SPN.png image as grayscale and display it in the notebook. 7 by 7
#a.	Filter this image using a median filter
def Median_filter(image,wind_size):
    row,col = image.shape
    output_image = np.zeros((row,col))
    padded_image = np.pad(image,(int(np.floor(wind_size/2)),int(np.floor(wind_size/2))),'symmetric')
    for i in range(row):
        for j in range(col):
            patch_curr = padded_image[i:i+wind_size,j:j+wind_size]
            med = np.median(patch_curr)
            output_image[i,j]=med
            print(med)
    cv2.imwrite("median_on_Spn.png",output_image)
    #cv2.imshow("output",output_image)
    plt.imshow(output_image)
    plt.show()

    #cv2.waitKey(-2)

img = cv2.imread("Exam1_SPN.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Median_filter(gray,7)

#b.	Filter this image using an average filter
def Average_filter(image,wind_size):
    row,col = image.shape
    output_image = np.zeros((row,col))
    padded_image = np.pad(image,(int(np.floor(wind_size/2)),int(np.floor(wind_size/2))),'symmetric')
    for i in range(row):
        for j in range(col):
            patch_curr = padded_image[i:i+wind_size,j:j+wind_size]
            sum=np.sum(patch_curr)/wind_size*wind_size
            output_image[i,j]=sum
    #   cv2.imwrite("median_on_Spn.png", output_image)
    # cv2.imshow("output",output_image)
    plt.imshow(output_image)
    plt.show()
#Average_filter(gray,7)
#c.	Open the Exam1_Gaus.png image as grayscale and display it in the notebook.
def open_grayscale_exam1_gaus(img):
    plt.imshow(img)
    plt.show()
i=cv2.imread("Exam1_Gaus.png",cv2.IMREAD_GRAYSCALE)
#open_grayscale_exam1_gaus(i)
# 6.Open the Example1.png image as grayscale and display it in the notebook.
#a.	Display histogram of this image (you may use function that was given during the class)
example1 = cv2.imread("Example1.png")
example1_gray = cv2.cvtColor(example1 , cv2.COLOR_BGR2GRAY)
def display_hist(image):
    plt.hist(image)
    plt.show()
#display_hist(example1_gray)
# b.Apply histogarm equlization and display the histagoram of the image and image itself.
def hist_equ(image):
   im= cv2.equalizeHist(image)
   plt.hist(cv2.hconcat([image,im]))
   plt.show()
#hist_equ(example1_gray)

# c.Apply CLAHE histogarm equlization and display the histagoram of the image and image itself. Just use predefined function for that purpose
def Clahe_hist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    plt.hist(cv2.hconcat([img, cl1]))
    plt.show()
#Clahe_hist(example1_gray)

# d.Open the Example1.png image as grayscale and display it in the notebook.d.	Open the Example1.png image as grayscale and display it in the notebook.
def Display_image(img):
    cv2.show(img)
    cv2.waitKey(-1)
#Display_image(example1_gray)