import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# READING AND DISPLAYING AN IMAGE
# img = cv.imread("C:/Users/mnimi/Downloads/cat.jpg")
# img = cv.imread("C:/Users/mnimi/Downloads/cat.jpg", cv.IMREAD_COLOR)
img = cv.imread("C:/Users/mnimi/Downloads/cat.jpg", cv.IMREAD_GRAYSCALE)
# img = cv.imread("C:/Users/mnimi/Downloads/cat.jpg", cv.IMREAD_UNCHANGED)
# cv.imshow("Display Image", img)

# CHANGING COLOR SPACES
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow("Grayscale Image", gray)
# cv.imshow("HSV Image", hsv)

# tracking hsv values
green = np.uint8([[[0, 255, 0]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
# print(hsv_green)
blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
# print(hsv_blue)

# GEOMETRIC TRANSFORMATIONS

# SCALING
# res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# res = cv.resize(img, None, fx=3, fy=3, interpolation=cv.INTER_AREA)
# res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
height, width = img.shape[:2]
res = cv.resize(img, (3*width, 3*height), interpolation=cv.INTER_CUBIC)
# cv.imshow("Scaled Image", res)

# TRANSLATION
row, col = img.shape
# a shift of (50,50)
M = np.float32([[1, 0, 50], [0, 1, 50]])
trsl = cv.warpAffine(img, M, (col, row))
# cv.imshow('Translated Image', trsl)

# ROTATION
# rotates the image by 90 degree with respect to center
M = cv.getRotationMatrix2D(((col-1)/2.0, (row-1)/2.0), 90, 1)
rot = cv.warpAffine(img, M, (col, row))
# cv.imshow('Rotated Image', rot)

# AFFINE TRANSFORMATION
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv.getAffineTransform(pts1, pts2)
aft = cv.warpAffine(img, M, (col, row))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(aft), plt.title('Output')
# plt.show()

# PERSPECTIVE TRANSFORMATION
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)
pspt = cv.warpPerspective(img, M, (300, 300))
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(pspt), plt.title('Output')
# plt.show()

# THRESHOLDING

# SIMPLE THRESHOLDING
ret, t1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, t2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, t3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, t4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, t5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
titles = ['Original', 'Binary', 'Binary Inverse',
          'Truncated', 'Zero', 'Zero Inverse']
images = [img, t1, t2, t3, t4, t5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

# ADAPTIVE THRESHOLDING
img = cv.medianBlur(img, 5)
ret, t1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
t2 = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
t3 = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
titles = ['Original', 'Global', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, t1, t2, t3]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
# plt.show()

# OTSU'S THRESHOLDING
# global thresholding
ret1, t1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# otsu's thresholding
ret2, t2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# otssu's thresholding after gaussian filtering
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, t3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot images and their histograms
images = [img, 0, t1, img, 0, t1, img, 0, t3]
titles = ['Original Noisy', 'Histogram', 'Global', 'Original Noisy',
          'Histogram', "Otsu's", 'Gaussian filtered', 'Histogram', "Otsu's"]
for i in range(3):
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()

# FILTERING

# 2D convolution
kernel = np.ones((5, 5), np.float32)/25
flt = cv.filter2D(img, -1, kernel)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(flt), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
# plt.show()

# IMAGE BLURRING/SMOOTHING

# AVERAGING/MEAN BLUR
# blur = cv.blur(img, (5, 5))

# GAUSSIAN BLUR
# blur = cv.GaussianBlur(img, (5, 5), 0)

# MEDIAN BLUR
# blur=cv.medianBlur(img, 5)

# BILATERAL FILTERING
blur = cv.bilateralFilter(img, 9, 75, 75)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
# plt.show()

# MORPHOLOGICAL TRANSFORMATIONS

# EROSION
img1 = cv.imread("C:/Users/mnimi/Downloads/calligraphy.jpg",
                 cv.IMREAD_GRAYSCALE)
# cv.imshow("Original", img1)
kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(img1, kernel, iterations=1)
# cv.imshow("Erosion", erosion)

# DILATION
dilation = cv.dilate(img1, kernel, iterations=1)
# cv.imshow("Dilation", dilation)

# OPENING
opening = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
# cv.imshow("Opening", opening)

# CLOSING
closing = cv.morphologyEx(img1, cv.MORPH_CLOSE, kernel)
# cv.imshow("Closing", closing)

# MORPHOLOGICAL GRADIENT
gradient = cv.morphologyEx(img1, cv.MORPH_GRADIENT, kernel)
# cv.imshow("Gradient", gradient)

# STRUCTURING ELEMENT
# rectangular kernel
rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# print(rect)
# elliptical kernel
ell = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# print(ell)
# cross-shaped kernel
cs = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
# print(cs)

# IMAGE GRADIENTS
# sobel
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# laplacian
l = cv.Laplacian(img, cv.CV_64F)
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(l, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.show()

# CANNY EDGE DETECTION
edges = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
# plt.show()

# HISTOGRAMS

# histogram calculation
hist = cv.calcHist([img], [0], None, [256], [0, 256])
# using numpy
hist, bins = np.histogram(img.ravel(), 256, [0, 256])
# plotting histograms
# using matplotlib
plt.hist(img.ravel(), 256, [0, 256])
# plt.show()
# using opencv
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
maskedimg = cv.bitwise_and(img, img, mask=mask)
# calculate histogram with and without mask
histfull = cv.calcHist([img], [0], None, [256], [0, 256])
histmask = cv.calcHist([img], [0], mask, [256], [0, 256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(maskedimg, 'gray')
plt.subplot(224), plt.plot(histfull), plt.plot(histmask)
plt.xlim([0, 256])
# plt.show()

# HISTOGRAM EQUALIZATION
eq = cv.equalizeHist(img)
res = np.hstack((img, eq))
# cv.imshow('res.png', res)

# FOURIER TRANSFORM

# DFT
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# inverse DFT
rows, cols = img.shape
crow, ccol = rows/2, cols/2
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()
