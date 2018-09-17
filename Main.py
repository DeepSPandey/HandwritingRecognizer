import cv2
import numpy as np
import GreyscaleImage

filename = "semeion.data.txt"
savefile = "tempFile.txt"

filedata = open(filename,"r")

array = (np.loadtxt(filedata)).astype(np.int64)
X = array[:,:-10]
Y = array[:,:10]

#print(len(Y[:,1]))
#print(len(X[1,:]))

img = cv2.imread('testImage.jpg')
img2 = GreyscaleImage.greyscale(img,False) # (r + b + g) / 3(simple average)
# better imgdir = cv2.imread('testImage.jpg', cv2.IMREAD_GRAYSCALE)

print(img2.shape)
