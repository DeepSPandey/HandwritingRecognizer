import cv2
import numpy as np

import GreyscaleImage
import BlackWhiteImage
import ResizeImage
import VisualizeData

filename = "semeion.data.txt"
savefile = "tempFile.txt"

filedata = open(filename,"r")

array = (np.loadtxt(filedata)).astype(np.int64)
X = array[:,:-10]
Y = array[:,:10]

#print(len(Y[:,1]))
#print(len(X[1,:]))

img = cv2.imread('testImage.jpg')
imgGrey = GreyscaleImage.greyscale(img,False) # (r + b + g) / 3(simple average)
# better imgdir = cv2.imread('testImage.jpg', cv2.IMREAD_GRAYSCALE)
imgBlack = BlackWhiteImage.blackWhite(img,False) # (r + b + g) / 255<1.5 -> 0(simple logic)

imgReduced = ResizeImage.resizeImage(img,20,20,False) # (r + b + g) / 255<1.5 -> 0(simple logic)

VisualizeData.visualize(X[0:100,:].T,100)
