import cv2
import numpy as np

import GreyscaleImage
import BlackWhiteImage
import ResizeImage
import VisualizeData

import Sigmoid
import InitializeParameters
import CostFunction

filename = "semeion.data.txt"
savefile = "tempFile.txt"

filedata = open(filename,"r")

array = (np.loadtxt(filedata)).astype(np.int64)
X = array[:,:-10]
Y = array[:,:10]

#add x0 = 1
X = np.hstack((np.ones((X.shape[0],1)),X))
#print(len(Y[:,1]))
#print(len(X[1,:]))

img = cv2.imread('testImage.jpg')
imgGrey = GreyscaleImage.greyscale(img,False) # (r + b + g) / 3(simple average)
# better imgdir = cv2.imread('testImage.jpg', cv2.IMREAD_GRAYSCALE)
imgBlack = BlackWhiteImage.blackWhite(img,False) # (r + b + g) / 255<1.5 -> 0(simple logic)

imgReduced = ResizeImage.resizeImage(img,20,20,False) # (r + b + g) / 255<1.5 -> 0(simple logic)

#VisualizeData.visualize(X[0:100,:].T,100)

Theta = InitializeParameters.initParams(X.shape,Y.shape)
#print(Theta.shape)
#print(b.shape)


print(X.shape)
z = np.dot(X,Theta)
h = Sigmoid.sigmoid(z)

#no of data
m = X.shape[0]
J = CostFunction.cost(h,Y,m)
print(J)

learning_rate = 0.01
for i in range(100000):
    dTheta =  1/m *np.dot( (h-Y).T,X ).T
    #print(dTheta.shape)
    Theta = Theta - learning_rate * dTheta
    z = np.dot(X,Theta)
    h = Sigmoid.sigmoid(z)
    J = CostFunction.cost(h,Y,m)
    print(J)
