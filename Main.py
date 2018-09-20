import cv2
import numpy as np

import GreyscaleImage
import BlackWhiteImage
import ResizeImage
import VisualizeData

import Sigmoid
import InitializeParameters
import CostFunction
import TrainGradientDescent
import PlotData


filename = "semeion.data"
savefile = "saveParams.txt"

filedata = open(filename,"r")

array = (np.loadtxt(filedata)).astype(np.int64)
X = array[:,:-10]
Y = array[:,-10:]

#print(Y[0:1500:50,:])

#add x0 = 1
X = np.hstack((np.ones((X.shape[0],1)),X))

#m no of data
m = X.shape[0]
train_size = int(0.8*m)
cv_size = int(0.9*m)

X_train = X[:train_size,:]
X_cv    = X[train_size:cv_size,:]
X_test  = X[cv_size:,:]

Y_train = Y[:train_size,:]
Y_cv    = Y[train_size:cv_size,:]
Y_test  = Y[cv_size:,:]

#print("x" + str(X_train.shape) + " a " + str(X_cv.shape) + " b " + str(X_test.shape))
#print(len(Y[:,1]))
#print(len(X[1,:]))

img = cv2.imread('testThis.png')
imgGrey = GreyscaleImage.greyscale(img,False) # (r + b + g) / 3(simple average)
# better imgdir = cv2.imread('testImage.jpg', cv2.IMREAD_GRAYSCALE)
imgBlack = BlackWhiteImage.blackWhite(img,False) # (r + b + g) / 255<1.5 -> 0(simple logic)
print(imgBlack.shape)
imgReduced = ResizeImage.resizeImage(imgBlack,16,16,False)
print(imgReduced)
print(imgReduced.shape)

#VisualizeData.visualize(X[0:100,:].T,100)

Theta = InitializeParameters.initParams(X.shape,Y.shape)
#print(Theta.shape)

learning_rate = 0.01
noOfIter = 10000
doITrain = False
if doITrain:
    Theta, J = TrainGradientDescent.train(Theta, X_train, Y_train,learning_rate, noOfIter)
    np.savetxt('saveParams.txt',Theta,delimiter = ",")
    print(Theta.dtype)
else:
    a = open('saveParams.txt', 'r')
    Theta = (np.loadtxt(a,delimiter = ',')).astype(np.float64)
#Train the model

#Test The model

X_my = np.reshape(imgReduced,(1,256))
#print(X_my)
X_my = np.hstack((np.ones((1,1)),X_my))
z = np.dot(X_my,Theta)
h = Sigmoid.sigmoid(z)
pred = np.where(h>0.5,1,0)


dNo = 189
VisualizeData.visualize(X[dNo,1:],1)
print(Y[dNo,:])
print(pred)
