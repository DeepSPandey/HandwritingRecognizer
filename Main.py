import cv2
import numpy as np

import GreyscaleImage
import BlackWhiteImage
import ResizeImage
import VisualizeData

import Sigmoid
import InitializeParameters
import CostFunction

filename = "semeion.data"
savefile = "tempFile.txt"

filedata = open(filename,"r")

array = (np.loadtxt(filedata)).astype(np.int64)
X = array[:,:-10]
Y = array[:,-10:]

print(Y[0:1500:50,:])

#add x0 = 1
X = np.hstack((np.ones((X.shape[0],1)),X))

#m no of data
m = X.shape[0]
train_size = int(0.6*m)
cv_size = int(0.8*m)

X_train = X[:train_size,:]
X_cv    = X[train_size:cv_size,:]
X_test  = X[cv_size:,:]

Y_train = Y[:train_size,:]
Y_cv    = Y[train_size:cv_size,:]
Y_test  = Y[cv_size:,:]

#print("x" + str(X_train.shape) + " a " + str(X_cv.shape) + " b " + str(X_test.shape))
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


#print(X.shape)
#Train the model
z = np.dot(X_train,Theta)
h = Sigmoid.sigmoid(z)


J = CostFunction.cost(h,Y_train,m)
#print(J)

learning_rate = 0.01
for i in range(1):
    dTheta =  1/m *np.dot( (h-Y_train).T,X_train ).T
    #print(dTheta.shape)
    Theta = Theta - learning_rate * dTheta
    z = np.dot(X_train,Theta)
    h = Sigmoid.sigmoid(z)
    J = CostFunction.cost(h,Y_train,m)
    print(J)


z = np.dot(X_test,Theta)
h = Sigmoid.sigmoid(z)
theCost  = CostFunction.cost(h,Y_test,m)
pred = np.zeros(h.shape)
pred = np.where(h>0.5,1,0)

VisualizeData.visualize(X_test[50,1:],1)
print(Y[0:1500:100,:])
print(pred[100,:])
