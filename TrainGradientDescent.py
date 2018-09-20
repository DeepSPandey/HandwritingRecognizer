import numpy as np
import Sigmoid
import CostFunction
import PlotData

def train(Theta, X_train, Y_train,learning_rate,noOfIter):
    train_size = X_train.shape[0]

    z = np.dot(X_train,Theta)
    h = Sigmoid.sigmoid(z)

    J = CostFunction.cost(h,Y_train,train_size)
    Jarr=[]
    Jarr = np.hstack((Jarr,J))

    for i in range(noOfIter):
        dTheta =  1/train_size *np.dot( (h-Y_train).T,X_train ).T
        #print(dTheta.shape)
        Theta = Theta - learning_rate * dTheta
        z = np.dot(X_train,Theta)
        h = Sigmoid.sigmoid(z)
        J = CostFunction.cost(h,Y_train,train_size)
        Jarr = np.hstack((Jarr,J))
        print(J)

    plotGraph = False
    if plotGraph:
        PlotData.plotGraph(Jarr,noOfIter)



    return Theta,J
