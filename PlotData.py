import numpy as np
import matplotlib.pyplot as plt

def plotGraph(J,iter):
    #t = np.arange(0.,5.,0.2)
    #plt.plot(t,t,'r--',t,t**2,'bs',t,t**4,'g^')

    #plt.plot([1,2,3,4],[1,4,9,16],'go')
    it = np.arange(0,iter+1,1)
    plt.plot(J,it,'b-')

    plt.title("Gradient Descent (J plot)")
    plt.ylabel("Cost")
    plt.xlabel("No.of iteration")
    plt.show()

    return
