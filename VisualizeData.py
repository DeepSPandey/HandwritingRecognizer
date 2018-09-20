import cv2
import numpy as np


def visualize(arr,noOfData=1):
    dispImg = arr.reshape(16,16,noOfData) *256
    finalImg = dispImg[:,:,0]
    for i in range(1,noOfData):
        finalImg = np.hstack((finalImg,dispImg[:,:,i].reshape(16,16)))
        #finalImg = [finalImg, dispImg[:,:,i]]

    print(finalImg.shape)
    finalImg2 = finalImg[:,0:160]
    for i in range(1,int(np.sqrt(noOfData))):
        finalImg2 = np.vstack((finalImg2,finalImg[:,160*i:160*(i+1)]))
    print(finalImg2.shape)
    cv2.imwrite('disp_img.PNG', finalImg2)
    cv2.imshow("image", finalImg2/255);
    cv2.waitKey(0)
