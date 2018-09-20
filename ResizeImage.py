import cv2
import numpy as np

#maybe later write your own implementation
def resizeImage(image,height,width,visualize):
    dims = (height,width)
    resizeImage = cv2.resize(image,dims,interpolation = cv2.INTER_AREA)
    #FLIP AS THE TRAINING DATA IS OPPOSITE
    resizeImage = np.where(resizeImage>0.5,0,1)
    cv2.imwrite('Reduced_Image.jpg', resizeImage*256)
    if visualize:
        cv2.imshow("new_Image", resizeImage)
        cv2.waitKey(0)

    return resizeImage
