import cv2
import numpy as np

#maybe later write your own implementation
def resizeImage(image,height,width,visualize):
    dims = (height,width)
    resizeImage = cv2.resize(image,dims,interpolation = cv2.INTER_AREA)
    cv2.imwrite('Reduced_Image.jpg', resizeImage)
    if visualize:
        cv2.imshow("new_Image", resizeImage)
        cv2.waitKey(0)

    return resizeImage
