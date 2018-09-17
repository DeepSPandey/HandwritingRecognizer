import cv2
import numpy as np

def greyscale(img, visualize):
    img3 = img
    image = np.sum(img, axis = 2,keepdims = True)/3
    image = image.astype(int)
    cv2.imwrite('grayscale_img.jpg', image)

    if visualize == True:
        img3[:,:,:] = image
        cv2.imshow('grayscale_image',img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img2d = np.zeros((len(image[:,0,0]),len(image[0,:,0])))
    img2d[:,:] = image[:,:,0]
    return img2d
