import cv2
import numpy as np

def blackWhite(img, visualize):
    img3 = img
    image = ( np.sum(img, axis = 2,keepdims = True)>(255+125) )
    #image = image.astype(int)
    cv2.imwrite('blackWhite_img.jpg', image*255)

    if visualize == True:
        img3[:,:,:] = image
        cv2.imshow('blackWhite_image',img3*255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img2d = np.zeros((len(image[:,0,0]),len(image[0,:,0])))
    img2d[:,:] = image[:,:,0]
    return img2d
