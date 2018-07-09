from scipy import misc
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def convertToGreyScale(image):
    
    def getWeightedAvg(pixel):
        return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]        
        
    grey = np.zeros(image.shape[0:-1])
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = getWeightedAvg(image[rownum][colnum])
        
    return grey

if __name__ == "__main__":
    image = misc.imread('503496.jpg')
    
    grey = convertToGreyScale(image)
    
    plt.title("Original")
    plt.imshow(image)
    plt.show()
   
    plt.title("Weighted Average")
    plt.imshow(grey, cmap = cm.Greys_r)
    plt.show()
