import numpy as np

def getHistGray(image):
    assert len(image.shape) == 2, "Must be grayscale image"
    hist = np.zeros(255)
    for row in image:
        for col in row:
            hist[int(col)] += 1
    return hist
    
if __name__ == "__main__":
    import oritogray as gs
    from scipy import misc
    image = misc.imread('503496.jpg')
    grey = gs.convertToGreyScale(image)
    hist = getHistGray(grey)
    print(hist)
