import numpy as np
import math
import scipy
from scipy import misc
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def sobel(array):
	#print array.shape
        Gdash=array
        #print Gdash.shape
        shape=array.shape
        gx=np.array([[-2,-1,0,1,2] for i in range(0,5)])
        gy=np.array([np.repeat(i,5) for i in range (-2,3)])
        gf=np.array([[np.exp(-(gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j])/3) for j in range(0,5)] for i in range(0,5)])
        smooth=np.array([([np.sum(gf*array[i:i+5,j:j+5])/25 for j in range(0,shape[1]-4)]) for i in range(0,shape[0]-4)])
        for i in range(2,shape[0]-2):
            for j in range(2,shape[1]-2):
                 Gdash[i,j]=smooth[i-2,j-2]
        G=Gdash
        #print G.shape
        sobely=np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
        sobelx=np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
        Gx=np.array([([np.sum(sobelx*array[i:i+3,j:j+3])/9 for j in range(0,shape[1]-2)]) for i in range(0,shape[0]-2)])
        Gy=np.array([([np.sum(sobely*array[i:i+3,j:j+3])/9 for j in range(0,shape[1]-2)]) for i in range(0,shape[0]-2)])                
        Gx=[[Gx[i,j]*Gx[i,j] for j in range(0,shape[1]-2)] for i in range(0,shape[0]-2)]
        Gy=[[Gy[i,j]*Gy[i,j] for j in range(0,shape[1]-2)] for i in range(0,shape[0]-2)]
        tempG=np.sqrt(np.add(Gx,Gy))
        for i in range(1,shape[0]-1):
            for j in range(1,shape[1]-1):
                 G[i,j]=tempG[i-1,j-1]
        G.astype(int)      
        return G

def Area(img):
	naffect=0
	affect=0
	for rownum in range(len(img)):
		for colnum in range(len(img[rownum])):
			if img[rownum,colnum] == 0 :
				naffect=naffect+1
			else:
				affect=affect+1
	a=naffect-affect
	A=affect
	return a,A	


'''def Bcalc(image):
	sx=ndimage.sobel(image,axis=0,mode='constant')
	sy=ndimage.sobel(image,axis=1,mode='constant')
	sob=np.hypot(sx,sy)
	plt.title('border image')
	plt.imshow(sob,cmap=cm.Greys_r)
	plt.show()'''

def ColourCalc(hsvimg):
	mean=0
	SD=0
	for row in range(len(hsvimg)):
		for col in range(len(hsvimg[row])):
			mean=mean+hsvimg[row,col,0]
	for row in range(len(hsvimg)):
		for col in range(len(hsvimg[row])):
			SD=SD+(hsvimg[row,col,0]-mean)*(hsvimg[row,col,0]-mean)
	width,height,val=hsvimg.shape
	n=width*height
	SD=math.sqrt(SD)/(n-1)
	#print('Standard deviation')
	#print(SD)
	return SD
					
if __name__== "__main__":

	img=misc.imread('result.png')
	img2=misc.imread('threshold.png')
	plt.title('Processed image')
	plt.imshow(img)
	plt.show()
	#plt.title('threshold')
	#plt.imshow(img2,cmap=cm.Greys_r)
	#plt.show()
	dilimg=ndimage.binary_dilation(img2)
	for x in range(25):
		dilimg=ndimage.binary_dilation(dilimg)
	erimg=ndimage.binary_erosion(dilimg)
	for x in range(25):
		erimg=ndimage.binary_erosion(erimg)
	plt.title('Closed image')
	plt.imshow(erimg,cmap=cm.Greys_r)
	plt.show()
	a,A=Area(erimg)
	#print(a)
	#print(A)
	Asymmetry=((a/A)*100)/10
	if Asymmetry<0 :
		Asymmetry=Asymmetry*(-1)
	print('Asymmetry')
	print(Asymmetry)

	#BORDER
	#gigi=ndimage.filters.sobel(erimg,axis=-1,output=None,mode='reflect',cval=0.0)
	gigi=sobel(erimg)
	plt.imshow(gigi,cmap=cm.Greys_r)
	plt.show()
	#perimeter
	pcount=0
	for row in range(len(gigi)):
		for col in range(len(gigi[row])):
			if gigi[row,col] == 1:
				pcount=pcount+1
	P=pcount
	Border=((P*P)/(4*3.14*A))/10
	print('Border')
	#print(P)
	print(Border)

	#DIAMETER
	n=(4*A)/P
	Diameter=(math.sqrt(n))/10
	print('Diameter')
	print(Diameter)
	
	#COLOR
	hsvimg=matplotlib.colors.rgb_to_hsv(img)
	#plt.title('HSV image')
	#plt.imshow(hsvimg)
	#plt.show()
	Colour=(ColourCalc(hsvimg))/10	
	print('Colour')
	print(Colour)
	
	#TOTAL DERMOSCOPY RULE
	TDS = (Asymmetry*1.3) + (Border*0.1) + (Colour*0.5) + (Diameter*0.5)
	print('TDS')
	print(TDS)
	if TDS<4.75:
		print('BENIGN MELACOCYTIC LESION')
	if TDS>=4.8 and TDS<=5.45:
		print('SUSPICIOUS LESION')
	if TDS>5.45:
		print('CANCEROUS MOLE')
