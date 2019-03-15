# imports
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

# taking single image input
img = cv2.imread('rainbow.png')
img = cv2.resize(img, (128,128))
#print(img.shape[0])
b, g, r = cv2.split(img)

# plotting histogram and lines thorough CG
color = ('b','g','r')
cg_list = np.zeros(3)
j = 0

for channel,col in enumerate(color):
    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
    w = 0.0
    num = 0.0
    for i in range(0,256):
        w = w + histr[i][0]
        num = num + (i *  histr[i][0])

    # print(w)
    # print(num)
    cg = num / w
    cg_list[j] = cg
    j += 1
    # print(int(cg))
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.axvline(x=int(cg),color='k',linestyle='--')
    plt.title('Histogram for color scale picture')
    plt.show()
#print(cg_list)


# classifying and storing pixels into bins

# initializing empty bins
bin0 = []
bin1 = []
bin2 = []
bin3 = []
bin4 = []
bin5 = []
bin6 = []
bin7 = []

temp = 0
temp1 = 0
temp2 = 0
temp3 = 0
temp4 = 0
temp5 = 0
temp6 = 0
temp7 = 0


# temp, temp1, temp2, temp3, temp4, temp5, temp6, temp7 = 0
'''
Bin contains only the pixel position not indiviual rgb values of each pixel
'''

# alloting pixels in bins
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        if (r[i][j] <= cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] <= cg_list[0]): # 000
            bin0.append([])
            bin0[temp].append(i)
            bin0[temp].append(j)
            temp += 1                

        elif (r[i][j] <= cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] > cg_list[0]): #001
            bin1.append([])
            bin1[temp1].append(i)
            bin1[temp1].append(j)
            temp1 += 1                
                        
        elif (r[i][j] <= cg_list[2] and g[i][j] > cg_list[1] and b[i][j] <= cg_list[0]): #010
            bin2.append([])
            bin2[temp2].append(i)
            bin2[temp2].append(j)
            temp2 += 1                
            

        elif (r[i][j] <= cg_list[2] and g[i][j] > cg_list[1] and b[i][j] > cg_list[0]): #011
            bin3.append([])
            bin3[temp3].append(i)
            bin3[temp3].append(j)
            temp3 += 1                
            

        elif (r[i][j] > cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] <= cg_list[0]): #100
            bin4.append([])
            bin4[temp4].append(i)
            bin4[temp4].append(j)
            temp4 += 1                
            

        elif (r[i][j] > cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] > cg_list[0]): #101
            bin5.append([])
            bin5[temp5].append(i)
            bin5[temp5].append(j)
            temp5 += 1                
            

        elif (r[i][j] > cg_list[2] and g[i][j] > cg_list[1] and b[i][j] <= cg_list[0]): #110
            bin6.append([])
            bin6[temp6].append(i)
            bin6[temp6].append(j)
            temp6 += 1                
            

        elif (r[i][j] > cg_list[2] and g[i][j] > cg_list[1] and b[i][j] > cg_list[0]): #111
            bin7.append([])
            bin7[temp7].append(i)
            bin7[temp7].append(j)
            temp7 += 1                
            
len_bin0 = len(bin0)
len_bin1 = len(bin1)
len_bin2 = len(bin2)
len_bin3 = len(bin3)
len_bin4 = len(bin4)
len_bin5 = len(bin5)
len_bin6 = len(bin6)
len_bin7 = len(bin7)

'''
def calulations(bin,len_bin):
    mean_r, mean_g, mean_b = 0
    std2_r, std2_g, std2_b = 0
    std3_r, std3_g, std3_b = 0
    std4_r, std4_g, std4_b = 0 
   
    # mean for r
    for i in (0, len_bin):
        for j in (0, len_bin):
           mean_r = mean_r + r[i][j]

    mean_r = mean_r / len_bin

    # mean for g
    for i in (0, len_bin):
        for j in (0, len_bin):
           mean_g = mean_g + g[i][j]

    mean_g = mean_g / len_bin

    # mean for b
    for i in (0, len_bin):
        for j in (0, len_bin):
           mean_b = mean_b + b[i][j]

    mean_b = mean_b / len_bin

    # stds for r
    for i in (0, len_bin):
        for j in (0, len_bin):
            tempp = (r[i][j] - mean_r)**2
            std2_r = std2_r + tempp

    std2_r = std2_r / len_bin
    std2_r = math.sqrt(std2_r)
'''         

print(len(bin0))
print(len(bin1))
print(len(bin2))
print(len(bin3))
print(len(bin4))
print(len(bin5))
print(len(bin6))
print(len(bin7))
