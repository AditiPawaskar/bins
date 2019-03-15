import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

img = cv2.imread('rainbow.png')
img = cv2.resize(img, (128,128))
#print(img.shape[0])
b, g, r = cv2.split(img)

'''
color = ('b','g','r')
for channel,col in enumerate(color):
    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,25resized_image = cv2.resize(image, (100, 50)) 6])
    plt.axvline(x=127,color='k',linestyle='--')
    plt.title('Histogram for color scale picture')
    plt.show()

def threshold(col):
    binary = np.zeros(np.shape(col))
    for i in range(0,col.shape[0]def threshold(col):
    binary = np.zeros(np.shape(col))
    for i in range(0,col.shape[0]):
        for j in range(0,col.shape[1]):
            #print(col[i][j])
            if col[i][j] <=127:
                binary[i][j] = 0
            else:
                binary[i][j] = 1

    return binary

r_binary = threshold(r)
g_binary = threshold(g)
b_binary = threshold(b)

rgb_binary = np.zeros(np.shape(r))

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        rgb_binary[i][j] = r_binary[i][j]*100 + g_binary[i][j]*10 + b_binary[i][j]

#print(rgb_binary)

count = {}

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        y = rgb_binary[i][j]
        if y not in count:0
            count[y] = 0;
        count[y] +=1 ;):
        for j in range(0,col.shape[1]):
            #print(col[i][j])
            if col[i][j] <=127:
                binary[i][j] = 0
            else:
                binary[i][j] = 1

    return binary

r_binary = threshold(r)
g_binary = threshold(g)
b_binary = threshold(b)

rgb_binary = np.zeros(np.shape(r))

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        rgb_binary[i][j] = r_binary[i][j]*100 + g_binary[i][j]*10 + b_binary[i][j]

#print(rgb_binary)

count = {}

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        y = rgb_binary[i][j]
        if y not in count:
            count[y] = 0;
        count[y] +=1 ;

#print (count);
'''

def calc(r_val,b_val,g_val,r,g,b):

    mean,std,count = ([] for i in range(3))

    r.append(r_val)
    g.append(g_val)
    b.append(b_val)

    count = [len(r),len(g),len(b)]

    mean_r = sum(r)/len(r)
    mean_g = sum(g)/len(g)
    mean_b = sum(b)/len(b)
    mean = [mean_r,mean_g,mean_b]

    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)
    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)
    std = [std_r,std_g,std_b]

    #print(count)
    #print(mean)
    #print(std)

    return count, mean, std

count0,count1,count2,count3,count4,count5,count6,count7 = ([] for i in range(8))
mean0,mecountan1,mean2,mean3,mean4,mean5,mean6,mean7 = ([] for i in range(8))
std0,std1,std2,std3,std4,std5,std6,std7 = ([] for i in range(8))
r0,r1,r2,r3,r4,r5,r6,r7 = ([] for i in range(8))
g0,g1,g2,g3,g4,g5,g6,g7 = ([] for i in range(8))
b0,b1,b2,b3,b4,b5,b6,b7 = ([] for i in range(8))

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        if (r[i][j] <= 127 and g[i][j] <= 127 and b[i][j] <= 127):
            count0, mean0, std0 = calc(r[i][j],g[i][j],b[i][j],r0,g0,b0)

        elif (r[i][j] <= 127 and g[i][j] <= 127 and b[i][j] > 127):
            count1, mean1, std1 = calc(r[i][j],g[i][j],b[i][j],r1,g1,b1)

        elif (r[i][j] <= 127 and g[i][j] > 127 and b[i][j] <= 127):
            count2, mean2, std2 = calc(r[i][j],g[i][j],b[i][j],r2,g2,b2)

        elif (r[i][j] <= 127 and g[i][j] > 127 and b[i][j] > 127):
            count3, mean3, std3 = calc(r[i][j],g[i][j],b[i][j],r3,g3,b3)

        elif (r[i][j] > 127 and g[i][j] <= 127 and b[i][j] <= 127):
            count4, mean4, std4 = calc(r[i][j],g[i][j],b[i][j],r4,g4,b4)

        elif (r[i][j] > 127 and g[i][j] <= 127 and b[i][j] > 127):
            count5, mean5, std5 = calc(r[i][j],g[i][j],b[i][j],r5,g5,b5)

        elif (r[i][j] > 127 and g[i][j] > 127 and b[i][j] <= 127):
            count6, mean6, std6 = calc(r[i][j],g[i][j],b[i][j],r6,g6,b6)

        elif (r[i][j] > 127 and g[i][j] > 127 and b[i][j] > 127):
            count7, mean7, std7 = calc(r[i][j],g[i][j],b[i][j],r7,g7,b7)

print(count0,mean0,std0)
print(count1,mean1,std1)
print(count2,mean2,std2)
print(count3,mean3,std3)
print(count4,mean4,std4)
print(count5,mean5,std5)
print(count6,mean6,std6)
print(count7,mean7,std7)

print(" ")
print("counts:")
print(count0, count1, count2, count3, count4, count5, count6, count7)
print("means: ")
print(mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7)
print("Standard deviations: ")
print(std0, std1, std2, std3, std4, std5, std6, std7)

print('done')
cv2.imshow('resize',img)
cv2.waitKey()
cv2.destroyAllWindows()
