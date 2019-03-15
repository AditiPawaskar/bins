import cv2
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

img = cv2.imread('/home/mayank/aditi/mask5.png')
img = cv2.resize(img, (128,128))
#print(img.shape[0])
b, g, r = cv2.split(img)


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

    print(w)
    print(num)
    cg = num / w
    cg_list[j] = cg
    j += 1
    print(int(cg))
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.axvline(x=int(cg),color='k',linestyle='--')
    plt.title('Histogram for color scale picture')
    plt.show()

print(cg_list)
'''
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
'''
def  generate_list(r_val,b_val,g_val,r,g,b):

    mean,std,count = ([] for i in range(3))

    r.append(r_val)
    g.append(g_val)
    b.append(b_val)
'''

'''
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
'''
    #print(count)
    #print(mean)
    #print(std)
'''
    return count, mean, std
'''

count0,count1,count2,count3,count4,count5,count6,count7 = ([] for i in range(8))
mean0,mean1,mean2,mean3,mean4,mean5,mean6,mean7 = ([] for i in range(8))
std0,std1,std2,std3,std4,std5,std6,std7 = ([] for i in range(8))
r0,r1,r2,r3,r4,r5,r6,r7 = ([] for i in range(8))
g0,g1,g2,g3,g4,g5,g6,g7 = ([] for i in range(8))
b0,b1,b2,b3,b4,b5,b6,b7 = ([] for i in range(8))

#bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7 = ([] for i in range(3))
bin0 = [[] for i in range(3)]
bin1 = [[] for i in range(3)]
bin2 = [[] for i in range(3)]
bin3 = [[] for i in range(3)]
bin4 = [[] for i in range(3)]
bin5 = [[] for i in range(3)]
bin6 = [[] for i in range(3)]
bin7 = [[] for i in range(3)]



for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        print("original rgb value:")
        print(r[0][0],g[0][0],b[0][0])

        if (r[i][j] <= cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] <= cg_list[0]): # 000
#            count0, mean0, std0 = calc(r[i][j],g[i][j],b[i][j],r0,g0,b0) # why passed r,g,b
            # third_std =  calc_std()
        # fourth_std =
            bin0[0].append(r[i][j])
            bin0[1].append(g[i][j])
            bin0[2].append(b[i][j])


        elif (r[i][j] <= cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] > cg_list[0]): #001
            #count1, mean1, std1 = calc(r[i][j],g[i][j],b[i][j],r1,g1,b1)
            bin1[0].append(r[i][j])
            bin1[1].append(g[i][j])
            bin1[2].append(b[i][j])


        elif (r[i][j] <= cg_list[2] and g[i][j] > cg_list[1] and b[i][j] <= cg_list[0]): #010
            #count2, mean2, std2 = calc(r[i][j],g[i][j],b[i][j],r2,g2,b2)
            bin2[0].append(r[i][j])
            bin2[1].append(g[i][j])
            bin2[2].append(b[i][j])


        elif (r[i][j] <= cg_list[2] and g[i][j] > cg_list[1] and b[i][j] > cg_list[0]): #011
            #count3, mean3, std3 = calc(r[i][j],g[i][j],b[i][j],r3,g3,b3)
            bin3[0].append(r[i][j])
            bin3[1].append(g[i][j])
            bin3[2].append(b[i][j])


        elif (r[i][j] > cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] <= cg_list[0]): #100
            #count4, mean4, std4 = calc(r[i][j],g[i][j],b[i][j],r4,g4,b4)
            bin4[0].append(r[i][j])
            bin4[1].append(g[i][j])
            bin4[2].append(b[i][j])


        elif (r[i][j] > cg_list[2] and g[i][j] <= cg_list[1] and b[i][j] > cg_list[0]): #101
            #count5, mean5, std5 = calc(r[i][j],g[i][j],b[i][j],r5,g5,b5)
            bin5[0].append(r[i][j])
            bin5[1].append(g[i][j])
            bin5[2].append(b[i][j])


        elif (r[i][j] > cg_list[2] and g[i][j] > cg_list[1] and b[i][j] <= cg_list[0]): #110
            #count6, mean6, std6 = calc(r[i][j],g[i][j],b[i][j],r6,g6,b6)
            bin6[0].append(r[i][j])
            bin6[1].append(g[i][j])
            bin6[2].append(b[i][j])


        elif (r[i][j] > cg_list[2] and g[i][j] > cg_list[1] and b[i][j] > cg_list[0]): #111
            #count7, mean7, std7 = calc(r[i][j],g[i][j],b[i][j],r7,g7,b7)
            bin7[0].append(r[i][j])
            bin7[1].append(g[i][j])
            bin7[2].append(b[i][j])


print(bin0[0][0][0])


'''
print(count0,mean0,std0)
print(count1,mean1,std1)
print(count2,mean2,std2)
print(count3,mean3,std3)
print(count4,mean4,std4)
print(count5,mean5,std5)
print(count6,mean6,std6)
print(count7,mean7,std7)
'''


print('done')
cv2.imshow('resize',img)
cv2.waitKey()
cv2.destroyAllWindows()
