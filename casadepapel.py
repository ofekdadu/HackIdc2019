
# coding: utf-8

# In[6]:


#from api import body_prefix, human_locations,crop_human
import api
import cv2
from matplotlib import pyplot as plt
import timeit
#plt.ion()
# get_ipython().run_line_magic('matplotlib', 'inline')


# # Find human location by using MobilenetSSD

# In[2]:


# img1 = cv2.imread('fofek1.png')[:,:,::-1]
# img2 = cv2.imread('fofek2.png')[:,:,::-1]
# img3 = cv2.imread('fofek3.png')[:,:,::-1]
# img4 = cv2.imread('fofek4.png')[:,:,::-1]
#
# img5 = cv2.imread('fofeklookalike.png')[:,:,::-1]


# ---------1
ofek1 = cv2.imread('ofek1.png')[:,:,::-1]
ofek2 = cv2.imread('ofek2.png')[:,:,::-1]
ofek3 = cv2.imread('ofek3.png')[:,:,::-1]

ofek4 = cv2.imread('ofek4.png')[:,:,::-1]
ofek5 = cv2.imread('ofek5.png')[:,:,::-1]
ofek6 = cv2.imread('ofek6.png')[:,:,::-1]

#  ------------
alon1 = cv2.imread('alon1.png')[:,:,::-1]
alon2 = cv2.imread('alon2.png')[:,:,::-1]
alon3 = cv2.imread('alon3.png')[:,:,::-1]

alon4 = cv2.imread('alon4.png')[:,:,::-1]
alon5 = cv2.imread('alon5.png')[:,:,::-1]
alon6 = cv2.imread('alon6.png')[:,:,::-1]

#  ------------
tamir1 = cv2.imread('tamir1.png')[:,:,::-1]
tamir2 = cv2.imread('tamir2.png')[:,:,::-1]
tamir3 = cv2.imread('tamir3.png')[:,:,::-1]

tamir4 = cv2.imread('tamir4.png')[:,:,::-1]
tamir5 = cv2.imread('tamir5.png')[:,:,::-1]
tamir6 = cv2.imread('tamir6.png')[:,:,::-1]

#  ------------
michael1 = cv2.imread('michael1.png')[:,:,::-1]
michael2 = cv2.imread('michael2.png')[:,:,::-1]
michael3 = cv2.imread('michael3.png')[:,:,::-1]

michael4 = cv2.imread('michael4.png')[:,:,::-1]
michael5 = cv2.imread('michael5.png')[:,:,::-1]
michael6 = cv2.imread('michael6.png')[:,:,::-1]

#  ------------
lili1 = cv2.imread('lili1.png')[:,:,::-1]
lili2 = cv2.imread('lili2.png')[:,:,::-1]
lili3 = cv2.imread('lili3.png')[:,:,::-1]

lili4 = cv2.imread('lili4.png')[:,:,::-1]
lili5 = cv2.imread('lili5.png')[:,:,::-1]
lili6 = cv2.imread('lili6.png')[:,:,::-1]

#  ------------
alice1 = cv2.imread('alice1.png')[:,:,::-1]
alice2 = cv2.imread('alice2.png')[:,:,::-1]
alice3 = cv2.imread('alice3.png')[:,:,::-1]

alice4 = cv2.imread('alice4.png')[:,:,::-1]
alice5 = cv2.imread('alice5.png')[:,:,::-1]
alice6 = cv2.imread('alice6.png')[:,:,::-1]


#  ------------
ido1 = cv2.imread('ido1.png')[:, :, ::-1]
ido2 = cv2.imread('ido2.png')[:, :, ::-1]
ido3 = cv2.imread('ido3.png')[:, :, ::-1]

ido4 = cv2.imread('ido4.png')[:, :, ::-1]
ido5 = cv2.imread('ido5.png')[:, :, ::-1]
ido6 = cv2.imread('ido6.png')[:, :, ::-1]

#-----------

bob1 = cv2.imread('bob1.png')[:, :, ::-1]
bob2 = cv2.imread('bob2.png')[:, :, ::-1]
bob3 = cv2.imread('bob3.png')[:, :, ::-1]

bob4 = cv2.imread('bob4.png')[:, :, ::-1]
bob5 = cv2.imread('bob5.png')[:, :, ::-1]
bob6 = cv2.imread('bob6.png')[:, :, ::-1]

#-----------

alex1 = cv2.imread('alex1.png')[:, :, ::-1]
alex2 = cv2.imread('alex2.png')[:, :, ::-1]
alex3 = cv2.imread('alex3.png')[:, :, ::-1]

alex4 = cv2.imread('alex4.png')[:, :, ::-1]
alex5 = cv2.imread('alex5.png')[:, :, ::-1]
alex6 = cv2.imread('alex6.png')[:, :, ::-1]


def enrichEmbed(img1,img2,img3):
    img1_location = api.human_locations(img1)
    img2_location = api.human_locations(img2)
    img3_location = api.human_locations(img3)

    img_1_human = api.crop_human(img1, img1_location)
    img_2_human = api.crop_human(img2, img2_location)
    img_3_human = api.crop_human(img3, img3_location)


    human_1_2 = img_2_human[0]
    plt.imshow(human_1_2)

    # human 1 photo 2
    human_1_3 = img_3_human[0]
    plt.imshow(human_1_3)


    human_1_1_vector = api.human_vector(human_1_1)
    human_1_2_vector = api.human_vector(human_1_2)
    human_1_3_vector = api.human_vector(human_1_3)

    return (human_1_1_vector +human_1_2_vector +human_1_3_vector )/3



ofekin = enrichEmbed(ofek1,ofek2,ofek3)
ofekout = enrichEmbed(ofek4,ofek5,ofek6)

tamirin = enrichEmbed(tamir1,tamir2,tamir3)
tamirout = enrichEmbed(tamir4,tamir5,tamir6)


alexin = enrichEmbed(alex1,alex2,alex3)
alexout = enrichEmbed(alex4,alex5,alex6)

alicein = enrichEmbed(alice1,alice2,alice3)
aliceout = enrichEmbed(alice4,alice5,alice6)

bobin = enrichEmbed(bob1,bob2,bob3)
bobout = enrichEmbed(bob4,bob5,bob6)

idoin = enrichEmbed(ido1,ido2,ido3)
idoout = enrichEmbed(ido4,ido5,ido6)

liliin = enrichEmbed(lili1,lili2,lili3)
liliout = enrichEmbed(lili4,lili5,lili6)

michaelin = enrichEmbed(michael1,michael2,michael3)
michaelout = enrichEmbed(michael4,michael5,michael6)

alonin = enrichEmbed(alon1,alon2,alon3)
alonout = enrichEmbed(alon4,alon5,alon6)

inList = {ofekin,tamirin,alonin,idoin,liliin,alicein,alexin,bobin,michaelin}
outList = {ofekout,tamirout,alonout,idoout,liliout,aliceout,alexout,bobout,michaelout}

results=[9,9]
i=0
j=0
for personIn in inList:
    for personOut in outList:
        results[i,j] = api.human_distance(personIn,personOut)
        i+=1
    j+=1

print(results)
