
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


img1 = cv2.imread('test/test1.png')[:,:,::-1]
img2 = cv2.imread('test/test2.png')[:,:,::-1]
img3 = cv2.imread('test/test3.png')[:,:,::-1]

img1_location = api.human_locations(img1)
img2_location = api.human_locations(img2)
img3_location = api.human_locations(img3)

img_1_human = api.crop_human(img1, img1_location)
img_2_human = api.crop_human(img2, img2_location)
img_3_human = api.crop_human(img3, img3_location)


# In[3]:


# human 1 photo 1
human_1_1 = img_1_human[0]
plt.imshow(human_1_1)


# In[4]:


# human 1 photo 2
human_1_2 = img_2_human[0]
plt.imshow(human_1_2)


# In[5]:


# human 2 photo 1
human_2_1 = img_3_human[0]
plt.imshow(human_2_1)


# # Make vector of human image

# In[6]:


t1 = timeit.default_timer()
human_1_1_vector = api.human_vector(human_1_1)
human_1_2_vector = api.human_vector(human_1_2)
human_2_1_vector = api.human_vector(human_2_1)
t2 = timeit.default_timer()
print('Time elapsed: {} sec'.format(round(t2 - t1, 3)))


# # Comparing vector

# In[7]:


api.human_distance(human_1_1_vector, human_1_2_vector)


# In[8]:


api.human_distance(human_1_2_vector, human_2_1_vector)


# In[9]:


api.human_distance(human_1_1_vector, human_2_1_vector)

