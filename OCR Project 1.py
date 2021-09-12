#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[14]:


## Install Dependencies


# In[15]:


get_ipython().system('python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple')
get_ipython().system('python -m pip install python-Levenshtein')
get_ipython().system('python -m pip install shapely')
get_ipython().system('python -m pip install geos')


# In[16]:


get_ipython().system('python -m pip install paddleocr')


# In[17]:


get_ipython().system('git clone https://github.com/PaddlePaddle/PaddleOCR')


# In[51]:


from paddleocr import PaddleOCR, draw_ocr
from shapely import geos
from matplotlib import pyplot as plt
import cv2
import os


# In[62]:


ocr_model = PaddleOCR(lang='en',use_gpu=False)


# In[84]:


img_path = os.path.join('.', 'drug3.jpg')


# In[85]:


result = ocr_model.ocr(img_path)


# In[86]:


result


# In[ ]:





# In[87]:


for res in result:
    print(res[1][0])


# In[ ]:





# In[88]:


boxes = [res[0] for res in result]
text = [res[1][0] for res in result]
scores = [res[1][1] for res in result]


# In[89]:


boxes


# In[90]:


font_path = os.path.join('PaddleOCR','doc','fonts','latin.ttf')


# In[91]:


img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[92]:


plt.figure(figsize = (15,15))
annotated = draw_ocr(img,boxes,text,scores,font_path=font_path)
plt.imshow(annotated)


# In[ ]:




