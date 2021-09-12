#!/usr/bin/env python
# coding: utf-8



## Install Dependencies

get_ipython().system('python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple')
get_ipython().system('python -m pip install python-Levenshtein')
get_ipython().system('python -m pip install shapely')
get_ipython().system('python -m pip install geos')


get_ipython().system('python -m pip install paddleocr')
get_ipython().system('git clone https://github.com/PaddlePaddle/PaddleOCR')


from paddleocr import PaddleOCR, draw_ocr
from shapely import geos
from matplotlib import pyplot as plt
import cv2
import os



ocr_model = PaddleOCR(lang='en',use_gpu=False)

img_path = os.path.join('.', 'drug3.jpg')

result = ocr_model.ocr(img_path)

for res in result:
    print(res[1][0])
boxes = [res[0] for res in result]
text = [res[1][0] for res in result]
scores = [res[1][1] for res in result]

font_path = os.path.join('PaddleOCR','doc','fonts','latin.ttf')

img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure(figsize = (15,15))
annotated = draw_ocr(img,boxes,text,scores,font_path=font_path)
plt.imshow(annotated)


