# -*- coding:utf-8 -*-

from __future__ import print_function
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json

json_file='/coco_dataset/annotations/instances_val2017.json' # # Object Instance 类型的标注

def resize_bbox(json_file, h_resize = 100, w_resize = 200):

    coco = COCO(json_file)
    images = coco.dataset["images"]
    annotation = []

    for i in range(len(images)):
        h_rato = h_resize / images[i]["height"]
        w_rato = w_resize / images[i]["width"]
        annIds = coco.getAnnIds(imgIds=images[i]["id"], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            ann["bbox"][0] = ann["bbox"][0]*w_rato
            ann["bbox"][1] = ann["bbox"][1]*h_rato
            ann["bbox"][2] = ann["bbox"][2]*w_rato
            ann["bbox"][3] = ann["bbox"][3]*h_rato
        annotation.append(anns)
    return annotation


    coco = COCO(json_file)
    catIds = coco.getCatIds(coco.getCatIds())
    imgIds = coco.getImgIds(catIds=catIds)
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # get bbox basic info
        # ex: w, h, c_x, c_y
        for ann in anns:
            w = ann["bbox"][2]
            h = ann["bbox"][3]

def save_resized_to_json(json_file):
    data=json.load(open(json_file,'r'))

    data_2={}
    data_2['info']=data['info']
    data_2['licenses']=data['licenses']
    data_2['images']=data['images'] # 只提取第一张图片
    data_2['categories']=data['categories']
    data_2['annotations']=resize_bbox(h_resize = 100, w_resize = 200)
    json.dump(data_2,open('/temp_res/new1_instances_val2017.json','w'),indent=4) # indent=4 更加美观显示

save_resized_to_json()
