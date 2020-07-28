from coco_info_v1 import COCO_Info
from visualize import Visualize
import coco_info_v1
# from coco_info_v1 import Visualize
# from coco_info_v1 import Process_dataset
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

user_categorial = ["person"]
# anno_file = "/data_process/json/crowdHuman_annotation_val.json"
anno_file = "/data_process/data_evaluate/anno/instances_val2017.json"
# anno_file = "/data_process/json/voc2007+2012_to_coco_v2_long_tail.json"
# anno_file = "/data_process/json/wider_face_val_annot_coco_style.json"
imgs_dir = "/data_process/data_evaluate/data"
dataset_type = "CocoDataset"
res_dir = "/data_process/data_evaluate/res_dir"
TMP_ANNO_FILE_DIR = '/gddi_output_config/gddi_data'


class COCO_Info2(object):
    def __init__(self):
        pass
    def bbox_area_rato(self):
        """
        得到面积32×32的bbox占总的bbox比例
        :return : x_bbox_area_rato, y_bbox_area_rato
        """
        bbox_num_sum = 0
        area_32_num = 0
        area_more_32_num = 0
        bbox_area = coco_info.get_all_bbox_area()
        for key, val in bbox_area.items():
            bbox_num_sum += len(val)
            area_32 = len(list(filter(lambda x: x<32*32, val)))
            area_more_32 = len(list(filter(lambda x: x>32*32, val))) # 面积大于32
            area_32_num += area_32
            area_more_32_num += area_more_32
        area_32_num_rato = area_32_num / bbox_num_sum
        area_more_32_num_rato = area_more_32_num / bbox_num_sum
        y_bbox_area_rato = [area_32_num_rato, area_more_32_num_rato]
        x_bbox_area_rato = ["area<32", "area>32"]

        return x_bbox_area_rato, y_bbox_area_rato

    def img_bbox_num_rato(self):
        """
        得到每张图片bbox数量的分布
        """
        each_img_bbox_area, each_img_num_bbox, each_img_bbox_dist, each_img_bbox_overlap = coco_info.get_every_image_info()  # 每张数据的信息
        temp_3 = len(list(filter(lambda x: x > 3, each_img_num_bbox.values())))  # one class less than area_threshold
        temp_5 = len(list(filter(lambda x: x > 5, each_img_num_bbox.values())))
        temp_7 = len(list(filter(lambda x: x > 7, each_img_num_bbox.values())))
        temp_9 = len(list(filter(lambda x: x > 9, each_img_num_bbox.values())))
        sum = len(each_img_num_bbox)
        rato_3 = temp_3 / sum
        rato_5 = temp_5 / sum
        rato_7 = temp_7 / sum
        rato_9 = temp_9 / sum
        x_img_bbox_num_rato = ["x>3","x>5","x>7","x>9"]
        y_img_bbox_num_rato = [rato_3, rato_5, rato_7, rato_9]
        print(y_img_bbox_num_rato)
        return x_img_bbox_num_rato, y_img_bbox_num_rato

    def img_bbox_dist_rato(self):
        """
        得到每张图片bbox距离的分布
        """
        dist = {}
        each_img_bbox_area, each_img_num_bbox, each_img_bbox_dist, each_img_bbox_overlap = coco_info.get_every_image_info()  # 每张数据的信息
        for key, val in each_img_bbox_dist.items():
            val.sort()
            if not val == []: # only focus img have bbox>2
                dist[key] = val[0]
        print(dist)
        return dist

    def img_overlap_rato(self):
        """
        """
        overlap_num = 0
        overlap_num_0_2 = 0
        overlap_num_0_1 = 0
        overlap_num_0_3 = 0
        each_img_bbox_area, each_img_num_bbox, each_img_bbox_dist, each_img_bbox_overlap = coco_info.get_every_image_info()  # 每张数据的信息  # 每张数据的信息
        res = []
        for key, val in each_img_bbox_overlap.items():
            temp_2 = [x for x in val if x > 0.2]
            if len(temp_2)>0:
                overlap_num_0_2 += 1
            temp_1 = [x for x in val if x > 0.1]
            if len(temp_1)>0:
                overlap_num_0_1 += 1
            temp_3 = [x for x in val if x > 0.3]
            if len(temp_3) > 0:
                overlap_num_0_3 += 1
        img_sum = coco_info.get_coco_image_sum()

        overlap_num_0_1_rato = overlap_num_0_1 / img_sum
        overlap_num_0_2_rato = overlap_num_0_2 / img_sum
        overlap_num_0_3_rato = overlap_num_0_3 / img_sum
        x_img_overlap_rato = ["x>0.1", "x>0.2", "x>0.3"]
        y_img_overlap_rato = [overlap_num_0_1_rato, overlap_num_0_2_rato, overlap_num_0_3_rato]
        return x_img_overlap_rato, y_img_overlap_rato

    def proportion_each_cat_rato(self):
        """
        获取最多类和最小类占总的比例
        """
        bbox_num, proportion_each_cat = coco_info.get_all_bbox_num()
        max_pro = max(proportion_each_cat.values())
        min_pro = min(proportion_each_cat.values())
        y_proportion_each_cat = [min_pro, max_pro]
        x_proportion_each_cat = ["min_pro", "max_pro"]
        return x_proportion_each_cat, y_proportion_each_cat

    def cat_area_distributed(self, id = 1):
        """
        获取coco id=1 person 类bbox标准差
        """
        bbox_area = coco_info.get_all_bbox_area()
        bbox_area_person = bbox_area[1]
        x_cat_area_std_rato = len(bbox_area_person)
        y_cat_area_std_rato = bbox_area_person
        return x_cat_area_std_rato, y_cat_area_std_rato

    def far_near(self, category="person"):
        bbox_area = coco_info.get_all_bbox_area()
        far_near_cout = 0
        for key, val in bbox_area.items():
            values = sorted(val)
            mid_val = values[len(val) // 2]
            max_val = values[-1]
            min_val = values[0]
            if max_val - min_val > mid_val:
                far_near_cout += 1
        class_sum = coco_info.get_coco_class_sum()
        temp_rato = far_near_cout / class_sum





if __name__ == "__main__":

    coco_info = COCO_Info(user_categorial, dataset_type, anno_file, imgs_dir, res_dir)
    detail_info = COCO_Info2()
    visualize = Visualize()

    # x_bbox_area_rato, y_bbox_area_rato = detail_info.bbox_area_rato()
    # visualize.plt_bbox_area_rato(x_bbox_area_rato, y_bbox_area_rato)

    # x_img_bbox_num_rato, y_img_bbox_num_rato = detail_info.img_bbox_num_rato()
    # visualize.plt_img_bbox_num_rato(x_img_bbox_num_rato, y_img_bbox_num_rato)

    # img_dist = detail_info.img_bbox_dist_rato()
    # visualize.plt_img_bbox_dist_rato(img_dist)

    # x_img_overlap_rato, y_img_overlap_rato = detail_info.img_overlap_rato()
    # visualize.plt_img_overlap_rato(x_img_overlap_rato, y_img_overlap_rato)

    # x_proportion_each_cat, y_proportion_each_cat = detail_info.proportion_each_cat()
    # visualize.plt_proportion_each_cat(x_proportion_each_cat, y_proportion_each_cat)
    x_cat_area_std_rato, y_cat_area_std_rato = detail_info.cat_area_distributed()
    visualize.plt_cat_area_distributed(x_cat_area_std_rato, y_cat_area_std_rato)

    # detail_info.cat_area_var_rato()


