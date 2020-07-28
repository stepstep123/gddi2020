from coco_info_v1 import COCO_Info
from coco_info_v1 import Visualize
from coco_info_v1 import Process_dataset
import numpy as np
import coco_info_v1
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
user_categorial = ["person"]
anno_file = "/data_process/json/voc2007+2012_to_coco_v2_long_tail.json"
# anno_file = "/data_process/data_evaluate/anno/instances_val2017.json"
# anno_file = "/data_process/json/voc_to_coco_fewshot.json"
# anno_file = "/data_process/json/crowdHuman_annotation_val.json"
anno_file = "/data_process/json/wider_face_val_annot_coco_style.json"
imgs_dir = "/data_process/data_evaluate/data"
dataset_type = "CocoDataset"
res_dir = "/data_process/data_evaluate/res_dir"


class Criteria(object):


    def small_object_detection(self, area_threshold=32*32, rato=0.7):

        """
        get a data is small object or not

            :paras: area_threshold : if some class area > area_threshold, it is small object
                    rato ： small_num / whole_num
            :return: True or False
        """
        # data_inspect = {}
        bbox_area = coco_info.get_all_bbox_area()
        bbox_num_sum = 0 # init the whole dataset bbox num
        small_num_sum = 0 # init the wholw dataset small bbox num
        for key, val in bbox_area.items():
            bbox_num_sum += len(val)
            small_area = list(filter(lambda x:x<area_threshold, val)) # one class less than area_threshold
            small_area_num = len(small_area)
            small_num_sum += small_area_num
        temp_rato = small_num_sum / bbox_num_sum
        print("small_num_sum:", small_num_sum)
        print("bbox_num_sum:", bbox_num_sum)
        print("small_num_sum / bbox_num_sum :", temp_rato)
        print(temp_rato)
        if temp_rato > rato:
            return True
        else:
            return False
    def fewshot(self, each_class_num = 50):
        """
        fewshot detection
        """
        class_num_dict = coco_info.get_coco_class_num_dic()
        if max(class_num_dict.values()) < each_class_num:
            return True #  few shot dataset
        else:
            return False # not few shot dataset

    def intensive(self, num_bbox_threshold=5, min_bbox_dist=100, rato=0.5):
        """
        intensive detection
        :args : num_bbox_threshold : every img bbox nums threshold
              : min_bbox_dis : each bbox distace squre
        """
        intensive_num = 0
        each_img_bbox_area, each_img_num_bbox, each_img_bbox_dist, each_img_bbox_overlap = coco_info.get_every_image_info()  # 每张数据的信息

        for key, val in each_img_bbox_dist.items():
            val.sort()
            num_bbox = each_img_num_bbox[key]
            if num_bbox > num_bbox_threshold: # every img bbox num
                if val[0] < min_bbox_dist:
                    intensive_num += 1
        imgs_sum = coco_info.get_coco_image_sum()
        temp_rato = intensive_num / imgs_sum
        if temp_rato > rato:
            return True # intensive
        else:
            return False # not intensive
    def occlusion(self, overlap_threshold=0.2, rato=0.5):
        """
        occlusion detection
        :args : overlap_threshold : every img overlap threshold
              : min_bbox_dis : occlusion img /  img sum
        """
        overlap_num = 0
        each_img_bbox_area, each_img_num_bbox, each_img_bbox_dist, each_img_bbox_overlap = coco_info.get_every_image_info()  # 每张数据的信息  # 每张数据的信息
        res = []
        for key, val in each_img_bbox_overlap.items():
            temp = [x for x in val if x > overlap_threshold]
            if len(temp)>0:
                overlap_num += 1

        img_sum = coco_info.get_coco_image_sum()
        if overlap_num / img_sum > rato:
            return True # occlusion
        else:
            return False # not occlusion
    def long_tail(self, big_divide_small=20):
        """
        long tail detection
        """
        bbox_num, proportion_each_cat = coco_info.get_all_bbox_num()
        max_pro = max(proportion_each_cat.values())
        min_pro = min(proportion_each_cat.values())
        if (max_pro / min_pro) > big_divide_small:
            return True # long tail dataset
        else:
            return False # not long tail dataset
    def far_near(self, rato=0.5):
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
        if temp_rato > rato:
            return True # far_near dataset
        else:
            return False # not far_near dataset
    def brightness(self, bright_thres=0.5, dark_thres = 0.4, imgs_dir=imgs_dir):
        underexposed, overexposed, normal = 0, 0, 0
        imagelist = os.listdir(imgs_dir)
        for img_path in imagelist:
            image_path = os.path.join(imgs_dir, img_path)
            frame = cv2.imread(image_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark_part = cv2.inRange(gray, 0, 30)
            bright_part = cv2.inRange(gray, 220, 255)
            total_pixel = np.size(gray)
            dark_pixel = np.sum(dark_part > 0)
            bright_pixel = np.sum(bright_part > 0)
            if (dark_pixel / total_pixel) > bright_thres:
                # print("Face is underexposed!")
                underexposed += 1
            if (bright_pixel / total_pixel) > dark_thres:
                # print("Face is overexposed!")
                overexposed += 1
            else:
                # print("Face is normal!")
                normal += 1
        status = max(normal, underexposed, overexposed)
        if status == normal:
            return True # normal image
        else:
            return False # unnormal image
    def get_result(self):
        res = {}
        small_detect = self.small_object_detection()
        few_shot = self.fewshot()
        intensive = self.intensive()
        occlusion = self.occlusion()
        brightness = self.brightness()
        # far_near = self.far_near()
        brightness = self.brightness()
        res["small_detect"] = small_detect
        res["few_shot"] = few_shot
        res["intensive"] = intensive
        res["occlusion"] = occlusion
        res["brightness"] = brightness
        # res["far_near"] = far_near
        res["brightness"] = brightness

        print(res)
        return res


if __name__ == "__main__":

    coco_info = COCO_Info(user_categorial, dataset_type, anno_file, imgs_dir, res_dir)
    visualize = Visualize()
    # coco_info.check()
    criteria = Criteria()
    res = criteria.small_object_detection()
    # res = criteria.small_object_detection()
    # criteria.show_intensive_dist()
    # visualize.show_intensive(x, y)
    # criteria.plot_intensive(x, y)
    # print(res)

