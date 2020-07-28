
import coco_info_v1
# from coco_info_v1 import Visualize
# from coco_info_v1 import Process_dataset
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Visualize(object):

    def __init__(self):
        pass

    def plt_bbox_area_rato(self, x, y, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(x, y)
        plt.title("bbox_area_rato")
        plt.xlabel(u"nums")
        plt.ylabel(u"rato")
        img_path = dst_dir + "bbox_area_rato.png"
        plt.savefig(img_path)
        plt.show()

    def plt_img_bbox_num_rato(self, x, y, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(x, y)
        plt.title("img_bbox_num_rato")
        plt.xlabel(u"img_bbox_num")
        plt.ylabel(u"rato")
        img_path = dst_dir + "img_bbox_num_rato.png"
        plt.savefig(img_path)
        plt.show()

    def plt_img_bbox_dist_rato(self, img_dist, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(list(img_dist.keys()), list(img_dist.values()))
        plt.title("img_bbox_dist_rato")

        # plt.xlim((-5, 5))
        plt.ylim((0, 5))
        plt.xlabel(u"img_id")
        plt.ylabel(u"dist")
        img_path = dst_dir + "img_bbox_dist_rato.png"
        plt.savefig(img_path)
        plt.show()

    def plt_img_overlap_rato(self, x, y, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(x, y)
        plt.title("img_overlap_rato")
        plt.xlabel(u"overlap")
        plt.ylabel(u"rato")
        img_path = dst_dir + "img_overlap_rato.png"
        plt.savefig(img_path)
        plt.show()

    def plt_proportion_each_cat(self, x, y, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(x, y)
        plt.title("proportion_each_cat")
        plt.xlabel(u"proportion_each_cat")
        plt.ylabel(u"rato")
        img_path = dst_dir + "proportion_each_cat.png"
        plt.savefig(img_path)
        plt.show()

    def plt_cat_area_distributed(self, x, y, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(x, y)
        plt.title("cat_area_distributed")
        plt.xlabel(u"bbox")
        plt.ylabel(u"area")
        img_path = dst_dir + "all_cat_area_distributed.png"
        plt.savefig(img_path)
        plt.show()



# if __name__ == "__main__":
#     detailInfo = COCO_Info2()
#
#     detailInfo.plt_bbox_area_rato()