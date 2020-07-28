import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from criteria_detection_v1 import Criteria
from coco_info_v1 import COCO_Info
import numpy as np
import matplotlib.pyplot as plt

user_categorial = ["person"]
anno_file = "/data_process/data_evaluate/anno/instances_val2017.json"
imgs_dir = "/data_process/data_evaluate/data"
dataset_type = "CocoDataset"
res_dir = "/data_process/data_evaluate/res_dir"

class Visualize(object):
    def __init__(self):
        pass

    def show_bbox_num(self):
        """
        return : nums(dic) : {"catId":bbox_num}
                 propertion(dic) : {"catId":propertion}
        """
        bbox_num, propertion = cocoInfo.get_all_bbox_num()
        # nums = {}
        # propertion  = {}
        # for key, vals in bbox_num.items():
        #     nums[key] = int(vals[0])
        #     propertion[key] = vals[1]
        return bbox_num, propertion

    def show_cat_area(self):
        res = {} # {1: area}
        bbox_area = cocoInfo.get_all_bbox_area()
        bbox_num = cocoInfo.get_all_bbox_num()
        for key, val in bbox_area.items():
            mean_area = np.mean(val)
            res[key] = int(mean_area)
        # print(res)
        return res

    def to_csv(self):
        """
        abstract infor from json
        and save to a csv
        """
        nums, prop = self.show_bbox_num()
        areas = self.show_cat_area()
        temp = {"catId":list(nums.keys()), "nums":list(nums.values()), "prop":list(prop.values()), "areas":list(areas.values())}
        dataframe = pd.DataFrame(temp)
        dataframe.to_csv("test.csv", index=False, sep=',')


    def plot_cat_info(self, info_csv_path="test.csv"):
        info = pd.read_csv(info_csv_path)
        # print(info["catId"])
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(info["catId"], info.nums)
        plt.xlabel(u"class")
        plt.ylabel(u"num")
        # plt.subplot2grid((2, 3), (0, 0))
        # info.nums.value_counts().plot(kind="bar") # bar

        fig2 = plt.figure()
        plt.bar(info["catId"], info.areas)
        plt.show()
    def plot_bbox_num(self):
        nums, prop =self.show_bbox_num()
        x1 = list(nums.keys())
        y1 = list(nums.values())

        x2 = list(prop.keys())
        y2 = list(prop.values())

        # plt.bar(x1, y1)

        plt.bar(x2, y2, fc='r')
        plt.show()
    def plot_bbox_prop(self):
        pass

cocoInfo = COCO_Info(user_categorial, dataset_type, anno_file, imgs_dir, res_dir)
visualize = Visualize()
visualize.to_csv()
visualize.plot_bbox_num()
# visualize.plot_cat_info()
# visualize.show_csv()
# visualize.show_cat_area()
# visualize.to_csv()
# visualize.show_bbox_num()
# visualize.plot_bbox_num()
# # step1：准备画图的数据
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [21, 27, 29, 32, 29, 28, 35, 39, 49]
# # step2：手动创建一个figure对象，相当于一个空白的画布
# figure = plt.figure()
# # step3：在画布上添加一个坐标系，标定绘图位置
# axes1 = figure.add_subplot(1, 1, 1)
# # step4：图片基本设置
# # 设置线条颜色、线型、点型
# axes1.plot(x, y, 'ro--')
# # 设置基本信息
# axes1.set_xlabel('time(s)')  # x轴标签
# axes1.set_ylabel('velocity(m/s)')  # y轴标签
# axes1.set_title("title:example")  # 标题
# axes1.text(6, 37, 'vt-demo')  # 文本
# axes1.legend(['vt'])  # 图例
# axes1.grid(linestyle='--', linewidth=1)  # 背景网格
# axes1.annotate('local max', xy=(4, 32), xytext=(4.5, 34),
#                arrowprops=dict(facecolor='black', shrink=0.05))  # 注解
# # step5：展示
# plt.show()
