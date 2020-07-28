import argparse
from pycocotools.coco import COCO
import os
from sklearn.model_selection import train_test_split
import json
import numpy as np
import xml.etree.ElementTree as ET
import time
import pandas as pd
import matplotlib.pyplot as plt
import math

# user_categorial = ["person"]
# anno_file = "/data_process/data_evaluate/anno/instances_val2017.json"
# imgs_dir = "/data_process/data_evaluate/data"
# dataset_type = "CocoDataset"
# res_dir = "/data_process/data_evaluate/res_dir"
# TMP_ANNO_FILE_DIR = '/gddi_output_config/gddi_data'



class COCO_Info(object):

    def __init__(self, user_categorial, dataset_type, anno_file, imgs_dir, res_dir):

        self.user_categorial = user_categorial # user need categorial
        self.dataset_type = dataset_type # dataset type, coco or voa
        self.anno_file = anno_file # annotation file. coco(json) VOA(xml or txt)
        self.imgs_dir = imgs_dir # dataset(images)
        self.res_dir = res_dir # res dir

    # check args
    def check(self):
        """
        initial parse
        """
        # anno_file
        SUPPORTED_DATATYPE = ['CocoDataset', 'VOCDataset']
        if not os.path.exists(self.anno_file):
            raise ValueError("the annotaion file {} not exist".format(self.anno_file))

        # dataset_type
        if dataset_type not in SUPPORTED_DATATYPE:
            raise ValueError("{} not suport".format(dataset_type))

        # imgs_dir
        if not os.path.exists(imgs_dir):
            raise ValueError("the imgs_dir {} not exist".format(imgs_dir))

        # res_dir
        if not os.path.exists(res_dir):
            self.mkdir(res_dir)
            # raise ValueError("the res_dir {} not exist".format(res_dir))

        # user_categorial
        class_num_dict = self.coco_class_num_dic()
        for cat in user_categorial:
            if cat not in class_num_dict:
                raise ValueError("{} class not exist".format(cat))

        # imgs_dir
        filename = self.coco_file_name()
        for img in os.listdir(imgs_dir):
            if img not in filename:
                pass
                # raise ValueError("{} not in annotation".format(img))
        print("every is fine!")

    # get number info
    def get_coco_image_sum(self):
        """
        get image num
        :return: image_sum
        """
        coco = COCO(self.anno_file)
        # image sum
        image_sum = len(coco.imgs)
        return image_sum
    def get_coco_class_sum(self):
        """
        get class_sum
        :return: class_num_dict
        """
        coco = COCO(self.anno_file)
        class_sum = len(coco.cats)
        return class_sum
    def get_coco_class_num_dic(self):
        """
        get class_num_dic
        :return: class_num_dict
        """
        coco = COCO(self.anno_file)
        class_num_dict = dict()
        for cat in coco.dataset["categories"]:
            class_num_dict[cat['name']] = len(set(coco.catToImgs[cat["id"]]))
        return class_num_dict
    def get_coco_file_name(self):
        """
        :return: filename(list)
        """
        file_name = []
        coco = COCO(self.anno_file)
        imgs = coco.dataset["images"]
        for img in imgs:
            file_name.append(img["file_name"])
        return file_name

    # get annotation info
    def get_all_bbox_num(self):
        """
        get all annotations per class
        for example: person: 20 bbox
                     dog: 100 bbox
        :return: dic_res(dic) : bbox of per class
                 {"1": [200, 0.2], "2": [300, 0.3]}
                 {"catId": [bbox_num, Proportion]}
        """
        coco = COCO(self.anno_file)
        dic_res = {}
        bbox_num = {}
        proportion_each_cat = {}
        cat_ids = coco.getCatIds()  # 1～90, but sum id is None so all id sum is 80
        for cat_id in cat_ids:
            annIds = coco.getAnnIds(catIds=cat_id, iscrowd=None)
            anns = coco.loadAnns(annIds)
            dic_res[cat_id] = len(anns)
        # print(dic_res)
        total_bbox = sum(dic_res.values())

        # print(sum(dic_res.values()))
        for key, val in dic_res.items():
            # dic_res[key]

            p = val / total_bbox
            bbox_num[key] = val
            proportion_each_cat[key] = p
            # dic_res[key] = [val, p]
            # print(bbox_num)
            # print(proportion_each_cat)
        # print(dic_res)
        return bbox_num, proportion_each_cat
    def get_all_bbox_info(self):
        """
            get all bbox of annotations per class
            for example: person:[27, 201, 209, 68] ....
            :return: res(dic) : bbox of per class
        """
        coco = COCO(self.anno_file)
        cat_ids = coco.getCatIds()  # 1～90
        bbox_info = {}
        for cat_id in cat_ids:
            annIds = coco.getAnnIds(catIds=cat_id, iscrowd=None)
            anns = coco.loadAnns(annIds)
            temp = [] # each class bbox info
            for ann in anns:
                temp.append(ann["bbox"])
            bbox_info[cat_id] = temp
        # print(bbox_info)
        return bbox_info
    def get_all_bbox_area(self):
        """
            get all bbox area per class
            for example: person:[area1, area2, area3] ....
            :return: res(dic) : bbox of per class
        """
        bbox_area = {}
        info = self.get_all_bbox_info()
        for key, vals in info.items():
            areas = []
            for val in vals:
                w = val[2]
                h = val[3]
                area = w * h
                areas.append(area)
            bbox_area[key] = areas
        return bbox_area
    def compute_iou(self, box1, box2, wh=True):
        """
        compute the iou of two boxes.
        Args:
            box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
            wh: the format of coordinate.
        Return:
            iou: iou of box1 and box2.
        """
        if wh == False:
            xmin1, ymin1, xmax1, ymax1 = box1
            xmin2, ymin2, xmax2, ymax2 = box2
        else:
            xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
            xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
            xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
            xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

        ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
        xx1 = np.max([xmin1, xmin2])
        yy1 = np.max([ymin1, ymin2])
        xx2 = np.min([xmax1, xmax2])
        yy2 = np.min([ymax1, ymax2])

        ## 计算两个矩形框面积
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
        # print((inter_area))
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比
        # print(iou)
        return iou
    def get_every_image_info(self):
        """
            针对每张数据，用每张图像的id作键， 每个bbox的面积， 总的 bbox数量， 每个bbox之间的距离与两两两之间的交并比
            get all annotations per image
            for example: person: [x, y, w, h, area], num_bbox
                         dog:[x, y, w, h, area], num_bbox
            :return: res(dic) : {15335:[[x, y, w, h, area], num_bbox]}
                     res(dic) ：{imgId：[[x, y, w, h, area], num_bbox, [d1, d2, d3], [olap1, olap2, olap3]]}
        """
        each_img_info = {}  # the last result
        each_img_bbox_area = {}
        each_img_num_bbox = {}
        each_img_bbox_dist = {}
        each_img_bbox_overlap = {}
        coco = COCO(self.anno_file)
        catIds = coco.getCatIds(coco.getCatIds())
        imgIds = coco.getImgIds(catIds=catIds)
        for imgId in imgIds:
            annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
            anns = coco.loadAnns(annIds)
            temp = []  # info of each image.such as w, h, area
            center = [] # [c_x, c_y, w, h] each image
            areas = [] # bbox area
            dist_bbox = [] # distance bbox
            overlap = []  # overlap area
            num_bbox = len(anns)  # sum of box per image

            # get bbox basic info
            # ex: w, h, c_x, c_y
            for ann in anns:
                w = ann["bbox"][2]
                h = ann["bbox"][3]
                area = w * h  # area
                areas.append(area)

                c_x = ann["bbox"][0] + w / 2
                c_y = ann["bbox"][1] + h / 2
                center.append([c_x, c_y, w, h])

                # temp.append(ann["bbox"] + [area])

            # bbox distacne
            for i in range(len(center) - 1):
                for j in range(i + 1, len(center)):
                    # distacne
                    abs_dist = (center[i][0] - center[j][0]) ** 2 + (center[i][1] - center[j][1]) ** 2
                    abs_dist = math.sqrt(abs_dist)
                    abs_w = (center[i][2] + center[j][2]) / 2
                    dist = abs_dist / abs_w
                    dist_bbox.append(dist)

                    # iou
                    iou = self.compute_iou(center[i], center[j], wh=True)
                    overlap.append(iou)

            each_img_info[imgId] = [areas, [num_bbox], dist_bbox, overlap]
            each_img_bbox_area[imgId] = areas
            each_img_num_bbox[imgId] = num_bbox
            each_img_bbox_dist[imgId] = dist_bbox
            each_img_bbox_overlap[imgId] = overlap
        # print(each_img_bbox_area[15335])
        # print(each_img_num_bbox[15335])
        # print(each_img_bbox_dist[15335])
        # print(each_img_bbox_overlap[15335])
        return each_img_bbox_area, each_img_num_bbox, each_img_bbox_dist, each_img_bbox_overlap
class Visualize(object):

    def __init__(self):
        pass

    def plot_intensive(self, x, y, dst_dir="/data_process/test_result/visulize_img/"):
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        plt.bar(x, y)
        plt.xlabel(u"nums")
        plt.ylabel(u"rato")
        img_path = dst_dir + "coco.png"
        plt.savefig(img_path)
        plt.show()
    # visualize image
    def show_few_shot(self, x, y):
        pass
    def show_bbox_num(self):
        """
        return : nums(dic) : {"catId":bbox_num}
                 propertion(dic) : {"catId":propertion}
        """
        bbox_num, propertion = cocoInfo.get_all_bbox_num()
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


    # save json
    def mkdir(self, dir_path):
        """
        make dir
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    def save_coco_dataset_info(self, old_path, new_path):
        """
        save old annotation json to new annotation json
        :param old_path: old annotation json path
        :param new_path: new annotation json path
        """
        if not os.path.exists(new_path):
            self.mkdir(res_dir)
        pass
        pass
class Process_dataset(object):
    def __init__(self, user_categorial, dataset_type, anno_file, imgs_dir, res_dir):

        self.user_categorial = user_categorial  # user need categorial
        self.dataset_type = dataset_type  # dataset type, coco or voa
        self.anno_file = anno_file  # annotation file. coco(json) VOA(xml or txt)
        self.imgs_dir = imgs_dir  # dataset(images)
        self.res_dir = res_dir  # res dir



    def sub_class_coco(self):
        """
        save user sub catagerial
        :return: sub_name(json)
        """
        coco = COCO(self.anno_file)
        cat_ids = coco.getCatIds(catNms=self.user_categorial)
        cates = []
        for cat in coco.dataset["categories"]:
            if cat["id"] in cat_ids:
                cates.append(cat)

        coco_sub_class_output = {
            "categories": cates,
            "images": [],
            "annotations": []
        }

        # sub_ids
        ids1 = set()

        # train_orig
        img_cat_dict = {}
        for cat_id in cat_ids:
            img_ids = coco.catToImgs[cat_id]
            for img_id in img_ids:
                img_cat_dict[img_id] = cat_id
            ids1 |= set(img_ids)

        # make sub json
        for img_id in ids1:
            image_info = coco.loadImgs(img_id)[0]
            coco_sub_class_output["images"].append(image_info)
            anno_ids = coco.getAnnIds(imgIds=img_id)
            for anno_id in anno_ids:
                anno_info = coco.loadAnns(anno_id)[0]
                if anno_info["category_id"] in cat_ids:
                    coco_sub_class_output["annotations"].append(anno_info)
        (filepath, tempfilename) = os.path.split(anno_file)
        sub_name = 'sub_' + tempfilename
        save_path = os.path.join(self.res_dir, sub_name)
        with open(save_path, 'w') as f:
            json.dump(coco_sub_class_output, f)
        print(save_path)
        return save_path

    def split_coco_train_val(self, annotations_path, split_ratio=0.8):
        """
         after from a original annotation json to a sub catogrial json
         :paras: annotations_path: a new json path
                 split_ratio: default = 0.8
         :return: image_sum: auto_train_name(path json), auto_val_name(path json)
        """


        coco = COCO(annotations_path)
        cat_ids = coco.getCatIds(catNms=self.user_categorial)
        if len(self.user_categorial) == 0:
            cates = coco.dataset["categories"]
        else:
            cates = []
            for cat in coco.dataset["categories"]:
                if cat["id"] in cat_ids:
                    cates.append(cat)

        coco_output_train = {
            "categories": cates,
            "images": [],
            "annotations": []
        }

        coco_output_val = {
            "categories": cates,
            "images": [],
            "annotations": []
        }

        ids1 = set()

        # train_orig
        img_cat_dict = {}
        for cat_id in cat_ids:
            img_ids = coco.catToImgs[cat_id]
            for img_id in img_ids:
                img_cat_dict[img_id] = cat_id
            ids1 |= set(img_ids)  # just for check number. equal to (train + val) , same sum as "img_cat_dict".

        img_ids_list = []
        cat_list_according2img = []
        for img_id in img_cat_dict:
            img_ids_list.append(img_id)
            cat_list_according2img.append(img_cat_dict[img_id])
        train_image_ids, val_image_ids = train_test_split(img_ids_list, shuffle=True, train_size=split_ratio,
                                                          stratify=cat_list_according2img)

        print('Total len', len(ids1))
        print('Train len', len(train_image_ids))
        print('Val len', len(val_image_ids))

        # train
        for img_id in train_image_ids:
            image_info = coco.loadImgs(img_id)[0]
            coco_output_train["images"].append(image_info)
            anno_ids = coco.getAnnIds(imgIds=img_id)
            for anno_id in anno_ids:
                anno_info = coco.loadAnns(anno_id)[0]
                if anno_info["category_id"] in cat_ids:
                    coco_output_train["annotations"].append(anno_info)

        # val
        for img_id in val_image_ids:
            image_info = coco.loadImgs(img_id)[0]
            coco_output_val["images"].append(image_info)
            anno_ids = coco.getAnnIds(imgIds=img_id)
            for anno_id in anno_ids:
                anno_info = coco.loadAnns(anno_id)[0]
                if anno_info["category_id"] in cat_ids:
                    coco_output_val["annotations"].append(anno_info)

        # save anno file
        (filepath, tempfilename) = os.path.split(annotations_path)
        auto_train_name = 'auto_train.json'
        auto_val_name = 'auto_val.json'
        auto_train_save_path = os.path.join(self.res_dir, auto_train_name)
        auto_val_save_path = os.path.join(self.res_dir, auto_val_name)

        print(auto_train_save_path, auto_val_save_path)
        with open(auto_train_save_path, 'w') as f:
            json.dump(coco_output_train, f)
        with open(auto_val_save_path, 'w') as f:
            json.dump(coco_output_val, f)

        return auto_train_name, auto_val_name

    def resize_bbox(self, annotations_path, h_resize=100, w_resize=200):
        """
              to resize bbox according given h_resize and w_resize
              :paras : annotations_path : need to resize json
                       h_resize : new hight size
                       w_resize : new width size
                       new_annotation_path : new josn name (eg: "/path/new.json")
               :return : return resized annotation
        """
        coco = COCO(annotations_path)
        images = coco.dataset["images"]
        annotation = []

        for i in range(len(images)):
            h_rato = h_resize / images[i]["height"]
            w_rato = w_resize / images[i]["width"]
            annIds = coco.getAnnIds(imgIds=images[i]["id"], iscrowd=None)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                ann["bbox"][0] = ann["bbox"][0] * w_rato
                ann["bbox"][1] = ann["bbox"][1] * h_rato
                ann["bbox"][2] = ann["bbox"][2] * w_rato
                ann["bbox"][3] = ann["bbox"][3] * h_rato
            annotation.append(anns)
        return annotation

    def save_resized_to_json(self, annotations_path, new_annotation_path):

        """
          to save the whole annotation after resize
          :paras : annotations_path : need to resize json
                   new_annotation_path : new josn name (eg: "/path/new.json")
        """

        data = json.load(open(annotations_path, 'r'))
        data_2 = {}
        data_2['info'] = data['info']
        data_2['licenses'] = data['licenses']
        data_2['images'] = data['images']  # 只提取第一张图片
        data_2['categories'] = data['categories']
        data_2['annotations'] = self.resize_bbox(h_resize=100, w_resize=200)
        json.dump(data_2, open(new_annotation_path, 'w'), indent=4)  # indent=4 更加美观显示


    def get(self, root, name):
        # find name in xml file
        # root is a memory address
        vars = root.findall(name)
        return vars

    def get_and_check(self, root, name, length):

        # get contents of the name
        vars = root.findall(name)
        if len(vars) == 0:
            raise ValueError("Can not find %s in %s." % (name, root.tag))
        if length > 0 and len(vars) != length:
            raise ValueError(
                "The size of %s is supposed to be %d, but is %d."
                % (name, length, len(vars))
            )
        if length == 1:
            vars = vars[0]
        return vars

    def get_categories(self, xml_files):
        """Generate category name to id mapping from a list of xml files.

        Arguments:
            xml_files {list} -- A list of xml file paths.

        Returns:
            dict -- category name to id mapping.
        """
        classes_names = []
        for xml_file in xml_files:
            if not os.path.exists(xml_file):
                raise ValueError('xml_file: {} does not exist!'.format(xml_file))
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall("object"):
                classes_names.append(member[0].text)
        classes_names = list(set(classes_names))
        # classes_names.sort()
        # return 'name':id
        return classes_names  # {name: i for i, name in enumerate(classes_names)}

    def convert(self, voc_dir, xml_files, image_dir, json_file, categories):
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(voc_dir, '{}.log'.format(timestamp))
        # logger = get_root_logger(log_file=log_file, log_level='INFO')

        json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

        # logger.info('{}'.format(json_file))
        # logger.info('Number of xml files: {}'.format(len(xml_files)))
        print("Number of xml files: {}".format(len(xml_files)))

        bnd_id = 1
        image_id = 1
        empty_object_list = []
        categories_in_xml = set()
        # annotations of each images
        for xml_file in xml_files:
            if not os.path.exists(xml_file):
                raise ValueError('xml_file: {} does not exist!'.format(xml_file))

            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename = self.get_and_check(root, "filename", 1).text
            file_path = os.path.join(image_dir, filename)
            # if not os.path.exists(file_path):
            #     raise ValueError('file_path: {} does not exist!'.format(file_path))

            size = self.get_and_check(root, "size", 1)
            width = int(self.get_and_check(size, "width", 1).text)
            height = int(self.get_and_check(size, "height", 1).text)

            if len(self.get(root, "object")) == 0:
                # logger.info('No Object in :{}'.format(filename))
                # print('No Object in :', filename)
                empty_object_list.append(filename)

            has_box = False
            for obj in self.get(root, "object"):
                category = self.get_and_check(obj, "name", 1).text
                if category not in categories:
                    continue
                categories_in_xml.add(category)
                category_id = categories.index(category)
                bndbox = self.get_and_check(obj, "bndbox", 1)
                has_box = True
                # coco 0_base voc 1_base
                xmin = int(float(self.get_and_check(bndbox, "xmin", 1).text)) - 1
                ymin = int(float(self.get_and_check(bndbox, "ymin", 1).text)) - 1
                xmax = int(float(self.get_and_check(bndbox, "xmax", 1).text))
                ymax = int(float(self.get_and_check(bndbox, "ymax", 1).text))
                assert xmax > xmin
                assert ymax > ymin
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    "area": o_width * o_height,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [xmin, ymin, o_width, o_height],
                    "category_id": category_id,
                    "id": bnd_id,
                    "ignore": 0,
                    "segmentation": [],
                }
                json_dict["annotations"].append(ann)
                bnd_id = bnd_id + 1

            if not has_box:
                empty_object_list.append(filename)
                continue
            else:
                image = {
                    "file_name": filename,
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
                json_dict["images"].append(image)
                image_id += 1

        wrong_cate = []
        for i in categories:
            if i not in categories_in_xml:
                wrong_cate.append(i)
        if len(wrong_cate) != 0:
            raise ValueError('category: {} does not exist in annotations!'.format(wrong_cate))

        for cate in categories:
            cat = {"supercategory": "none", "id": categories.index(cate), "name": cate}
            json_dict["categories"].append(cat)

        json_file = os.path.join(TMP_ANNO_FILE_DIR, json_file)
        json_fp = open(json_file, "w")
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()

        for n, i in enumerate(empty_object_list):
            # logger.info('{}/{} No object in image:{}'.format(n, len(empty_object_list),i))
            print(n, '/', len(empty_object_list), 'No object in image:', i)

        # logger.info('Number of images without object:{}'.format(len(empty_object_list)))
        # logger.info('Number of images with object:{}'.format(image_id - len(empty_object_list)))
        # logger.info('Total images:{}'.format(image_id))
        print('Number of images without object:', len(empty_object_list))
        print('Number of images with object:', image_id - 1)

    def voc_to_coco(self, voc_dir, txt_file, json_dir, json_name, image_dir_name):
        '''
        :param voc_dir:  string
            e.g. voc_dir = '~/Data/VOC/VOCdevkit/VOC2007/'
        :param class_json_path:  string
            e.g. class_json_path = '~/Data/VOC/VOCdevkit/VOC2007/class.json'
        :return: image_dir, json_list
        '''

        # voc dir
        if not os.path.exists(voc_dir):
            raise ValueError('voc_dir: {} does not exist!'.format(voc_dir))

        # image dir
        image_dir = os.path.join(voc_dir, image_dir_name)
        # if not os.path.exists(image_dir):
        #     raise ValueError('image_dir: {} does not exist!'.format(image_dir))

        # xml_dir
        xml_dir = os.path.join(voc_dir, 'Annotations/')
        if not os.path.exists(xml_dir):
            raise ValueError('xml_dir: {} does not exist!'.format(xml_dir))

        # todo json_save_dir
        json_save_dir = os.path.join(voc_dir, json_dir)
        if not os.path.exists(json_save_dir):
            os.mkdir(json_save_dir)

        # txt_path
        txt_path = os.path.join(voc_dir, txt_file)
        if not os.path.exists(txt_path):
            raise ValueError('txt_path: {} does not exist!'.format(txt_path))
        json_save_path = os.path.join(json_save_dir, json_name)
        print('Converting', os.path.split(txt_path)[-1], 'to COCO format')
        with open(txt_path) as f:
            xml_files = f.readlines()
            xml_files = [line.strip() for line in xml_files]
        xml_files = [os.path.join(xml_dir, item + '.xml') for item in xml_files]

        # categories: if None, get all categories from xml_files
        # if class_json_path is not None:
        #     categories = json.load(open(class_json_path, 'r'))['PRE_DEFINE_CATEGORIES']
        # else:
        print('Getting all categories from xml_files')
        categories = self.get_categories(xml_files)
        json_save_path = os.path.join(TMP_ANNO_FILE_DIR, json_name)
        self.convert(voc_dir, xml_files, image_dir, json_save_path, categories)
        return os.path.join(json_dir, json_name)
def parse_args():
    """
    add user parse
    """
    parser = argparse.ArgumentParser(description='data_processing')
    parser.add_argument('--user_class', default=None,
                        type=str, help='user need categorial')
    parser.add_argument('--dataset_type', default=None,
                        type=str, help='dataset type, coco or voa')
    parser.add_argument('--anno_file', default=None,
                        type=str, help='annotation file. coco(json) VOA(xml or txt)')
    parser.add_argument('--imgs_dir', default=None,
                        type=str, help='dataset(images)')
    args = parser.parse_args()
    print(args)



# if __name__ == "__main__":
#     cocoInfo = COCO_Info(user_categorial, dataset_type, anno_file, imgs_dir, res_dir)
#     process_dataset = Process_dataset(user_categorial, dataset_type, anno_file, imgs_dir, res_dir)
#
#     voc_dir = "/coco_dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
#     # txt_file = "/coco_dataset/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Layout/test.txt"
#     txt_file = "/data_process/voc_few_shot/voc_train_30.txt"
#     json_dir = "/temp_res"
#     json_name = "voc_to_coco_fewshot.json"
#     image_dir_name= "/coco_dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
#     name = process_dataset.voc_to_coco(voc_dir, txt_file, json_dir, json_name, image_dir_name)
#     print(name)
    # sub_annotation_path = process_dataset.sub_class_coco()
    # process_dataset.split_coco_train_val(anno_file)
