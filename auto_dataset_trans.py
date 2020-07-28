import argparse
import os
import json
import xml.etree.ElementTree as ET
import time
import copy
import shutil
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from mmdet.utils import get_root_logger

TMP_ANNO_FILE_DIR = '/gddi_output_config/gddi_data'
def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get(root, name):
    # find name in xml file
    # root is a memory address
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):

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


def get_categories(xml_files):
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


def convert(voc_dir, xml_files, image_dir, json_file, categories):
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
        filename = get_and_check(root, "filename", 1).text
        file_path = os.path.join(image_dir, filename)
        # if not os.path.exists(file_path):
        #     raise ValueError('file_path: {} does not exist!'.format(file_path))

        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        if len(get(root, "object")) == 0:
            # logger.info('No Object in :{}'.format(filename))
            # print('No Object in :', filename)
            empty_object_list.append(filename)

        has_box = False
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                continue
            categories_in_xml.add(category)
            category_id = categories.index(category)
            bndbox = get_and_check(obj, "bndbox", 1)
            has_box = True
            # coco 0_base voc 1_base
            xmin = int(float(get_and_check(bndbox, "xmin", 1).text)) - 1
            ymin = int(float(get_and_check(bndbox, "ymin", 1).text)) - 1
            xmax = int(float(get_and_check(bndbox, "xmax", 1).text))
            ymax = int(float(get_and_check(bndbox, "ymax", 1).text))
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


def voc_to_coco(voc_dir, txt_file, json_dir, json_name, image_dir_name):
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
    categories = get_categories(xml_files)
    json_save_path =  os.path.join(TMP_ANNO_FILE_DIR, json_name)
    convert(voc_dir, xml_files, image_dir, json_save_path, categories)

    return os.path.join(json_dir, json_name)


def sub_class_coco(annotations_path, CATEGORIES=[], save_path=None):
    coco = COCO(annotations_path)

    # get all class
    cats_name_list = []
    for c in coco.dataset["categories"]:
        cats_name_list.append(c['name'])
    # check CATEGORIES
    wrong_cats = []
    for n in CATEGORIES:
        if n not in cats_name_list:
            wrong_cats.append(n)
    if len(wrong_cats) > 0:
        raise ValueError('CATEGORY: {} does not exist in dataset!'.format(wrong_cats))

    cat_ids = coco.getCatIds(catNms=CATEGORIES)
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

    # save name to sub_name.json
    if not save_path:
        (filepath, tempfilename) = os.path.split(annotations_path)
        sub_name = 'sub_' + tempfilename
    save_path = os.path.join(TMP_ANNO_FILE_DIR, sub_name)
    with open(save_path, 'w') as f:
        json.dump(coco_sub_class_output, f)

    return sub_name


def split_coco_train_val(annotations_path, split_ratio, CATEGORIES=[]):
    coco = COCO(annotations_path)
    cat_ids = coco.getCatIds(catNms=CATEGORIES)
    if len(CATEGORIES) == 0:
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
    auto_train_save_path = os.path.join(TMP_ANNO_FILE_DIR, auto_train_name)
    auto_val_save_path = os.path.join(TMP_ANNO_FILE_DIR, auto_val_name)

    print(auto_train_save_path,auto_val_save_path)
    with open(auto_train_save_path, 'w') as f:
        json.dump(coco_output_train, f)
    with open(auto_val_save_path, 'w') as f:
        json.dump(coco_output_val, f)

    return auto_train_name, auto_val_name

# todo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! new 9 new 13 save_dir
def split_coco_train_val_few_shot(annotations_path,
                                  split_ratio,
                                  CATEGORIES=[],
                                  auto_train_save_path=None,
                                  auto_val_save_path=None):
    split_ratio = split_ratio/(1-split_ratio)
    coco = COCO(annotations_path)
    cat_ids = coco.getCatIds(catNms=CATEGORIES)
    if len(CATEGORIES) == 0:
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
        ids1 |= set(img_ids)  # just for check number. equal to (train + val) , same "sum" as "img_cat_dict".

    # # split
    # img_ids_list = []
    # cat_list_according2img = []
    # for img_id in img_cat_dict:
    #     img_ids_list.append(img_id)
    #     cat_list_according2img.append(img_cat_dict[img_id])
    # train_image_ids, val_image_ids = train_test_split(img_ids_list, shuffle=True, train_size=split_ratio,
    #                                                   stratify=cat_list_according2img)

    # split image by image
    train_image_ids, val_image_ids = [],[]
    # ratio dict
    train_class_num_dict = dict()
    val_class_num_dict = dict()
    ratio_dict = dict()

    for cat_id in cat_ids:
        train_class_num_dict[cat_id] = 0
        val_class_num_dict[cat_id] = 0
        ratio_dict[cat_id] = 100

    for img_id in list(ids1):
        anno_ids = coco.getAnnIds(imgIds=img_id)
        img_cat_ids = set()
        max_ratio = 0
        for anno_id in anno_ids:
            anno_info = coco.loadAnns(anno_id)[0]
            cat_id = anno_info["category_id"]
            if cat_id in cat_ids:
                img_cat_ids.add(cat_id)
                if ratio_dict[cat_id] > max_ratio:
                    max_ratio = ratio_dict[cat_id]

        if max_ratio >= split_ratio:
            val_image_ids.append(img_id)
            for cat_id in list(img_cat_ids):
                val_class_num_dict[cat_id] += 1
                ratio_dict[cat_id] = train_class_num_dict[cat_id]/val_class_num_dict[cat_id]
        else:
            train_image_ids.append(img_id)
            for cat_id in list(img_cat_ids):
                train_class_num_dict[cat_id] += 1
                ratio_dict[cat_id] = train_class_num_dict[cat_id]/val_class_num_dict[cat_id]

    for i in val_class_num_dict.items():
        if i[1] == 0:
            raise ValueError('image of class {} in split val_dataset is empty, check the data used for split!'.format(coco.loadCats(i[0])))
    for i in train_class_num_dict.items():
        if i[1] == 0:
            raise ValueError('image of class {} in split train_dataset is empty, check the data used for split!'.format(coco.loadCats(i[0])))

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
    # todo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! new 14 save_dir 478-487
    if (auto_train_save_path and auto_val_save_path):
        (filepath, auto_train_name) = os.path.split(os.path.abspath(auto_train_save_path))
        (filepath, auto_val_name) = os.path.split(os.path.abspath(auto_val_save_path))
    else:
        (filepath, tempfilename) = os.path.split(annotations_path)
        auto_train_name = 'auto_train.json'
        auto_val_name = 'auto_val.json'
        auto_train_save_path = os.path.join(filepath, auto_train_name)
        auto_val_save_path = os.path.join(filepath, auto_val_name)
        auto_train_save_path = os.path.join(TMP_ANNO_FILE_DIR, auto_train_name)
        auto_val_save_path = os.path.join(TMP_ANNO_FILE_DIR, auto_val_name)
    with open(auto_train_save_path, 'w') as f:
        json.dump(coco_output_train, f)
    with open(auto_val_save_path, 'w') as f:
        json.dump(coco_output_val, f)

    return auto_train_name, auto_val_name



# Dataset info
def coco_dataset_ann(coco_datasets_path):
    """
    :param coco_datasets_path: annotation json path

    """
    coco = COCO(coco_datasets_path)


# todo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! new 3
def coco_dataset_info(coco_datasets_path):
    """
    :param coco_datasets_path: annotation json path
    :return: image_sum, class_sum, class_num_dict
    """
    # use coco-api read
    print("508:", coco_datasets_path)
    coco = COCO(coco_datasets_path)
    # image sum
    image_sum = len(coco.imgs)
    # class number
    class_sum = len(coco.cats)
    # class images number
    class_num_dict = dict()
    for cat in coco.dataset["categories"]:
        class_num_dict[cat['name']] = len(set(coco.catToImgs[cat["id"]]))
    # print and log
    return image_sum, class_sum, class_num_dict


# todo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  new 4
# Dataset info save
def save_coco_dataset_info(jsoncfg, save_path):
    data_dir = jsoncfg['data_dir']
    dataset_info_jsontext = {"dataset_info": []}
    train_json_path = jsoncfg['args']['data']['train_ann_file']
    if train_json_path != "":
        train_json_abspath = os.path.join(data_dir, train_json_path)
        image_sum, class_sum, class_num_dict = coco_dataset_info(train_json_abspath)
        # dataset_info_jsontext["dataset_info"].append("train_set_info:")
        dataset_info_jsontext["dataset_info"].append({"train_set_image number": image_sum})
        dataset_info_jsontext["dataset_info"].append({"train_set_class number": class_sum})
        dataset_info_jsontext["dataset_info"].append({"train_set_classes": list(class_num_dict.keys())})
        dataset_info_jsontext["dataset_info"].append({"train_set_class_image_number": class_num_dict})

    val_json_path = jsoncfg['args']['data']['val_ann_file']
    if val_json_path != "":
        val_json_abspath = os.path.join(data_dir, val_json_path)
        image_sum, class_sum, class_num_dict = coco_dataset_info(val_json_abspath)
        # dataset_info_jsontext["dataset_info"].append("val_set_info:")
        dataset_info_jsontext["dataset_info"].append({"val_set_image number": image_sum})
        dataset_info_jsontext["dataset_info"].append({"val_set_class number": class_sum})
        dataset_info_jsontext["dataset_info"].append({"val_set_classes": list(class_num_dict.keys())})
        dataset_info_jsontext["dataset_info"].append({"val_set_class_image_number": class_num_dict})

    test_json_path = jsoncfg['args']['data']['test_ann_file']
    if test_json_path != "":
        test_json_abspath = os.path.join(data_dir, test_json_path)
        image_sum, class_sum, class_num_dict = coco_dataset_info(test_json_abspath)
        # dataset_info_jsontext["dataset_info"].append("test_set_info:")
        dataset_info_jsontext["dataset_info"].append({"test_set_image number": image_sum})
        dataset_info_jsontext["dataset_info"].append({"test_set_class number": class_sum})
        dataset_info_jsontext["dataset_info"].append({"test_set_classes": list(class_num_dict.keys())})
        dataset_info_jsontext["dataset_info"].append({"test_set_class_image_number": class_num_dict})
    # save info txt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset_json_fp = open(save_path, "w")
    dataset_json_str = json.dumps(dataset_info_jsontext, indent=4)
    dataset_json_fp.write(dataset_json_str)
    dataset_json_fp.close()


# todo !!!!!!!!!!!!!!!!!!!!!!!!! new 10
def deep_split(annotation_path,
               use_subclass=True,
               biggest_subclass_number=10,
               sub_class_save_path="./temp_sub_class.json",
               expect_num=500,
               split_keep_save_path="./split_keep.json",
               split_remove_save_path="./split_remove.json"
               ):
               
    image_sum, class_sum, class_num_dict = coco_dataset_info(annotation_path)

    if expect_num > image_sum:
        return annotation_path, image_sum, class_sum, class_num_dict


    if use_subclass:
        class_number_list_ = sorted(class_num_dict.items(), key=lambda x: x[1], reverse=True)
        class_number_list = [i[0] for i in class_number_list_][:biggest_subclass_number]
        sub_class_coco(annotation_path, CATEGORIES=class_number_list, save_path=sub_class_save_path)
        annotation_path = sub_class_save_path
        image_sum, class_sum, class_num_dict = coco_dataset_info(annotation_path)
    if expect_num > image_sum:
        return annotation_path, image_sum, class_sum, class_num_dict

    split_ratio = expect_num/image_sum
    split_coco_train_val_few_shot(
        annotation_path,
        split_ratio,
        CATEGORIES=[],
        auto_train_save_path=split_keep_save_path,
        auto_val_save_path=split_remove_save_path)
    image_sum, class_sum, class_num_dict = coco_dataset_info(split_keep_save_path)
    return split_keep_save_path, image_sum, class_sum, class_num_dict

def parse_args():
    # 传入参数
    parser = argparse.ArgumentParser(description='data_processing')
    parser.add_argument('--config_file', default=None,
                        type=str, help='config file path')
    parser.add_argument('--task_type', default=None,
                        type=int, help='task_type')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not os.path.exists(args.config_file):
        raise ValueError('the config file is not exist {}'.format(args.config_file))
    if  not isinstance(args.task_type, int) or args.task_type > 1 :
        raise ValueError('task type  is not support {}'.format(args.task_type))

    curr_path = os.getcwd()

    # read config
    with open(args.config_file) as cfg_hd:
        jsoncfg = json.loads(cfg_hd.read())
        new_jsoncfg = copy.deepcopy(jsoncfg)


    if not jsoncfg["job_id"] :
        raise ValueError('please provide job id {}'.format(jsoncfg["job_id"]))

    # split sign and ratio
    split_sign = False
    split_train_ratio = 0.8
    jsoncfg['data_dir'] = '/gddi_data'

    print(TMP_ANNO_FILE_DIR)
    os.makedirs(TMP_ANNO_FILE_DIR, exist_ok=True)
    dataset_info_json_path = '/gddi_output/dev_gddi_all_dataset_info.json'
    subclass_dataset_info_json_path = '/output/subclass_dataset_info.json'
    new_cfg_path = args.config_file
    print(new_cfg_path)

    # data_dir
    print(jsoncfg['data_dir'])
    if not jsoncfg['data_dir'] or not os.path.exists(jsoncfg['data_dir']):
        raise ValueError('data path: {} does not exist'.format(jsoncfg['data_dir']))
    else:
        data_dir = jsoncfg['data_dir']

    # check data_type: VOC or COCO
    SUPPORTED_DATATYPE = ['CocoDataset', 'VOCDataset']
    if jsoncfg['args']['data']['data_type'] not in SUPPORTED_DATATYPE:
        raise ValueError('data type: {} is not supported'.format(jsoncfg['args']['data']['data_type']))
    else:
        dataset_type = jsoncfg['args']['data']['data_type']

    if args.task_type == 0:
    # VOC: convert VOC to COCO
        if dataset_type == 'VOCDataset':
            print('NOTICE : VOC -- >> COCO')
            new_jsoncfg['args']['data']['data_type'] = 'CocoDataset'

            # convert whole VOC to COCO
            convert_list = []
            for i, k in enumerate(jsoncfg['args']['data'].keys()):
                print(jsoncfg['args']['data'].keys())
                if k[-9:] == '_ann_file':
                    if jsoncfg['args']['data']["train_ann_file"] == jsoncfg['args']['data']["val_ann_file"]:
                        print("config val_ann_file is same as train_ann_file")
                    if jsoncfg['args']['data'][k][-4:] == '.txt':
                        txt_file = jsoncfg['args']['data'][k]
                        next_k = list(jsoncfg['args']['data'].keys())[i + 1]
                        if next_k[-11:] == '_img_prefix':
                            image_dir_name = jsoncfg['args']['data'][next_k]
                        else:
                            raise ValueError('"img_prefix" should follow "ann_file" in config file')
                        # convert
                        json_name = 'auto_' + k[:-9] + '.json'
                        json_dir = '/gddi_output_config/gddi_data'
                        coco_json_path = voc_to_coco(data_dir, txt_file, json_dir, json_name, image_dir_name)
                        new_jsoncfg['args']['data'][k] = os.path.join('/gddi_output_config/gddi_data',json_name)
                        #new_jsoncfg['args']['data'][k] = coco_json_path
                        convert_list.append(k)

                    elif jsoncfg['args']['data'][k] == "":
                        pass

                    else:
                        raise ValueError('Wrong file type in jsoncfg.args.data.{}, should be .txt or ""'.format(k))

            # log, what we convert, if none raise Error
            if len(convert_list) == 0:
                raise ValueError('Nothing converted')

        json_fp = open(new_cfg_path, "w")
        json_str = json.dumps(new_jsoncfg, indent=4)
        json_fp.write(json_str)
        json_fp.close()
        save_coco_dataset_info(new_jsoncfg, dataset_info_json_path)

    elif args.task_type == 1:
        # Check COCO
        # check category check split
        new_2_jsoncfg = copy.deepcopy(new_jsoncfg)

        if new_2_jsoncfg['args']['data']["train_ann_file"] == new_2_jsoncfg['args']['data']["val_ann_file"]:
            print("config val_ann_file is same as train_ann_file")

        train_json_path = new_2_jsoncfg['args']['data']['train_ann_file']
        if train_json_path == "":
            raise ValueError('"train_ann_file" in config should not be empty')
        else:
            train_json_abspath = os.path.join(data_dir, train_json_path)

        val_json_path = new_2_jsoncfg['args']['data']['val_ann_file']
        if val_json_path == "":
            split_sign = True
        else:
            val_json_abspath = os.path.join(data_dir, val_json_path)

        test_json_path = new_2_jsoncfg['args']['data']['test_ann_file']
        if test_json_path == "":
            print('test_ann_file in config is empty')
            # logger.info('test_ann_file in config is empty')
        else:
            test_json_abspath = os.path.join(data_dir, test_json_path)

        CATEGORIES = new_2_jsoncfg['args']['data']['classes']

        # Sub_Class:
        if CATEGORIES:
            print('NOTICE : SUBCLASS')
            if train_json_path == "":
                raise ValueError('"train_ann_file" in config should not be empty')
            else:
                print(train_json_abspath)
                sub_train_json_name = sub_class_coco(train_json_abspath, CATEGORIES)
                sub_train_json_dir = os.path.split(train_json_path)[0]
                sub_train_json_path =  os.path.join(TMP_ANNO_FILE_DIR,sub_train_json_name)
                #sub_train_json_path = os.path.join(sub_train_json_dir, sub_train_json_name)
                new_2_jsoncfg['args']['data']['train_ann_file'] = sub_train_json_path

            if val_json_path == "":
                split_sign = True
            else:
                sub_val_json_name = sub_class_coco(val_json_abspath, CATEGORIES)
                sub_val_json_dir = os.path.split(val_json_path)[0]
                sub_val_json_path =  os.path.join(TMP_ANNO_FILE_DIR,sub_val_json_name) 
                #sub_val_json_path = os.path.join(sub_val_json_dir, sub_val_json_name)
                new_2_jsoncfg['args']['data']['val_ann_file'] = sub_val_json_path

            if test_json_path == "":
                print('test_ann_file in config is empty')
                # logger.info('test_ann_file in config is empty')
            else:
                sub_test_json_name = sub_class_coco(test_json_abspath, CATEGORIES)
                sub_test_json_dir = os.path.split(test_json_path)[0]
                sub_test_json_path =  os.path.join(TMP_ANNO_FILE_DIR,sub_test_json_dir) 
                #sub_test_json_path = os.path.join(sub_test_json_dir, sub_test_json_name)
                new_2_jsoncfg['args']['data']['test_ann_file'] = sub_test_json_path

        # Split train val
        if split_sign:

            # # check few shot learning
            # few_shot_learning_sign = True
            # _image_sum, _class_sum, _class_num_dict = coco_dataset_info(train_json_abspath)
            # for i in _class_num_dict.items():
            #     if i[1] > 50:
            #         few_shot_learning_sign = False

            # split coco
            auto_train_json_name, auto_val_json_name = \
                split_coco_train_val_few_shot(train_json_abspath, split_train_ratio, CATEGORIES)

            # train path
            auto_train_json_dir = os.path.split(train_json_path)[0]
            auto_train_json_path = os.path.join(TMP_ANNO_FILE_DIR, auto_train_json_name)
            #auto_train_json_path = os.path.join(auto_train_json_dir, auto_train_json_name)
            # val path
            auto_val_json_dir = os.path.split(val_json_path)[0]
            auto_val_json_path = os.path.join(TMP_ANNO_FILE_DIR, auto_val_json_name)
            #auto_val_json_path = os.path.join(auto_train_json_dir, auto_val_json_name)

            new_2_jsoncfg['args']['data']['train_ann_file'] = auto_train_json_path
            new_2_jsoncfg['args']['data']['val_ann_file'] = auto_val_json_path
            new_2_jsoncfg['args']['data']["val_img_prefix"] = new_2_jsoncfg['args']['data']["train_img_prefix"]

        # CLASSES
        coco = COCO(train_json_abspath)
        if len(CATEGORIES) == 0:
            cates = []
            for cat in coco.dataset['categories']:
                cates.append(cat['name'])
            CATEGORIES = cates
        new_2_jsoncfg['args']['data']['classes'] = CATEGORIES

        save_coco_dataset_info(new_2_jsoncfg, subclass_dataset_info_json_path)
        json_fp = open(new_cfg_path, "w")
        json_str = json.dumps(new_2_jsoncfg, indent=4)
        json_fp.write(json_str)
        json_fp.close()
    
    

if __name__ == '__main__':
    #try:
    main()
    #except Exception as err:
    #    print('gddi automl data process raise error: {}'.format(str(err)))
    #    os._exit(1)
        