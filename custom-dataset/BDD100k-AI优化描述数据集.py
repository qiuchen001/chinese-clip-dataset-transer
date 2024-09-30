import os
import json
import shutil
from typing import List, Dict
import random
from describe_image import DescribeImage, get_image_base64_str
import re


# 获取标签与标签id影射
def get_category_id_map():
    with open('../bdd100k/category_map.json', 'r', encoding='utf-8') as f:
        category_id_map = json.load(f)
    return category_id_map


# 读取图片字符串名称和数字名称的映射
def get_image_id_map(split):
    image_id_map_filename = "../bdd100k/{}_image_id_map.json".format(split)
    with open(image_id_map_filename, 'r') as f:
        image_id_map = json.load(f)

    return image_id_map


# 从bdd100k图片标签文件中提取标签
def extractTags(jsonFile):
    tags = []
    f = open(jsonFile)
    info = json.load(f)
    objects = info['frames'][0]['objects']
    attributes = info['attributes']

    category_id_map = get_category_id_map()
    for i in objects:
        if i['category'] in category_id_map:
            if i['category'] == 'traffic sign' or i['category'] == 'traffic light':
                name = i['category'] + "/" + i['attributes']['trafficLightColor']
            else:
                name = i['category']
            tagInfo = {
                "name": category_id_map[name]['name'],
                "id": category_id_map[name]['id']
            }
            tags.append(tagInfo)
        else:
            print("new category:" + i['category'])

    if 'weather' in attributes:
        name = 'weather/' + attributes['weather']
        tagInfo = {
            "name": category_id_map[name]['name'],
            "id": category_id_map[name]['id']
        }
        tags.append(tagInfo)

    if 'scene' in attributes:
        name = 'scene/' + attributes['scene']
        tagInfo = {
            "name": category_id_map[name]['name'],
            "id": category_id_map[name]['id']
        }
        tags.append(tagInfo)

    if 'timeofday' in attributes:
        name = 'timeofday/' + attributes['timeofday']
        tagInfo = {
            "name": category_id_map[name]['name'],
            "id": category_id_map[name]['id']
        }
        tags.append(tagInfo)

    # 对图片标签进行去重
    unique_list = list({tuple(d.items()) for d in tags})
    unique_list = [dict(t) for t in unique_list]

    return unique_list


def save_image_by_desc(imageId, desc):
    # 清理描述信息中的非法字符
    cleaned_desc = re.sub(r'[\\/*?:"<>|]', '_', desc)

    src_dir = r"E:\playground\ai\datasets\bdd100k\bdd100k_images\bdd100k\images\100k\train_id"
    dst_dir = r"E:\playground\ai\projects\chinese-clip-dataset-transer\custom-dataset\bdd100k-images"
    src_image_path = os.path.join(src_dir, str(imageId) + ".jpg")
    dst_image_path = os.path.join(dst_dir, cleaned_desc + ".jpg")

    shutil.copy(src_image_path, dst_image_path)


def tag_filter(tags: List[Dict]):
    need_tags = [
        "绿色交通灯",
        "红色交通灯",
        "黄色交通灯",
        "未知的交通灯",
    ]

    new_tags = []
    for index, tag in enumerate(tags):
        if tag['name'] in need_tags:
            new_tags.append(tag['name'])

    return new_tags


def get_image_path(image_id, split):
    image_dir = r"E:\playground\ai\datasets\bdd100k\bdd100k_images\bdd100k\images\100k\{}_id" . format(split)
    image_path = os.path.join(image_dir, str(image_id) + ".jpg")
    return image_path


def main(src_dir, dst_dir, split):
    # 读取图片字符串名称和数字名称的映射
    image_id_map = get_image_id_map(split)

    text_id = 0
    for dir_path, _, filenames in os.walk(src_dir):
        for i, filename in enumerate(filenames):
            # print("processing: {}, {}".format(i, filename))

            # 提取图片标签
            filepath = os.path.join(dir_path, filename)
            tags = extractTags(str(filepath))

            # 过滤标签
            new_tags = tag_filter(tags)

            # 获得图片数字id名称
            imageName = filename.rstrip(".json")
            imageId = image_id_map[imageName]

            if len(new_tags):
                texts_jsonl_filepath = os.path.join(dst_dir, split + "_texts.jsonl")
                text_id = text_id + 1
                # 以标签-图片的形式保存
                with open(texts_jsonl_filepath, 'a', encoding='utf-8') as f:
                    print("原始的标签：", tags)

                    image_path = get_image_path(imageId, split)
                    image_base64_str = get_image_base64_str(image_path)
                    obj = DescribeImage(image_base64_str)
                    new_txt = obj.get_describe(new_tags)
                    # new_txt = obj.get_describe()

                    imageCNClipJson = {
                        "text_id": text_id,
                        "text": new_txt,
                        "image_ids": [imageId],
                    }

                    save_image_by_desc(imageId, new_txt)

                    json.dump(imageCNClipJson, f, ensure_ascii=False)
                    f.write('\n')


if __name__ == '__main__':
    # srcDir = r'E:\playground\ai\datasets\bdd100k\bdd100k_labels\bdd100k\labels\100k\train'
    # dstDir = r'E:\playground\ai\datasets\bdd100kLabelsToCNClipTag'
    # split = 'train'
    # main(srcDir, dstDir, split)

    split_map = [
        {
            "base_name": "train",
            "new_name": "train",
        },
        {
            "base_name": "val",
            "new_name": "valid",
        },
        {
            "base_name": "test",
            "new_name": "test",
        }
    ]

    for item in split_map:
        base_dir = r'E:\playground\ai\datasets\bdd100k\bdd100k_labels\bdd100k\labels\100k'
        src_dir = os.path.join(base_dir, item['base_name'])
        dst_dir = r'E:\playground\ai\datasets\bdd100kLabelsToCNClipAIDesc'

        print(src_dir, dst_dir, item['new_name'])

        main(src_dir, dst_dir, item['new_name'])
