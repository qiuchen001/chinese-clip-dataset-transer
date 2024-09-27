import json
import base64
from io import BytesIO
from PIL import Image
import os

jsonl_src_path = r"E:\playground\ai\datasets\Flickr30k-CN\train_texts.jsonl"
jsonl_dst_path = r"E:\playground\ai\datasets\custom-cn-clip-datasets\train_texts.jsonl"

tsv_src_path = r"E:\playground\ai\datasets\Flickr30k-CN\train_imgs.tsv"
tsv_dst_path = r"E:\playground\ai\datasets\custom-cn-clip-datasets\train_imgs.tsv"


def save_image(image_b64_str, image_name):
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64_str)))
    # img.show()

    dst_dir = r"E:\playground\ai\datasets\custom-cn-clip-datasets\images"

    dst_image_path = os.path.join(dst_dir, f"{image_name}.png")
    img.save(dst_image_path)


def read_tsv(file_path, seek_id, text):
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            pair_list = line.split('\t')
            if int(pair_list[0]) == seek_id:
                save_image(pair_list[1], text)

                write_tsv(tsv_dst_path, line)


def write_tsv(file_path, data):
    with open(file_path, "a") as file:
        file.write(data)


def read_jsonl(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            text = json_obj['text']
            if len(text) > 50:
                continue

            if "人行横道" in text:
            # if "交通灯" in text or "汽车" in text:
            # if "公交车" in text or "行人" in text or "路口" in text or "自行车" in text:
                print(text)
                write_jsonl(jsonl_dst_path, line)

                image_ids = json_obj['image_ids']
                for item in image_ids:
                    read_tsv(tsv_src_path, item, text)


def write_jsonl(file_path, data):
    with open(file_path, "a", encoding='utf-8') as file:
        file.write(data)


read_jsonl(jsonl_src_path)
