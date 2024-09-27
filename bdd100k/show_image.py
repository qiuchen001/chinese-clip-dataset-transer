import lmdb
import base64
from io import BytesIO
from PIL import Image
import chardet

# 666622
# 783255
image_ids = [666622]

lmdb_imgs = r"E:\playground\ai\datasets\MUGE\lmdb\valid\imgs"
# env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
txn_imgs = env_imgs.begin(buffers=True)
for image_id in image_ids:
    image_b64 = txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
    img.show()


def show_text():
    txt_ids = [1, 2]

    lmdb_txt = r"E:\playground\ai\datasets\bdd100kLabelsToCNClipTag\lmdb\valid\\pairs"
    env_txt = lmdb.open(lmdb_txt, readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_imgs = env_txt.begin(buffers=True)
    for txt_id in txt_ids:
        # 将 txt_id 转换为字符串
        txt_id_str = str(txt_id).encode('ascii')

        # 获取对应 txt_id 的数据
        txt_data = txn_imgs.get(txt_id_str)

        # 解码数据并打印
        if txt_data is not None:
            # 将 memoryview 对象转换为 bytes 对象
            txt_data = bytes(txt_data)

            # 检测数据的编码
            result = chardet.detect(txt_data)
            encoding = result['encoding']

            # 如果检测到的编码为 None，使用默认编码 utf-8
            if encoding is None:
                encoding = 'utf-8'

            # 使用检测到的编码解码数据
            try:
                txt_content = txt_data.decode(encoding)
                print(f"Text ID: {txt_id}")
                print(f"Content: {txt_content}")
            except UnicodeDecodeError:
                print(f"Text ID: {txt_id} - Unable to decode content with encoding {encoding}")
        else:
            print(f"Text ID: {txt_id} not found")


# show_text()


def show_all_texts(lmdb_path):
    # 打开 LMDB 数据库
    env_txt = lmdb.open(lmdb_path, readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_imgs = env_txt.begin(buffers=True)

    # 获取数据库中的所有键
    with txn_imgs.cursor() as cursor:
        for key, value in cursor:
            # 将 memoryview 对象转换为 bytes 对象
            value = bytes(value)

            # 检测数据的编码
            result = chardet.detect(value)
            encoding = result['encoding']

            # 如果检测到的编码为 None，使用默认编码 utf-8
            if encoding is None:
                encoding = 'utf-8'

            # 使用检测到的编码解码数据
            try:
                txt_content = value.decode(encoding)
                print(f"Key: {key.decode('ascii')}")
                print(f"Content: {txt_content}")
            except UnicodeDecodeError:
                print(f"Key: {key.decode('ascii')} - Unable to decode content with encoding {encoding}")

# # 调用函数
# lmdb_txt = r"E:\playground\ai\datasets\Flickr30k-CN\lmdb\valid\\pairs"
# show_all_texts(lmdb_txt)