import lmdb
import base64
from io import BytesIO
from PIL import Image

image_ids = [2446244213, 3856130327, 2765787222, 3535056297, 4848497007, 23775702, 111285749, 2422233191, 1947351225, 4441983387]

lmdb_imgs = r"E:\playground\ai\datasets\Flickr30k-CN\lmdb\valid\imgs"
env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
txn_imgs = env_imgs.begin(buffers=True)
for image_id in image_ids:
    image_b64 = txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
    img = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))
    img.show()