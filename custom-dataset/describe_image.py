from IPython.display import display, Image, Audio, clear_output

from openai import OpenAI
import os
import requests
from PIL import Image
from io import BytesIO
import base64

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_image_base64_str(file_name):
    img = Image.open(file_name)  # 访问图片路径
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


class DescribeImage:
    def __init__(self, image_base64_str):
        self.prompt = """
        描述这张图片，字数限制在50以内。
        你需要特别关注以下关键词，如果图片中有命中某些关键词，就需要在描述中体现出这些关键词：
        汽车
        公交车
        行人
        自行车
        卡车
        摩托车
        火车
        骑手
        交通标志
        绿色交通标志
        红色交通标志
        黄色交通标志
        未知的交通标志
        交通灯
        绿色交通灯
        红色交通灯
        黄色交通灯
        未知的交通灯
        人行横道
        路缘石
        黄色的单车道线
        白色的单车道线
        其他的单车道
        黄色的双车道线
        白色的双车道线
        其他的双车道线
        可行驶区域
        替代区域
        未知区域
        天气
        阴天
        晴朗
        未知的天气
        下雪
        多云
        有雨
        有雾
        场景
        城市街道
        高速公路
        住宅区
        未定义的场景
        停车场
        隧道
        加油站
        时间
        夜晚
        白天
        黎明或黄昏
        未定义的时间
        """
        self.image_base64_str = image_base64_str

    def get_describe(self):
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [self.prompt, {
                    "image": self.image_base64_str
                }],
            },
        ]
        params = {
            "model": "gpt-4o",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 200,
        }

        result = client.chat.completions.create(**params)
        print(result.choices[0].message.content)

        return result.choices[0].message.content


if __name__ == '__main__':
    image_path = r"E:\playground\ai\datasets\custom-cn-clip-datasets\images\白天，城市街道，两位行人正经过人行横道。.png"
    image_base64_str = get_image_base64_str(image_path)
    obj = DescribeImage(image_base64_str)
    obj.get_describe()
