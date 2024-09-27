import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_image(image_labels_file_path, image_file_path, draw_image_path):
    # 示例JSON数据
    with open(image_labels_file_path, 'r') as f:
        json_data = json.load(f)

    # 解析JSON数据
    frames = json_data['frames']

    # 加载图片
    image = mpimg.imread(image_file_path)

    # 创建图形
    fig, ax = plt.subplots()

    # 显示图片
    ax.imshow(image)

    # 遍历每个帧
    for frame in frames:
        objects = frame['objects']

        # 遍历每个对象
        for obj in objects:
            if 'box2d' in obj:
                box2d = obj['box2d']
                x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']

                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # 添加类别标签
                ax.text(x1, y1, obj['category'], color='r', fontsize=8)

    # 设置图形属性
    ax.set_aspect('equal')

    # 保存图像到文件
    plt.savefig(draw_image_path)

    # 显示图形
    plt.show()



if __name__ == "__main__":
    image_labels_file_path = r"E:\playground\ai\datasets\0000f77c-6257be58.json"
    image_file_path = r"E:\playground\ai\datasets\bdd100k\bdd100k_images\bdd100k\images\100k\train\0000f77c-6257be58.jpg"
    draw_image_path = 'output.png'
    draw_image(image_labels_file_path, image_file_path, draw_image_path)