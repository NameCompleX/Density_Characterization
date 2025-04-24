import numpy as np
import cv2
import os

# 文件路径
IMAGE_DIR = "F:/B208/High_density/ML_generation/image_train"
TXT_DIR = "F:/B208/High_density/ML_generation/txt_train"
WIDTH = 4
LENGTH = 4
IMAGE_SIZE = 512

# 确保输出目录存在
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)


import numpy as np
import cv2

def generate_image(img_size, WIDTH):
    image = np.zeros(img_size, dtype=np.float32)  # 初始化图像
    pixel_info = []  # 存储像素数据

    # 全图的默认线段灰度范围
    grayscale_range = (120, 200)

    # 计算块数：每个块的宽度为 WIDTH，沿着 x 轴切割
    num_blocks = img_size[1] // WIDTH  # 根据图像宽度和块宽度计算切割块数

    line_count = 0  # 记录总共生成的线条数量

    for block in range(num_blocks):
        # 计算每个块的左边界和右边界
        x_start_block = block * WIDTH
        x_end_block = (block + 1) * WIDTH

        # 每块内随机生成的线条数量
        num_lines_in_block = np.random.randint(5 * WIDTH, 100 * WIDTH)
        for _ in range(num_lines_in_block):
            # 随机生成线段的起始位置、长度和强度（根据全图的灰度范围）
            intensity_value = np.random.randint(grayscale_range[0], grayscale_range[1])
            x_start = np.random.randint(x_start_block, x_end_block)  # 线条起始位置在当前块内
            y_start = np.random.randint(-300, 600)  # 线条起始位置的 y 坐标（允许超出图像边界）
            line_length = np.random.randint(50, 201)  # 随机线条长度 100-600
            y_end = y_start + line_length  # 线条结束 y 坐标，允许超出图像边界

            # 画线（将线条绘制在图像中，确保不超出边界）
            for y in range(y_start, y_end + 1):
                if 0 <= y < img_size[0]:  # 确保像素数据在图像范围内
                    image[y, x_start] += intensity_value  # 将灰度值叠加到图像上

            # 记录像素数据
            for y in range(y_start, y_end + 1):
                if 0 <= y < img_size[0]:  # 确保像素数据在图像范围内
                    pixel_info.append([x_start, y, int(intensity_value)])

    # 归一化处理：将像素值归一化到 [120, 200] 之间
    # 1. 将灰度值归一化到 [0, 1] 之间
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

    max = 200
    min = 50
    # 2. 映射到 [15, 200] 区间
    image_normalized = image_normalized * (max - min) + min

    # 3. 将大于 200 的值设置为 255，确保不超过 255
    image_normalized = np.clip(image_normalized, min, max)

    return image_normalized.astype(np.uint8), pixel_info



def save_image_and_data(img, pixel_data, image_path, txt_path, block_size=(WIDTH, LENGTH)):
    # 保存图像
    cv2.imwrite(image_path, img)

    # 将图像划分为块
    height, width = img.shape
    block_data = {}  # 用于存储每个块的密度信息，格式： {(x_start, y_start): (x_end, y_end, 密度)}

    # 计算每个块的线段密度
    for (x_start, y_start, intensity) in pixel_data:
        if intensity > 0:  # 只考虑灰度值大于30的线段
            
            x_block_start = (x_start // block_size[0]) * block_size[0]
            y_block_start = (y_start // block_size[1]) * block_size[1]
            x_block_end = x_block_start + block_size[0] - 1
            y_block_end = y_block_start + block_size[1] - 1

            # 计算该块的坐标范围
            block_key = (x_block_start, y_block_start)

            # 初始化该块的计数
            if block_key not in block_data:
                block_data[block_key] = {"count": 0, "line_count": 0}

            # 更新该块的线段数量
            block_data[block_key]["count"] += 1
            block_data[block_key]["line_count"] += 1

    # 保存txt文件
    with open(txt_path, 'w') as f:
        for (x_start, y_start), data in block_data.items():
            x_end = x_start + block_size[0] 
            y_end = y_start + block_size[1] 

            # 计算该块的线段密度
            line_count = data["count"]
            pixel_count = block_size[0] # 每个块有固定的像素数
            density = line_count / pixel_count  # 密度计算

            # 保存坐标范围和密度
            f.write(f"{x_start} {x_end} {y_start} {y_end} {density}\n")  # 保存坐标范围和密度

    print(f"Data saved to {txt_path}")

def generate_images_in_batches(total_images, img_size=(IMAGE_SIZE, IMAGE_SIZE)):
    for idx in range(total_images):
        image, pixel_data = generate_image(img_size, WIDTH)

        image_path = os.path.join(IMAGE_DIR, f"img{idx+1}.png")
        txt_path = os.path.join(TXT_DIR, f"img{idx+1}.txt") 
        save_image_and_data(image, pixel_data, image_path, txt_path)

        print(f"Image {idx + 1}/{total_images} saved.")

def main():
    total_images = 1   # 生成的图像数量

    print(f"Generating {total_images} images...")
    generate_images_in_batches(total_images)
    print("Image generation completed.")

if __name__ == "__main__":
    main()
