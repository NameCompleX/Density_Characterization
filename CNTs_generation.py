import numpy as np
import cv2
import os
from multiprocessing import Pool
import time

# --------------------------
# 配置参数 (按需修改)
# --------------------------
IMAGE_DIR = "D:/cey/CNTs_detials/ML_generation/image_train/lable"  # 输出目录
WIDTH = 4                 # 横向分块宽度
IMAGE_SIZE = 640         # 图像尺寸 (建议测试时调至 512 或 1024)
TOTAL_IMAGES = 200000       # 生成图像总数
START_INDEX = 1          # 文件名起始编号
NUM_PROCESSES = 16         # 进程数 (建议设为CPU核心数的50-75%)
USE_GPU = False           # 是否启用GPU加速 (需安装CuPy)

# --------------------------
# GPU加速依赖 (可选)
# --------------------------
if USE_GPU:
    try:
        import cupy as cp
        print("[INFO] GPU加速已启用")
    except ImportError:
        print("[WARN] CuPy未安装，自动回退到CPU模式")
        USE_GPU = False

# 确保输出目录存在
os.makedirs(IMAGE_DIR, exist_ok=True)

# --------------------------
# 核心图像生成函数
# --------------------------
def generate_image(img_size, width):
    # 选择使用GPU数组还是CPU数组
    xp = cp if USE_GPU else np
    
    # 初始化图像
    image = xp.zeros(img_size, dtype=xp.float32)
    grayscale_range = (120, 200)
    num_blocks = img_size[1] // width
    
    for block in range(num_blocks):
        x_start_block = block * width
        x_end_block = (block + 1) * width
        
        # 批量生成线段参数
        num_lines = np.random.randint(5*width, 50*width)
        x_starts = np.random.randint(x_start_block, x_end_block, num_lines)
        y_starts = np.random.randint(-1000, 2000, num_lines)
        lengths = np.random.randint(50, 201, num_lines)
        intensities = np.random.randint(*grayscale_range, num_lines)
        
        # 向量化处理有效区域
        valid_mask = (y_starts >= 0) & (y_starts + lengths < img_size[0])
        y_starts = y_starts[valid_mask]
        lengths = lengths[valid_mask]
        x_starts = x_starts[valid_mask]
        intensities = intensities[valid_mask]
        
        # 批量更新图像
        for i in range(len(y_starts)):
            y_end = y_starts[i] + lengths[i]
            image[y_starts[i]:y_end, x_starts[i]] += intensities[i]
    
    # 归一化处理
    image = (image - image.min()) / (image.max() - image.min()) * 150 + 50
    image = xp.clip(image, 50, 200)
    
    # 返回CPU数组 (如果使用GPU)
    return image.get().astype(np.uint8) if USE_GPU else image.astype(np.uint8)

# --------------------------
# 单张图像保存函数
# --------------------------
def save_single_image(args):
    idx, img_size, width = args
    try:
        image = generate_image(img_size, width)
        filename = f"img{idx + START_INDEX:04d}.png"
        cv2.imwrite(os.path.join(IMAGE_DIR, filename), image)
        print(f"已保存: {filename}")
    except Exception as e:
        print(f"生成失败 [{idx}]: {str(e)}")

# --------------------------
# 多进程控制器
# --------------------------
def generate_images_parallel():
    img_size = (IMAGE_SIZE, IMAGE_SIZE)
    task_args = [(i, img_size, WIDTH) for i in range(TOTAL_IMAGES)]
    
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(save_single_image, task_args, chunksize=10)

# --------------------------
# 主函数
# --------------------------
if __name__ == "__main__":
    start_time = time.time()
    print(f"开始生成 {TOTAL_IMAGES} 张图像...")
    generate_images_parallel()
    elapsed = time.time() - start_time
    print(f"生成完成! 总耗时: {elapsed/60:.2f} 分钟")