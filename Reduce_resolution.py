import tensorflow as tf
import os
from pathlib import Path

def resize_images(input_dir, output_dir, target_size=(64, 64)):
    """
    将输入目录中的所有640x640图片压缩到指定尺寸并保存到输出目录
    :param input_dir: 输入图片目录路径
    :param output_dir: 输出目录路径
    :param target_size: 目标尺寸，默认为64x64
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # 遍历输入目录
    for img_path in Path(input_dir).glob('*'):
        try:
            # 跳过非图片文件
            if img_path.suffix.lower() not in valid_extensions:
                continue

            # 读取文件
            image_data = tf.io.read_file(str(img_path))
            
            # 解码图像（保留原始通道信息）
            image = tf.image.decode_image(image_data)
            
            # 验证原始尺寸
            if image.shape[0] != 640 or image.shape[1] != 640:
                print(f"跳过 {img_path.name}: 尺寸不是640x640 (实际尺寸 {image.shape[:2]})")
                continue

            # 调整尺寸（使用双线性插值）
            resized_image = tf.image.resize(image, target_size, method='bilinear')
            
            # 转换回uint8类型
            resized_image = tf.cast(resized_image, tf.uint8)
            
            # 根据原始格式编码
            suffix = img_path.suffix.lower()
            if suffix in ['.jpg', '.jpeg']:
                encoded_image = tf.image.encode_jpeg(resized_image)
            elif suffix == '.png':
                encoded_image = tf.image.encode_png(resized_image)
            
            # 保存文件
            output_path = Path(output_dir) / img_path.name
            tf.io.write_file(str(output_path), encoded_image)
            print(f"成功处理: {img_path.name} => {output_path}")

        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_directory = "D:/cey/CNTs_detials/ML_generation/image_train/lable"  # 替换为你的输入目录
    output_directory = "D:/cey/CNTs_detials/ML_generation/image_train/base"  # 替换为你的输出目录
    
    resize_images(input_directory, output_directory)