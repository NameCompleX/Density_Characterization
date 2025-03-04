import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import os
import numpy as np
from PIL import Image

# --------------------
# 配置参数
# --------------------
config = {
    "scale_factor": 10,           # 超分辨率放大倍数（2/3/4）
    "num_channels": 3,           # 图像通道数（RGB=3，灰度=1）
    "batch_size": 16,            # 训练批大小
    "epochs": 100,               # 训练轮数
    "lr_size": 64,               # 低分辨率图像块大小（hr_size = lr_size * scale_factor）
    "train_data_dir": "D:/cey/CNTs_detials/ML_generation/image_train",  # 替换为您的训练集路径
    "val_data_dir": "D:/cey/CNTs_detials/ML_generation/image_val",      # 替换为您的验证集路径
    "save_dir": "D:/cey/CNTs_detials/Model"   # 模型保存路径
}

# 创建保存目录
os.makedirs(config["save_dir"], exist_ok=True)

# --------------------
# 自定义 PSNR 指标
# --------------------
class PSNR(tf.keras.metrics.Metric):
    def __init__(self, max_val=1.0, name="psnr", **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr = self.add_weight(name="psnr", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 计算 PSNR
        psnr_value = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        # 更新状态
        self.psnr.assign_add(tf.reduce_sum(psnr_value))
        self.count.assign_add(tf.cast(tf.size(psnr_value), tf.float32))

    def result(self):
        # 返回平均 PSNR
        return self.psnr / self.count

    def reset_state(self):
        # 重置状态
        self.psnr.assign(0.0)
        self.count.assign(0.0)

# --------------------
# 数据加载与预处理
# --------------------
def load_and_preprocess(lr_path, hr_path):
    # 读取图像
    lr_img = tf.io.read_file(lr_path)
    lr_img = tf.image.decode_image(lr_img, channels=config["num_channels"], expand_animations=False)
    hr_img = tf.io.read_file(hr_path)
    hr_img = tf.image.decode_image(hr_img, channels=config["num_channels"], expand_animations=False)
    
    # 归一化到 [-1, 1]
    lr_img = (tf.cast(lr_img, tf.float32) / 127.5) - 1.0
    hr_img = (tf.cast(hr_img, tf.float32) / 127.5) - 1.0
    
    # 调整低分辨率图像的尺寸
    hr_shape = tf.shape(hr_img)
    expected_lr_shape = (hr_shape[0] // config["scale_factor"], 
                         hr_shape[1] // config["scale_factor"])
    lr_img = tf.image.resize(lr_img, expected_lr_shape)
    
    return lr_img, hr_img

def build_dataset(data_dir):
    # 获取所有低/高分辨率图像路径对
    lr_dir = os.path.join(data_dir, "lr")
    hr_dir = os.path.join(data_dir, "hr")
    lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
    hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 随机裁剪为固定块
    def random_crop(lr, hr):
        lr_crop = tf.image.random_crop(lr, [config["lr_size"], config["lr_size"], config["num_channels"]])
        hr_crop = tf.image.random_crop(hr, [config["lr_size"] * config["scale_factor"], 
                                       config["lr_size"] * config["scale_factor"], 
                                       config["num_channels"]])
        return lr_crop, hr_crop
    
    dataset = dataset.map(random_crop, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 数据增强（可选）：翻转、旋转
    dataset = dataset.map(
        lambda lr, hr: (tf.image.random_flip_left_right(lr),
                        tf.image.random_flip_left_right(hr)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        lambda lr, hr: (tf.image.random_flip_up_down(lr),
                        tf.image.random_flip_up_down(hr)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 批处理与优化
    dataset = dataset.batch(config["batch_size"])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 构建训练集和验证集
train_dataset = build_dataset(config["train_data_dir"])
val_dataset = build_dataset(config["val_data_dir"])

# --------------------
# 定义 ESPCN 模型
# --------------------
class ESPCN(Model):
    def __init__(self, scale_factor=2, num_channels=3):
        super(ESPCN, self).__init__()
        self.scale = scale_factor
        
        # 特征提取层
        self.conv1 = layers.Conv2D(64, 5, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        
        # 亚像素卷积层
        self.conv3 = layers.Conv2D(num_channels * (scale_factor ** 2), 3, padding='same')
        self.pixel_shuffle = lambda x: tf.nn.depth_to_space(x, block_size=scale_factor)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

# 初始化模型
model = ESPCN(scale_factor=config["scale_factor"], num_channels=config["num_channels"])

# --------------------
# 训练配置
# --------------------
# 自定义学习率衰减
def lr_schedule(epoch):
    initial_lr = 1e-3
    decay_factor = 0.5
    decay_epochs = 30
    return initial_lr * (decay_factor ** (epoch // decay_epochs))

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',  # 均方误差
    metrics=[PSNR(max_val=2.0)]  # 使用自定义 PSNR 指标
)

# 回调函数
callbacks = [
    callbacks.ModelCheckpoint(
        filepath=os.path.join(config["save_dir"], 'espcn_best.h5'),
        save_best_only=True,
        monitor='val_psnr',  # 根据验证集 PSNR 保存最佳模型
        mode='max'
    ),
    callbacks.LearningRateScheduler(lr_schedule),
    callbacks.TensorBoard(log_dir='logs')
]

# --------------------
# 训练模型
# --------------------
history = model.fit(
    train_dataset,
    epochs=config["epochs"],
    validation_data=val_dataset,
    callbacks=callbacks
)

# --------------------
# 推理代码（示例）
# --------------------
def predict_single_image(model, lr_image_path, output_path):
    # 加载低分辨率图像
    lr_img = Image.open(lr_image_path)
    lr_img = np.array(lr_img) / 127.5 - 1.0  # 归一化到 [-1, 1]
    lr_tensor = tf.expand_dims(lr_img, axis=0)
    
    # 推理
    sr_tensor = model.predict(lr_tensor)
    sr_img = (sr_tensor[0] + 1.0) * 127.5  # 反归一化
    sr_img = np.clip(sr_img, 0, 255).astype('uint8')
    
    # 保存结果
    Image.fromarray(sr_img).save(output_path)
    print(f"超分辨率结果已保存至 {output_path}")

# 使用示例
# predict_single_image(model, "test_lr.png", "test_sr.png")