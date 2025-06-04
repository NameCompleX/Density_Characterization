# 基于SEM的碳纳米管阵列密度表征

本项目提供了一个完整的流程，用于图像数据的生成、处理和基于机器学习的密度表征任务。涵盖了数据生成、模型训练与测试、灰度提取与灰度-密度回归等步骤。

## 📁 项目结构说明

- **Generate.py**
  用于生成机器学习训练数据。生成过程中会自动保存每张图像中对应区域的**坐标**和**密度值**。
- **MLtrain.ipynb**
  使用生成的数据集对模型进行训练。
- **MLtest.ipynb**
  测试训练好的模型在模拟数据集上的预测效果。
- **extract_gray_degree.ipynb**
  根据生成数据时保存的坐标，提取图像中对应区域的灰度值，并保存为灰度-密度映射文档。
- **density-grayscale.ipynb**
  获取指定文件夹中每张图像的**平均灰度值**，用于后续灰度与密度的映射建模。
- **density-grayscale_regression.ipynb**
  根据灰度与密度的关系，将**真实图像样本**调整为与训练数据风格一致的形式，使其适配模型进行密度预测。

## 🔁 推荐使用流程

1. **数据生成**
   运行 `Generate.py`，生成图像并保存坐标和密度信息。
2. **模型训练**
   在 `MLtrain.ipynb` 中加载并训练模型。
3. **模型测试**
   使用 `MLtest.ipynb` 测试模型在生成数据集上的表现。
4. **灰度提取与分析**

   - 运行 `extract_gray_degree.ipynb` 提取局部区域的灰度值。
   - 使用 `density-grayscale.ipynb` 获取图像的整体灰度-密度趋势。
5. **真实图像预测适配**
   使用 `density-grayscale_regression.ipynb` 将真实图像灰度风格调整为训练数据风格，并进行预测。

## ⚙️ 环境依赖

- `pip install -r requirements.txt`

## 📌 项目说明

该项目旨在探索SEM图像灰度与碳纳米管密度之间的关系，并基于此构建可用于真实场景表征碳纳米管密度的机器学习模型。

# ML Density characterizeion Pipeline

This project provides a complete pipeline for generating, processing, and applying machine learning models for density characterizeion based on SEM images of SWNT arrays. It includes steps for data generation, model training/testing, grayscale extraction, and grayscale-density regression.

## 📁 Project Structure

- **Generate.py**
  Generates synthetic training data and saves the **coordinates** and **density values** of each region in the image.
- **MLtrain.ipynb**
  Trains a machine learning model using the generated dataset.
- **MLtest.ipynb**
  Tests the trained model's performance on the synthetic data.
- **extract_gray_degree.ipynb**
  Extracts grayscale values from specific regions based on the saved coordinates and stores them as grayscale-density mappings.
- **density-grayscale.ipynb**
  Calculates the **average grayscale value** of each image in a given folder for further analysis.
- **density-grayscale_regression.ipynb**
  Adapts real image samples using the grayscale-density relationship to match the synthetic training data format, enabling characterizeion using the trained model.

## 🔁 Recommended Workflow

1. **Data Generation**
   Run `Generate.py` to generate images and save coordinate-density information.
2. **Model Training**
   Use `MLtrain.ipynb` to train the model on the generated dataset.
3. **Model Testing**
   Test the model performance using `MLtest.ipynb`.
4. **Grayscale Extraction & Analysis**

   - Use `extract_gray_degree.ipynb` to extract grayscale values from target regions.
   - Use `density-grayscale.ipynb` to analyze grayscale-density relationships.
5. **Real Image characterizeion**
   Use `density-grayscale_regression.ipynb` to adapt real-world images and run characterizations with the trained model.

## ⚙️ Dependencies

- `pip install -r requirements.txt`

## 📌 Characterization Description

This project explores the relationship between image grayscale and SWNT arrays' density and builds models for density characterization tasks.
