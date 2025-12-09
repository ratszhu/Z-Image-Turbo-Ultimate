# -*- coding: utf-8 -*-
"""
全局配置文件
用于统一管理模型路径、默认参数及系统常量。
"""
import os

# --- 路径配置 ---
# 基础模型路径 (请确保该路径下包含完整的 diffusers 模型文件)
MODEL_PATH = "./Z-Image-Model"

# LoRA 文件路径 (请修改为您实际的文件名)
LORA_PATH = "./Technically_Color_Z_Image_Turbo_v1_renderartist_2000.safetensors"

# --- 默认生成参数 ---
DEFAULT_PROMPT = "A cinematic shot of a cyberpunk city, neon lights, rain, high detail, 8k"
DEFAULT_NEGATIVE_PROMPT = "cartoon, painting, 3d render, low poly, blurry, low quality, distorted, ugly, watermark"

# 尺寸与步数
DEFAULT_STEPS = 9      # Turbo 模型推荐步数
DEFAULT_CFG = 0.0      # Turbo 模型推荐 CFG 为 0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED = -1      # -1 代表随机种子

# LoRA 默认设置
DEFAULT_LORA_SCALE = 1.3
DEFAULT_LORA_ENABLE = True

# --- 系统配置 ---
# 代理设置 (解决 Gradio 本地连接报 502 错误的问题)
os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'