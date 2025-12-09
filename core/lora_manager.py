'''
Descripttion: 
Author: 朱东帅
Date: 2025-12-08 13:45:44
LastEditors: 朱东帅
LastEditTime: 2025-12-08 14:01:12
'''
# -*- coding: utf-8 -*-
"""
LoRA 管理器
专门处理 Z-Image 等非标准结构模型的 LoRA 权重注入。
由于 Z-Image 包含特殊的 Refiner 层，普通的 load_lora_weights 无法生效，
必须使用此类进行层对层的精准注入。
"""
import torch
import safetensors.torch
import re
import os

class LoRAMerger:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.loaded_path = None

    def load_lora_weights(self, lora_path: str, lora_scale: float = 1.0):
        """加载并合并 LoRA 权重"""
        if not os.path.exists(lora_path):
            print(f"⚠️ [LoRA Manager] 文件未找到: {lora_path}，跳过加载。")
            return

        print(f"🎨 [LoRA Manager] 正在注入 LoRA: {os.path.basename(lora_path)} (强度: {lora_scale})...")
        try:
            tensors = safetensors.torch.load_file(lora_path)
            self._merge_lora_weights(tensors, lora_scale)
            self.loaded_path = lora_path
        except Exception as e:
            print(f"❌ [LoRA Manager] 注入失败: {e}")

    def _get_module_from_path(self, module_path):
        """工具函数：通过字符串路径获取模型中的层对象"""
        try:
            parts = module_path.split('.')
            current = self.pipeline
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current
        except AttributeError:
            return None

    def _get_module_path_from_lora_key(self, lora_key):
        """核心逻辑：将 LoRA 的键名映射到 Z-Image 模型的实际层路径"""
        key = lora_key.replace('diffusion_model.', '')
        
        # 1. 匹配 Refiner 层 (这是 Z-Image 画质好的关键)
        context_match = re.match(r'context_refiner\.(\d+)\.attention\.(to_q|to_k|to_v|to_out\.0)', key)
        if context_match:
            layer_idx, target = context_match.groups()
            return f"transformer.context_refiner.{layer_idx}.attention.{target}"

        noise_match = re.match(r'noise_refiner\.(\d+)\.attention\.(to_q|to_k|to_v|to_out\.0)', key)
        if noise_match:
            layer_idx, target = noise_match.groups()
            return f"transformer.noise_refiner.{layer_idx}.attention.{target}"

        # 2. 匹配常规 Transformer 层
        if key.startswith('layers.'):
            return f"transformer.{key}"
            
        return None

    def _merge_lora_weights(self, lora_state_dict, lora_scale):
        """执行权重合并：W_new = W_old + (B @ A) * scale"""
        count = 0
        device = self.pipeline.device
        
        for key in lora_state_dict.keys():
            if ".lora_A.weight" in key:
                base_key = key.replace(".lora_A.weight", "")
                b_key = f"{base_key}.lora_B.weight"
                alpha_key = f"{base_key}.alpha"
                
                module_path = self._get_module_path_from_lora_key(key)
                if not module_path: continue
                
                module = self._get_module_from_path(module_path)
                if module is None or not hasattr(module, 'weight'): continue

                # 临时提升到 FP32 进行计算以保证精度
                A = lora_state_dict[key].to(dtype=torch.float32, device=device)
                B = lora_state_dict[b_key].to(dtype=torch.float32, device=device)
                
                rank = A.shape[0]
                alpha = lora_state_dict[alpha_key].item() if alpha_key in lora_state_dict else rank
                scale = lora_scale * (alpha / rank)
                
                # 计算增量并注入
                delta = (B @ A) * scale
                
                with torch.no_grad():
                    module.weight.data += delta.to(module.weight.dtype)
                    count += 1
                    
        print(f"✅ [LoRA Manager] 注入完成，共修改 {count} 层权重。")