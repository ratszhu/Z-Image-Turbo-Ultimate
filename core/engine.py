# -*- coding: utf-8 -*-
"""
推理引擎
负责模型的生命周期管理：加载、显存优化配置、以及最终的生成执行。
"""
import torch
from diffusers import DiffusionPipeline # type: ignore
import gc
from core.utils import detect_device, get_torch_dtype
from core.lora_manager import LoRAMerger
import config

class ZImageEngine:
    def __init__(self):
        self.pipe = None
        self.device = None
        self.dtype = None
        self.lora_merger = None
        self.current_lora_applied = False

    def load_model(self):
        """
        加载模型 (自动检测设备)。
        """
        # 1. 自动检测设备
        self.device = detect_device()
        self.dtype = get_torch_dtype(self.device)
        
        print(f"🚀 [Engine] 正在加载模型... 自动检测设备: {self.device.upper()}, 精度: {self.dtype}")
        
        # 2. 清理旧显存 (防止重载时爆内存)
        if self.pipe:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        # 3. 加载 Diffusers Pipeline
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                config.MODEL_PATH,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.pipe.to(self.device)
            
            # 初始化 LoRA 管理器
            self.lora_merger = LoRAMerger(self.pipe)
            self.current_lora_applied = False
            
            # 4. 应用硬件特定的优化策略
            self._apply_optimizations()
            
            print("✅ [Engine] 模型加载完毕，准备就绪。")
            return f"就绪 | 设备: {self.device.upper()} | 精度: {self.dtype}"
            
        except Exception as e:
            print(f"❌ [Engine] 加载失败: {e}")
            return f"加载失败: {e}"

    def _apply_optimizations(self):
        """根据硬件类型应用显存和画质优化"""
        # [通用] VAE 精度修复: 强制 FP32 以解决模糊问题
        if hasattr(self.pipe, "vae"):
            self.pipe.vae.to(dtype=torch.float32) # type: ignore
            self.pipe.vae.config.force_upcast = True # type: ignore
            print("👁️ [Optim] VAE 已切换至 FP32 (画质锐化)")

        # [Mac] M1/M2/M3 优化
        if self.device == "mps":
            # 关闭 Tiling 以获得最佳清晰度 (M1 Max 显存足够)
            # 如果是 16G 内存的 Mac，可能需要开启 self.pipe.enable_vae_tiling()
            print("🧠 [Optim] MPS 模式: 已配置 Bfloat16 + VAE FP32。")
        
        # [Windows] NVIDIA 优化
        elif self.device == "cuda":
            # 开启 CPU Offload 以节省显存 (这对 8G 显存的 4070 很重要)
            self.pipe.enable_model_cpu_offload() # type: ignore
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling() # type: ignore
            print("🧠 [Optim] CUDA 模式: CPU Offload 已开启。")

    def update_lora(self, enable, scale):
        """更新 LoRA 状态 (启用/禁用/调整强度)"""
        # 情况A: 从无到有 -> 直接加载
        if enable and not self.current_lora_applied:
            self.lora_merger.load_lora_weights(config.LORA_PATH, scale) # type: ignore
            self.current_lora_applied = True
            return "LoRA 已启用"
            
        # 情况B: 需要卸载或改变参数 -> 重载模型 (最稳妥的方式)
        # 因为手动注入修改了权重，为了画质纯净，我们选择重置模型
        if (not enable and self.current_lora_applied) or (enable and self.current_lora_applied):
            print("🔄 [Engine] LoRA 设置变更，正在重置模型...")
            self.load_model() # 重载
            if enable:
                self.lora_merger.load_lora_weights(config.LORA_PATH, scale) # type: ignore
                self.current_lora_applied = True
            return "模型已重置并应用新 LoRA 设置"

    def generate(self, prompt, neg_prompt, steps, cfg, width, height, seed, seed_mode):
        """生成图片的核心逻辑"""
        # 显存清理
        gc.collect()
        if self.device == "mps": torch.mps.empty_cache()
        if self.device == "cuda": torch.cuda.empty_cache()

        # 种子处理逻辑
        if seed_mode == "随机" or seed == -1:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            actual_seed = int(seed)
            
        # 创建 Generator (MPS 需要在 CPU 初始化)
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(gen_device).manual_seed(actual_seed) # type: ignore

        print(f"🎨 [Generate] 尺寸: {width}x{height} | 步数: {steps} | 种子: {actual_seed}")

        try:
            image = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator
            ).images[0] # type: ignore
            
            return image, f"Used Seed: {actual_seed}"
        except Exception as e:
            return None, f"Error: {str(e)}"