# -*- coding: utf-8 -*-
"""
Z-Image-Turbo WebUI
主程序入口，负责构建 Gradio 界面。
"""
import gradio as gr
from core.engine import ZImageEngine
import config

# 初始化引擎
engine = ZImageEngine()

# --- 事件回调函数 ---

def initialize_app():
    """
    APP 启动时自动调用：
    1. 自动检测设备并加载模型
    2. 加载默认 LoRA
    """
    status_msg = engine.load_model() # 自动检测
    engine.update_lora(config.DEFAULT_LORA_ENABLE, config.DEFAULT_LORA_SCALE)
    
    # 返回状态信息给底部栏，返回设备信息给设置栏
    device_info_str = f"自动检测: {engine.device.upper()} (精度: {str(engine.dtype).split('.')[-1]})" # type: ignore
    return status_msg, device_info_str

def on_generate_click(
    prompt, neg_prompt, 
    steps, cfg, 
    width, height, 
    seed_mode, seed_val,
    lora_enable, lora_scale,
    output_format
):
    """点击生成按钮"""
    # 1. 检查 LoRA 变更 (可能会触发重载)
    if lora_enable != engine.current_lora_applied: 
        engine.update_lora(lora_enable, lora_scale)
        
    # 2. 生成图片
    image, info = engine.generate(
        prompt, neg_prompt, steps, cfg, width, height, seed_val, seed_mode
    )
    
    # 3. 返回图片 (Gradio 会根据界面组件的 format 配置自动处理格式)
    return image, info

# --- 界面构建 ---

# 移除 theme 参数以确保兼容性
with gr.Blocks(title="Z-Image-Turbo-Ultimate") as demo:
    gr.Markdown("# 🚀 Z-Image-Turbo-Ultimate")
    
    with gr.Row():
        # === 左侧控制面板 ===
        with gr.Column(scale=4):
            # 1. 提示词
            prompt_input = gr.Textbox(
                label="正面提示词 (Prompt)", 
                value=config.DEFAULT_PROMPT, 
                lines=3
            )
            neg_prompt_input = gr.Textbox(
                label="负面提示词 (Negative Prompt)", 
                value=config.DEFAULT_NEGATIVE_PROMPT, 
                lines=2
            )
            
            # 2. 生成参数
            with gr.Accordion("⚙️ 生成参数设置", open=True):
                with gr.Row():
                    steps = gr.Slider(1, 50, value=config.DEFAULT_STEPS, step=1, label="迭代步数 (Steps)")
                    cfg = gr.Slider(0.0, 10.0, value=config.DEFAULT_CFG, step=0.1, label="引导系数 (CFG)")
                
                with gr.Row():
                    width = gr.Slider(512, 2048, value=config.DEFAULT_WIDTH, step=64, label="宽度 (Width)")
                    height = gr.Slider(512, 2048, value=config.DEFAULT_HEIGHT, step=64, label="高度 (Height)")
                
                with gr.Row():
                    seed_mode = gr.Radio(["随机", "固定"], value="固定", label="种子模式")
                    # 只有选固定时，数字框才生效(逻辑在下面绑定)
                    seed_val = gr.Number(label="种子数值", value=12345, precision=0)
            
            # 3. 风格 LoRA
            with gr.Accordion("🎨 风格/LoRA 设置", open=True):
                with gr.Row():
                    lora_enable = gr.Checkbox(label="启用色彩增强 LoRA", value=config.DEFAULT_LORA_ENABLE)
                    lora_scale = gr.Slider(0.0, 2.0, value=config.DEFAULT_LORA_SCALE, label="LoRA 强度")

            # 4. 硬件与输出 (改为只读显示)
            with gr.Accordion("🖥️ 硬件与输出设置", open=False):
                with gr.Row():
                    # [修改] 改为 Textbox 显示，用户不可交互
                    device_display = gr.Textbox(
                        label="当前推理设备 (自动托管)", 
                        value="检测中...", 
                        interactive=False,
                        scale=2
                    )
                    format_select = gr.Dropdown(
                        ["png", "jpeg", "webp"], 
                        value="png", 
                        label="图片输出格式",
                        scale=1
                    )

            run_btn = gr.Button("✨ 开始生成 (Generate)", variant="primary", size="lg")

        # === 右侧结果面板 ===
        with gr.Column(scale=5):
            # format 参数决定了右键保存时的格式，WebP虽然快但有损，这里默认PNG无损预览
            output_img = gr.Image(label="生成结果", type="pil", format="png")
            status_info = gr.Textbox(label="运行状态", interactive=False)

    # === 交互逻辑绑定 ===
    
    # 1. 启动时自动初始化 (加载模型 -> 更新UI显示)
    demo.load(initialize_app, inputs=None, outputs=[status_info, device_display])
    
    # 2. 种子模式切换逻辑
    def update_seed_interactive(mode):
        # 如果是随机，禁用输入框；如果是固定，启用
        return gr.Number(interactive=(mode=="固定"))
    
    seed_mode.change(update_seed_interactive, inputs=[seed_mode], outputs=[seed_val])

    # 3. 生成按钮
    run_btn.click(
        fn=on_generate_click,
        inputs=[
            prompt_input, neg_prompt_input,
            steps, cfg,
            width, height,
            seed_mode, seed_val,
            lora_enable, lora_scale,
            format_select
        ],
        outputs=[output_img, status_info]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1")