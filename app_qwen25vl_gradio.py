# app_qwen25vl_gradio.py
import os
import gc
import glob
import torch
from PIL import Image
import gradio as gr
from transformers import AutoModelForVision2Seq, AutoProcessor  # ← 改成 Vision2Seq

# —— 基础环境设置 ——
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = os.path.abspath("./Qwen2.5-VL-7B-Instruct")

# ======= 设备/精度 =======
if torch.cuda.is_available():
    TORCH_DTYPE = torch.float16
    DEVICE_MAP = "auto"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
else:
    # CPU 或其他设备回退
    TORCH_DTYPE = torch.float32
    DEVICE_MAP = "auto"

# ======= 模型加载 =======
def load_model_and_processor():
    model = AutoModelForVision2Seq.from_pretrained(   # ← 改成 Vision2Seq
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True,
        # local_files_only=True,  # 完全离线可开启
        # attn_implementation="flash_attention_2",  # 可选：已装FA2时可开
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        # local_files_only=True,
    )
    return model, processor

MODEL, PROCESSOR = load_model_and_processor()

# ======= 工具函数 =======
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def list_local_images(search_dir="."):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(search_dir, f"*{ext}")))
    files = [os.path.relpath(p, start=search_dir) for p in files]
    files.sort()
    return files

def load_image_from_path(path_str):
    if not path_str:
        return None, "请输入图片本地路径"
    path = os.path.expanduser(path_str)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        return None, f"文件不存在：{path}"
    try:
        im = Image.open(path).convert("RGB")
        return im, f"已加载：{path}"
    except Exception as e:
        return None, f"加载失败：{e}"

def load_image_from_dropdown(name):
    if not name:
        return None, "请先从下拉框选择图片"
    path = os.path.abspath(name)
    return load_image_from_path(path)

# ======= 推理（支持多轮） =======
def generate_reply(image: Image.Image, user_text: str, history, t=0.7, p=0.9, mx=256):
    if image is None and not user_text.strip():
        return history

    MAX_HISTORY = 6
    messages = []

    # 注入最近的历史（仅文本往来）
    for u, a in history[-MAX_HISTORY:]:
        if u:
            messages.append({"role": "user", "content": [{"type": "text", "text": u}]})
        if a:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})

    # 本轮输入
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if user_text.strip():
        content.append({"type": "text", "text": user_text.strip()})
    else:
        content.append({"type": "text", "text": "请用中文详细描述这张图片的主要内容、关键物体与场景信息。"})
    messages.append({"role": "user", "content": content})

    # 编码与生成（与测试样例一致的范式）
    prompt = PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)
    inputs = PROCESSOR(
        text=[prompt],
        images=[image] if image is not None else None,  # 纯文本轮次时传 None
        return_tensors="pt"
    )

    # 兼容 device_map="auto" 的分布式权重：尽量将输入移到第一个可用设备
    try:
        inputs = inputs.to(MODEL.device)  # 大多数情况下可用
    except Exception:
        # 退而求其次：不强求 .to(MODEL.device)
        pass

    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            max_new_tokens=int(mx),
            do_sample=True,
            temperature=float(t),
            top_p=float(p),
        )

    output_text = PROCESSOR.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # 尽量抽取最后一段 assistant 回复；若无标记则直接用全文
    answer = output_text
    for tok in ["\nassistant\n", "\nassistant:\n", "assistant\n", "assistant:"][::-1]:
        if tok in output_text:
            answer = output_text.split(tok)[-1].strip()
            break

    history = history + [(user_text if user_text else "(图片描述)", answer)]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return history

# ======= Gradio UI =======
with gr.Blocks(title="Qwen2.5-VL 本地多模态（Gradio）", css="footer {display: none !important;}") as demo:
    gr.Markdown(
        """
        # Qwen2.5-VL-7B-Instruct 本地多模态演示
        - 模型：`./Qwen2.5-VL-7B-Instruct`（本地加载）
        - **用法**：① 鼠标上传图片（左侧）；② 输入本地路径加载；③ 下拉框一键选择当前目录图片。
        - 右侧聊天窗口支持**多轮对话**（保留最近 6 轮）。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 方式 ①：上传图片
            image_in = gr.Image(type="pil", label="上传图片（方式①）")

            # 方式 ②：文本路径加载
            with gr.Row():
                path_tb = gr.Textbox(label="本地图片路径（方式②）", placeholder="/abs/path/to/图片.png 或 ./图片1.png")
                load_from_path_btn = gr.Button("加载该路径图片", variant="secondary")
            path_status = gr.Markdown("")

            # 方式 ③：下拉选择当前目录图片
            with gr.Row():
                dropdown = gr.Dropdown(choices=list_local_images("."), label="当前目录图片（方式③）", allow_custom_value=False)
                refresh_btn = gr.Button("刷新列表", variant="secondary")
                load_from_dd_btn = gr.Button("加载选中图片", variant="secondary")
            dd_status = gr.Markdown("")

            # 文本输入
            text_in = gr.Textbox(label="文本指令（留空=自动做中文图片描述）", lines=3)

            with gr.Accordion("高级参数", open=False):
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                max_tokens = gr.Slider(16, 1024, value=256, step=8, label="max_new_tokens")

            gen_btn = gr.Button("🚀 生成回复", variant="primary")

        with gr.Column(scale=1):
            chat = gr.Chatbot(label="对话窗口", height=560)
            clear_btn = gr.ClearButton([image_in, text_in, chat, path_tb], value="清空")

    # 交互：加载图片（方式②：路径）
    def _load_path(path_str):
        im, msg = load_image_from_path(path_str)
        return im, msg

    load_from_path_btn.click(
        fn=_load_path,
        inputs=[path_tb],
        outputs=[image_in, path_status]
    )

    # 交互：刷新下拉框（方式③）
    def _refresh_dd():
        return gr.update(choices=list_local_images("."))

    refresh_btn.click(
        fn=_refresh_dd,
        inputs=[],
        outputs=[dropdown]
    )

    # 交互：加载下拉框选中图片（方式③）
    def _load_dd(name):
        im, msg = load_image_from_dropdown(name)
        return im, msg

    load_from_dd_btn.click(
        fn=_load_dd,
        inputs=[dropdown],
        outputs=[image_in, dd_status]
    )

    # 生成回复（多轮）
    def _infer(img, txt, hist, t, p, mx):
        return generate_reply(img, txt, hist, t, p, mx)

    gen_btn.click(
        fn=_infer,
        inputs=[image_in, text_in, chat, temperature, top_p, max_tokens],
        outputs=[chat]
    )

demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
