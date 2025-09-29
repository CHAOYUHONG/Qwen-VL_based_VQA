# app_qwen25vl_gradio_auto.py
# 自动识别：优先视频 > 图片 > 纯聊天
# 依赖：pip install gradio pillow transformers torch
#       （视频分析需：pip install opencv-python）

import os
import gc
import glob
from typing import List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# （可选）限制可见 GPU（在 import 前设置更稳）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"   # NVIDIA
# os.environ["HIP_VISIBLE_DEVICES"]  = "0,1,2,3"   # 海光/ROCm

import torch
from PIL import Image
import gradio as gr
from transformers import AutoModelForVision2Seq, AutoProcessor   # ← 改为 Vision2Seq

# OpenCV（仅视频模式需要）
try:
    import cv2
except Exception:
    cv2 = None

# ===== 基础配置 =====
MODEL_DIR = os.path.abspath("./Qwen2.5-VL-7B-Instruct")

if torch.cuda.is_available():
    TORCH_DTYPE = torch.float16
    DEVICE_MAP = "auto"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
else:
    TORCH_DTYPE = torch.float32       # CPU 回退更稳
    DEVICE_MAP = "auto"

# ===== 模型加载（优先 flash-attn2，失败回退 sdpa）=====
def _load_with_attn(attn_impl: Optional[str]):
    kw = dict(
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True,
        # local_files_only=True,  # 完全离线可开启
    )
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    # 关键：使用 Vision2Seq
    return AutoModelForVision2Seq.from_pretrained(MODEL_DIR, **kw)

def load_model_and_processor():
    try:
        model = _load_with_attn("flash_attention_2")
        print("[info] 使用 flash_attention_2")
    except Exception as e:
        print(f"[warn] flash_attention_2 不可用，回退 sdpa：{e}")
        model = _load_with_attn("sdpa")

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        # local_files_only=True,
    )
    return model, processor

MODEL, PROCESSOR = load_model_and_processor()

# ===== 工具函数：图片文件列表 & 加载 =====
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def list_local_images(search_dir="."):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(search_dir, f"*{ext}")))
    files = [os.path.relpath(p, start=search_dir) for p in files]
    files.sort()
    return files

def load_image_from_path(path_str: str) -> Tuple[Optional[Image.Image], str]:
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

def load_image_from_dropdown(name: str) -> Tuple[Optional[Image.Image], str]:
    if not name:
        return None, "请先从下拉框选择图片"
    path = os.path.abspath(name)
    return load_image_from_path(path)

# ===== 视频抽帧 =====
def sample_video_frames_cv2(video_path: str, num_frames: int = 16) -> List[Image.Image]:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，请先：pip install opencv-python")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在：{video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count == 0:
        cap.release()
        raise RuntimeError("视频帧数为 0")

    idxs = list({int(i) for i in [round(k * (frame_count - 1) / max(1, num_frames - 1)) for k in range(num_frames)]})
    idxs.sort()
    images: List[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame).convert("RGB"))
    cap.release()
    if not images:
        raise RuntimeError("未成功抽取到帧")
    return images

# ===== 构造多模态消息 =====
def build_messages_text(user_text: str):
    return [{"role": "user", "content": [{"type": "text", "text": user_text}]}]

def build_messages_image(image: Image.Image, user_text: Optional[str]):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text or "请用中文详细描述这张图片的主要内容、关键物体与场景信息。"}
        ]
    }]

def build_messages_video(frames: List[Image.Image], user_text: Optional[str]):
    # Qwen2.5-VL 支持将多帧作为一个 video 段传入
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": user_text or "请用中文概述该视频的主要内容、场景变化与关键动作。"}
        ]
    }]

# ===== 通用推理 =====
def infer(messages, max_new_tokens=256, temperature=0.7, top_p=0.9):
    prompt = PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)

    images = None
    videos = None
    # 只取本轮的图/视频（如需支持多段，可扩展为累积列表）
    for m in messages:
        for c in m.get("content", []):
            if c.get("type") == "image":
                images = [c["image"]]
            elif c.get("type") == "video":
                videos = [c["video"]]

    inputs = PROCESSOR(text=[prompt], images=images, videos=videos, return_tensors="pt")
    # 尝试送到模型设备；device_map=auto 时不强求
    try:
        inputs = inputs.to(MODEL.device)
    except Exception:
        pass

    with torch.no_grad():
        out_ids = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
        )
    out_text = PROCESSOR.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    # 简单抓取最后一个 assistant 段
    answer = out_text
    for tok in ["\nassistant\n", "\nassistant:\n", "assistant\n", "assistant:"][::-1]:
        if tok in out_text:
            answer = out_text.split(tok)[-1].strip()
            break
    return answer

# ===== Gradio UI（自动识别模式）=====
with gr.Blocks(title="Qwen2.5-VL 自动模式（聊天/图片/视频）", css="footer {display:none!important;}") as demo:
    gr.Markdown(
        """
        # Qwen2.5-VL-7B-Instruct 多模态演示（自动识别：视频 > 图片 > 纯聊天）
        - 模型目录：`./Qwen2.5-VL-7B-Instruct`
        - 优先使用 **flash_attention_2**；不可用时自动回退 **sdpa**。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 文本提示：所有模式都会用作“指令”
            text_in = gr.Textbox(label="文本指令（可选：用于纯聊/图片/视频的提示词）", lines=3,
                                 placeholder="在此输入你的问题或指令")

            # 图片三种方式：上传 / 本地路径 / 下拉
            image_in = gr.Image(type="pil", label="上传图片（可选）")
            with gr.Row():
                path_tb = gr.Textbox(label="本地图片路径（可选）", placeholder="/abs/path/to/img.png 或 ./图片1.png")
                load_from_path_btn = gr.Button("加载路径图片", variant="secondary")
            path_status = gr.Markdown("")
            with gr.Row():
                dropdown = gr.Dropdown(choices=list_local_images("."), label="当前目录图片（可选）", allow_custom_value=False)
                refresh_btn = gr.Button("刷新列表", variant="secondary")
                load_from_dd_btn = gr.Button("加载选中图片", variant="secondary")
            dd_status = gr.Markdown("")

            # 视频：上传或路径
            video_in = gr.Video(label="上传视频（可选）")
            with gr.Row():
                video_path_tb = gr.Textbox(label="本地视频路径（可选）", placeholder="/abs/path/to/video.mp4 或 ./a.mp4")
                video_path_btn = gr.Button("确认视频路径", variant="secondary")
            video_status = gr.Markdown("")
            num_frames = gr.Slider(4, 64, value=16, step=2, label="视频抽帧数量（默认 16）")

            with gr.Accordion("高级生成参数", open=False):
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                max_tokens = gr.Slider(16, 1024, value=256, step=8, label="max_new_tokens")

            gen_btn = gr.Button("🚀 生成（自动识别模式）", variant="primary")

        with gr.Column(scale=1):
            chat = gr.Chatbot(label="对话窗口（保留最近 6 轮）", height=560)
            clear_btn = gr.ClearButton([image_in, text_in, chat, path_tb, dropdown, video_in, video_path_tb],
                                       value="清空")

    # 图片：路径 & 下拉
    load_from_path_btn.click(lambda p: load_image_from_path(p), inputs=[path_tb], outputs=[image_in, path_status])
    refresh_btn.click(lambda: gr.update(choices=list_local_images(".")), inputs=[], outputs=[dropdown])
    load_from_dd_btn.click(lambda n: load_image_from_dropdown(n), inputs=[dropdown], outputs=[image_in, dd_status])

    # 视频：仅确认路径存在性（真正抽帧在推理时做）
    def _confirm_video_path(p):
        if not p:
            return "请输入视频本地路径（可选）"
        path = os.path.abspath(os.path.expanduser(p))
        return f"已设置视频路径：{path}（将在推理时尝试读取并抽帧）"
    video_path_btn.click(_confirm_video_path, inputs=[video_path_tb], outputs=[video_status])

    # —— 自动识别 + 多轮 —— #
    def _infer(txt, img, vid, img_path, dd_name, vid_path, frames, hist, t, p, mx):
        """
        自动判别优先级：视频 > 图片 > 纯聊天
        vid: gr.Video 返回的对象（可能是临时文件路径或 dict 带 name）
        """

        # 汇入历史
        MAX_HISTORY = 6
        messages = []
        for u, a in hist[-MAX_HISTORY:]:
            if u:
                messages.append({"role": "user", "content": [{"type": "text", "text": u}]})
            if a:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})

        # 解析视频输入
        video_file = None
        if isinstance(vid, dict) and "name" in vid and vid["name"]:
            video_file = vid["name"]
        elif isinstance(vid, str) and vid:
            video_file = vid
        if not video_file and vid_path:
            video_file = os.path.abspath(os.path.expanduser(vid_path))

        # 若有视频 → 视频分析
        if video_file and os.path.exists(video_file):
            try:
                frames_list = sample_video_frames_cv2(video_file, num_frames=int(frames))
            except Exception as e:
                return hist + [("(视频)", f"抽帧失败：{e}")]
            messages.extend(build_messages_video(frames_list, (txt or "").strip() or None))
            answer = infer(messages, max_new_tokens=mx, temperature=t, top_p=p)
            hist = hist + [(txt if txt else "(视频分析)", answer)]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return hist

        # 解析图片输入：优先已上传 → 路径 → 下拉
        image_obj = img
        if image_obj is None and img_path:
            image_obj, _ = load_image_from_path(img_path)
        if image_obj is None and dd_name:
            image_obj, _ = load_image_from_dropdown(dd_name)

        if image_obj is not None:
            messages.extend(build_messages_image(image_obj, (txt or "").strip() or None))
            answer = infer(messages, max_new_tokens=mx, temperature=t, top_p=p)
            hist = hist + [(txt if txt else "(图片分析)", answer)]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return hist

        # 否则走纯聊天
        user_text = (txt or "").strip()
        if not user_text:
            user_text = "我们来聊聊：你支持哪些能力？"
        messages.extend(build_messages_text(user_text))
        answer = infer(messages, max_new_tokens=mx, temperature=t, top_p=p)
        hist = hist + [(user_text, answer)]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return hist

    gen_btn.click(
        fn=_infer,
        inputs=[text_in, image_in, video_in, path_tb, dropdown, video_path_tb,
                num_frames, chat, temperature, top_p, max_tokens],
        outputs=[chat]
    )

demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
