# # run_qwen2_5_vl_caption.py
# # 用本地 Qwen2.5-VL-7B-Instruct 对 图片1.png 进行中文描述

# import os
# from PIL import Image
# import torch
# from transformers import AutoModelForCausalLM, AutoProcessor

# MODEL_DIR = os.path.abspath("./Qwen2.5-VL-7B-Instruct")
# IMAGE_PATH = os.path.abspath("./1.png")  # 如果中文文件名有编码问题，可改成 ./image1.png

# def main():
#     # 1) 加载模型与处理器（完全本地）
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_DIR,
#         trust_remote_code=True,
#         torch_dtype="auto",
#         device_map="auto"  # 自动放到可用的GPU/CPU；ROCm/海光DCU环境同样可用
#         # attn_implementation="flash_attention_2",  # 更快，显存更省
#     )
#     processor = AutoProcessor.from_pretrained(
#         MODEL_DIR,
#         trust_remote_code=True
#     )

#     # 2) 读取图片
#     image = Image.open(IMAGE_PATH).convert("RGB")

#     # 3) 构造多模态对话消息（Qwen2.5-VL 支持 messages + chat template）
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text",  "text": "请用中文详细描述这张图片的主要内容、关键物体与场景信息。"}
#             ]
#         }
#     ]
#     prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

#     # 4) 编码输入并推理
#     inputs = processor(
#         text=[prompt],
#         images=[image],
#         return_tensors="pt"
#     ).to(model.device)

#     with torch.no_grad():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )

#     # 5) 解码输出
#     output = processor.batch_decode(
#         generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )[0]

#     # Qwen 的 chat template 往往会把历史也打印出来，这里做个简单截断，只取 assistant 段落（如果模板不同可直接 print 全文）
#     # 简单做法：找最后一个“assistant”或“助手”开头的回答。为稳妥起见先直接打印全量：
#     print("\n===== 模型输出（原文） =====")
#     print(output)

# if __name__ == "__main__":
#     main()






# run_qwen2_5_vl_caption.py
# 用本地 Qwen2.5-VL-7B-Instruct 对 图片1.png 进行中文描述

import os
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor  # ← 关键：换成 Vision2Seq

MODEL_DIR = os.path.abspath("./Qwen2.5-VL-7B-Instruct")
IMAGE_PATH = os.path.abspath("./1.png")  # 如果中文文件名有编码问题，可改成 ./image1.png

def main():
    # 1) 加载模型与处理器（完全本地）
    model = AutoModelForVision2Seq.from_pretrained(             # ← 关键：换成 Vision2Seq
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",  # 自动把权重放到可用的 GPU/CPU；ROCm/海光DCU环境亦可
        # attn_implementation="flash_attention_2",  # 可选：若环境已装FA2
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )

    # 2) 读取图片
    image = Image.open(IMAGE_PATH).convert("RGB")

    # 3) 构造多模态对话消息（Qwen2.5-VL 支持 messages + chat template）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": "请用中文详细描述这张图片的主要内容、关键物体与场景信息。"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # 4) 编码输入并推理
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # 5) 解码输出
    output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n===== 模型输出（原文） =====")
    print(output)

if __name__ == "__main__":
    main()
