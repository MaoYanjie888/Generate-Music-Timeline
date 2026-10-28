# -*- coding: utf-8 -*-
"""
情绪语音合成 Tab (Qwen3-TTS 参考音色克隆 + 情绪指令)
====================================================
独立 Gradio 分页: 上传母版音色 -> 输入文字 -> 选择情绪 -> 合成情绪语音。

依赖 Qwen3-TTS Base 模型与 qwen_tts 包(均位于 ComfyUI 环境),
合成实际由 singing/emotion_tts_worker.py 子进程完成。
"""

import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

try:
    from .emotion_tts_worker import EMOTION_PROMPTS  # 作为 singing 包内模块被 webui.py 导入
except ImportError:
    from emotion_tts_worker import EMOTION_PROMPTS  # 直接运行本文件时

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# qwen_tts 只安装在 ComfyUI 虚拟环境
COMFYUI_PYTHON = r"F:\ComfyUI\venv\Scripts\python.exe"
WORKER_PY = PROJECT_ROOT / "singing" / "emotion_tts_worker.py"
MODEL_DIR = r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base"

EMOTION_CHOICES = list(EMOTION_PROMPTS.keys())


def check_env():
    if not Path(COMFYUI_PYTHON).is_file():
        return False, f"未找到 ComfyUI 解释器: {COMFYUI_PYTHON}"
    if not Path(MODEL_DIR).is_dir():
        return False, f"未找到 Qwen3-TTS Base 模型: {MODEL_DIR}"
    return True, "环境就绪"


def do_emotion_tts(reference, text, emotion, ref_text,
                   max_new_tokens, temperature, seed,
                   progress=gr.Progress()):
    if reference is None:
        raise gr.Error("请先上传母版音色音频(参考音色, 建议 5~30 秒)")
    if not (text and text.strip()):
        raise gr.Error("请输入要朗读/合成的文字")

    ok, msg = check_env()
    if not ok:
        raise gr.Error(msg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    head = "".join(ch for ch in text.strip()[:12] if ch not in '\\/:*?"<>|').strip() or "speech"
    out_dir = PROJECT_ROOT / "outputs" / "emotion_tts" / f"{stamp}_{emotion}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"{head}_{emotion}.wav"

    cmd = [
        COMFYUI_PYTHON, str(WORKER_PY),
        "--reference", str(reference),
        "--text", text.strip(),
        "--emotion", emotion,
        "--output", str(out_wav),
        "--model_dir", MODEL_DIR,
        "--max_new_tokens", str(int(max_new_tokens)),
        "--temperature", str(float(temperature)),
        "--seed", str(int(seed)),
    ]
    if ref_text and ref_text.strip():
        cmd += ["--ref_text", ref_text.strip()]

    progress(0.05, desc="启动合成引擎(首次需加载模型, 约 1~3 分钟)")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            text=True,
        )
        last_json = None
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            if line.startswith("[emotion-tts]"):
                progress(0.5, desc=line.replace("[emotion-tts]", "").strip())
            if line.lstrip().startswith("{") and '"ok"' in line:
                try:
                    last_json = json.loads(line)
                except Exception:
                    pass
        proc.wait()
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"合成失败: {exc}")

    if proc.returncode != 0 or not out_wav.is_file():
        tail = "\n".join(lines[-8:]) if lines else "(无输出)"
        raise gr.Error(f"合成失败(exit={proc.returncode}):\n{tail}")

    if last_json and last_json.get("ok"):
        info = f"情绪语音完成: {emotion} | 时长 {last_json.get('duration', '?')}s | 采样率 {last_json.get('sample_rate', '?')}Hz"
    else:
        info = f"情绪语音完成: {emotion}"

    return str(out_wav), info, str(out_dir)


def build_emotion_tts_tab():
    """构建「情绪语音合成」Tab 内容。"""
    gr.Markdown(
        "### 情绪语音合成（Qwen3-TTS 参考音色克隆 + 情绪指令）\n\n"
        "上传一段**母版音色**（参考音频，建议 5~30 秒人声），输入文字并选择情绪标签，"
        "即可用该音色朗读文字，并按所选情绪演绎语气。\n"
        "原理：Qwen3-TTS Base 音色克隆（x-vector / ICL）+ 情绪指令注入，全程本地推理。"
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 输入")
            reference_audio = gr.Audio(
                label="母版音色（上传参考人声音频，建议 5~30 秒）", type="filepath")
            text = gr.Textbox(label="文字内容", lines=5, placeholder="输入要朗读/合成的文字…")
            emotion = gr.Dropdown(
                choices=EMOTION_CHOICES, value="自然", label="情绪",
                info="选择语气情绪，合成时将按此情绪演绎文本")
            ref_text = gr.Textbox(
                label="参考音频转录文本（可选，提供则音色还原更精准）", lines=2,
                placeholder="若已知参考音频内容可填写，留空则仅用音色嵌入")
            create_btn = gr.Button("合成情绪语音", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 合成参数")
            max_new_tokens = gr.Slider(
                label="最大生成码元数", minimum=512, maximum=4096, step=64, value=2048,
                info="文本越长所需越大；过大可能卡住（默认 2048）")
            temperature = gr.Slider(
                label="采样温度", minimum=0.5, maximum=1.5, step=0.05, value=0.9,
                info="越大越有表现力但越不稳定（默认 0.9）")
            seed = gr.Number(label="随机种子", value=42, precision=0)

    with gr.Row():
        out_audio = gr.Audio(label="合成结果（情绪语音）", type="filepath")
        status_box = gr.Textbox(label="状态", interactive=False, lines=2)
    work_dir_box = gr.Textbox(label="作品目录", interactive=False)

    create_btn.click(
        do_emotion_tts,
        inputs=[reference_audio, text, emotion, ref_text,
                max_new_tokens, temperature, seed],
        outputs=[out_audio, status_box, work_dir_box],
    )

    return dict(
        reference_audio=reference_audio, text=text, emotion=emotion, ref_text=ref_text,
        create_btn=create_btn, max_new_tokens=max_new_tokens,
        temperature=temperature, seed=seed,
        out_audio=out_audio, status_box=status_box, work_dir_box=work_dir_box,
    )


def create_app():
    with gr.Blocks(title="情绪语音合成") as demo:
        build_emotion_tts_tab()
    return demo


if __name__ == "__main__":
    ok, msg = check_env()
    print(f"[情绪语音] {msg}")
    demo = create_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True,
                        theme=gr.themes.Soft(),
                        allowed_paths=[str(PROJECT_ROOT / "outputs"), str(PROJECT_ROOT)])
