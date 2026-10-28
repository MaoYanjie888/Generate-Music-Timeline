#!/usr/bin/env python3
"""Generate-Music-Timeline — 整合版双子页面入口 (唱歌创作 + 歌词时间戳)

Tab1 唱歌创作: 参考音色 + 歌词 + 风格 -> 完整歌曲
  (主通道: 母版人声 -> Seed-VC 换参考音色, 保留旋律唱腔; 备用: VoxCPM 文本合成)
Tab2 歌词时间戳: 上传音频 + 歌词文件 -> 每句起止时间 -> 导出 lrc/csv/复制

启动: F:\\Generate-Music-Timeline\\venv_timeline\\Scripts\\python.exe webui.py
"""
import os
import sys
from pathlib import Path

# 本机走 Clash 代理时需放行本地地址，避免 gradio 本地访问被代理干扰
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

PROJECT_ROOT = Path(__file__).resolve().parent
# 统一工作目录到项目根: 保证时间戳模块相对路径 output/ 与唱歌模块 outputs/ 均落在新项目下
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SINGING_DIR = PROJECT_ROOT / "singing"
if str(SINGING_DIR) not in sys.path:
    sys.path.insert(0, str(SINGING_DIR))

import gradio as gr

from src.webui.app import build_mvgr_tab
from src.utils.check_updates import run_check
from singing.webui import build_singing_tab, ensure_yue2_service
from singing.emotion_tts_webui import build_emotion_tts_tab
from singing.master_voice_webui import build_master_voice_tab

APP_TITLE = "音乐创作 · 情绪语音 · 歌词时间戳 · 母版培育 · 整合版"
APP_DESC = (
    "整合 MVGR（歌词时间戳）、YuE2 + Seed-VC（参考音色唱歌）、Qwen3-TTS（情绪语音合成）与母版音色培育。\n"
    "Tab1 唱歌创作：参考音色 + 歌词 + 风格提示词 → 完整歌曲"
    "（主通道：母版人声→Seed-VC 换参考音色，保留旋律唱腔；备用：VoxCPM 文本合成）。\n"
    "Tab2 情绪语音合成：母版音色 + 文字 + 情绪标签 → 该音色朗读并带指定情绪的有声语音（Qwen3-TTS 本地推理）。\n"
    "Tab3 歌词时间戳：上传歌曲音频与歌词文件 → 识别每句歌词起止时间 → 导出 .lrc/.csv 或一键复制。\n"
    "Tab4 母版音色培育：从无到有捏音色（Qwen3/VoxCPM 声音设计，描述驱动）或参考音色克隆（Qwen3/VoxCPM 声音克隆）。\n"
)


def main() -> None:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_DESC}")
        with gr.Row():
            update_btn = gr.Button("检查模型更新", size="sm")
            update_result = gr.Textbox(
                label="更新检测结果",
                interactive=False,
                lines=8,
                visible=False,
            )
        update_btn.click(fn=run_check, outputs=[update_result]).then(
            lambda: gr.update(visible=True), outputs=[update_result]
        )
        with gr.Tabs():
            with gr.Tab("唱歌创作"):
                build_singing_tab()
            with gr.Tab("情绪语音合成"):
                build_emotion_tts_tab()
            with gr.Tab("歌词时间戳"):
                build_mvgr_tab()
            with gr.Tab("母版音色培育"):
                build_master_voice_tab()

    ok, msg = ensure_yue2_service()
    print(f"[YuE2] {msg}")
    if not ok:
        print(f"提示: 请检查 {PROJECT_ROOT}\\deps 下的 YuE2 一键包是否完整")

    port = 7860
    last_err = None
    for _ in range(10):
        try:
            demo.queue(max_size=8).launch(
                server_name="127.0.0.1",
                server_port=port,
                inbrowser=True,
                theme=gr.themes.Soft(),
                allowed_paths=[str(PROJECT_ROOT / "outputs"), str(PROJECT_ROOT)],
            )
            return
        except OSError as e:
            last_err = e
            print(f"[Generate-Music-Timeline] 端口 {port} 被占用，自动切换 {port + 1} ...")
            port += 1
    print(f"[Generate-Music-Timeline] 启动失败: {last_err}")
    raise last_err


if __name__ == "__main__":
    main()
