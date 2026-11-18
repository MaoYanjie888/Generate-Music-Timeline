#!/usr/bin/env python3
"""Generate-Music-Timeline — 一站式音乐创作工作台

Tab 顺序（精简后连续编号）：
  ① 歌词时间戳：音频+歌词 → 逐句时间 → lrc/csv
  ② 情绪语音合成：音色+文字+情绪 → 有声语音
  ③ 母版音色培育：声音设计 / 参考音色克隆
  ④ ACE-Step 音色跟随：参考音频 → Cover 翻唱，端到端无机械音（独立 API）

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
from singing.emotion_tts_webui import build_emotion_tts_tab
from singing.master_voice_webui import build_master_voice_tab
from singing.acestep_cover_ui import build_acestep_cover_tab

APP_TITLE = "音乐创作工作台"
APP_DESC = (
    "一站式本地音乐创作工具：歌词时间戳精对齐、情绪语音合成、母版音色培育，"
    "以及 ACE-Step 端到端音色跟随翻唱。全部本地运行，数据不出本机。\n\n"
    "| Tab | 用途 | 适用 |\n"
    "|---|---|---|\n"
    "| ① 歌词时间戳 | 音频+歌词 → 逐句起止时间 → lrc/csv | 辅助·精对齐 |\n"
    "| ② 情绪语音合成 | 音色+文字+情绪 → 带情绪朗读 | 辅助 |\n"
    "| ③ 母版音色培育 | 声音设计 / 参考音色克隆 | 辅助 |\n"
    "| ④ ACE-Step 音色跟随 | 参考音频 → Cover 翻唱，端到端无机械音（独立 API） | **推荐·端到端** |\n"
)


def main() -> None:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n\n{APP_DESC}")
        with gr.Accordion("模型更新检测（可选）", open=False):
            with gr.Row():
                update_btn = gr.Button("检查模型更新", size="sm")
                update_result = gr.Textbox(label="更新检测结果", interactive=False, lines=8)
            update_btn.click(fn=run_check, outputs=[update_result])

        with gr.Tabs():
            with gr.Tab("① 歌词时间戳"):
                build_mvgr_tab()
            with gr.Tab("② 情绪语音合成"):
                build_emotion_tts_tab()
            with gr.Tab("③ 母版音色培育"):
                build_master_voice_tab()
            with gr.Tab("④ ACE-Step 音色跟随"):
                build_acestep_cover_tab()

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
