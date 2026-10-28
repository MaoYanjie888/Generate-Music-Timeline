# -*- coding: utf-8 -*-
"""
母版音色培育 Tab (声音设计 + 声音克隆 四引擎)
=============================================
独立 Gradio 分页, 参照「母版TTS生成工作流.json」实现从无到有捏音色:

  A. 声音设计(无需参考音频, 从描述捏出全新音色):
     A1 Qwen3 VoiceDesign : 填音色描述 + 文本 -> generate_voice_design
     A2 VoxCPM voice_design : 填音色描述 + 文本 -> "(描述)"前缀文本纯文本合成
  B. 声音克隆(以参考音色为基底微调):
     B1 Qwen3 克隆   : 上传参考音频(可选转录文本) -> 零样本克隆
     B2 VoxCPM 克隆  : 上传参考音频(可选转录文本) -> 极致克隆

产物落 outputs/master_voice/<stamp>/ 供 Tab1 参考音色自动扫描。
Qwen3 引擎由 F:\ComfyUI\venv\Scripts\python.exe 子进程执行,
VoxCPM 引擎由 F:\Generate-Music-Timeline\venv_voxcpm\Scripts\python.exe 子进程执行。
"""

import json
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMFYUI_PYTHON = r"F:\ComfyUI\venv\Scripts\python.exe"
VENV_VOXCPM_PYTHON = str(PROJECT_ROOT / "venv_voxcpm" / "Scripts" / "python.exe")
WORKER_PY = PROJECT_ROOT / "singing" / "master_voice_worker.py"

QWEN3_DESIGN_MODEL_DIR = r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN3_CLONE_MODEL_DIR = r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base"
VOXCPM_MODEL_DIR = str(PROJECT_ROOT / "models" / "VoxCPM2")

ENGINE_CHOICES = [
    ("qwen3-design", "Qwen3 声音设计（从描述捏音色）"),
    ("voxcpm-design", "VoxCPM 声音设计（从描述捏音色）"),
    ("qwen3-clone", "Qwen3 声音克隆（参考音色）"),
    ("voxcpm-clone", "VoxCPM 声音克隆（参考音色）"),
]
ENGINE_KEYS = [k for k, _ in ENGINE_CHOICES]
ENGINE_LABELS = dict(ENGINE_CHOICES)


def check_env(engine: str):
    if engine.startswith("qwen3"):
        python = COMFYUI_PYTHON
        model_dir = QWEN3_DESIGN_MODEL_DIR if engine == "qwen3-design" else QWEN3_CLONE_MODEL_DIR
        model_name = "Qwen3 VoiceDesign" if engine == "qwen3-design" else "Qwen3 Base"
    else:
        python = str(VENV_VOXCPM_PYTHON)
        model_dir = VOXCPM_MODEL_DIR
        model_name = "VoxCPM2"
    if not Path(python).is_file():
        return False, f"未找到解释器: {python}"
    if not Path(model_dir).is_dir():
        return False, f"未找到 {model_name} 模型: {model_dir}"
    return True, f"环境就绪（{ENGINE_LABELS[engine]}）"


def do_master_voice(engine, design, corpus, reference, ref_text,
                    max_new_tokens, temperature, seed,
                    cfg_value, inference_timesteps,
                    progress=gr.Progress()):
    if engine not in ENGINE_KEYS:
        raise gr.Error("请选择培育引擎")
    design = (design or "").strip()
    corpus = (corpus or "").strip()
    if engine.endswith("design"):
        if not design:
            raise gr.Error("声音设计引擎必须填写「音色描述指令」，例如：清冷空灵的国风少女声线，带古筝余韵，语速舒缓")
        if not corpus:
            raise gr.Error("请输入要合成的文本（每行一条）")
    else:
        if reference is None:
            raise gr.Error("声音克隆引擎请先上传参考音色音频")
        if not corpus:
            raise gr.Error("请输入要合成的文本（每行一条）")

    ok, msg = check_env(engine)
    if not ok:
        raise gr.Error(msg)

    lines = [ln.strip() for ln in corpus.splitlines() if ln.strip()]
    if not lines:
        raise gr.Error("合成文本解析后为空，请检查输入格式")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "master_voice" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    python = COMFYUI_PYTHON if engine.startswith("qwen3") else str(VENV_VOXCPM_PYTHON)
    cmd = [
        python, str(WORKER_PY),
        "--engine", engine,
        "--output_dir", str(out_dir),
    ]
    for ln in lines:
        cmd += ["--lines", ln]
    if engine.endswith("design"):
        cmd += ["--design", design]
    else:
        cmd += ["--reference", str(reference)]
        if ref_text and ref_text.strip():
            cmd += ["--ref_text", ref_text.strip()]
    if engine.startswith("qwen3"):
        cmd += [
            "--max_new_tokens", str(int(max_new_tokens)),
            "--temperature", str(float(temperature)),
            "--seed", str(int(seed)),
        ]
    else:
        cmd += [
            "--cfg_value", str(float(cfg_value)),
            "--inference_timesteps", str(int(inference_timesteps)),
        ]

    progress(0.05, desc="启动培育引擎（首次需加载模型，约 1~3 分钟）")
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
        lines_out = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines_out.append(line)
            if line.startswith("[master-voice]"):
                progress(0.3, desc=line.replace("[master-voice]", "").strip())
            if line.lstrip().startswith("{") and '"ok"' in line:
                try:
                    last_json = json.loads(line)
                except Exception:
                    pass
        proc.wait()
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"培育失败: {exc}")

    wavs = sorted(str(p) for p in out_dir.glob("*.wav"))
    if proc.returncode != 0 or not wavs:
        tail = "\n".join(lines_out[-10:]) if lines_out else "(无输出)"
        raise gr.Error(f"培育失败(exit={proc.returncode}):\n{tail}")

    if last_json and last_json.get("ok"):
        info = (
            f"培育完成: 成功 {last_json.get('success')}/{last_json.get('total')} 条, "
            f"引擎 {ENGINE_LABELS.get(last_json.get('engine'), last_json.get('engine'))}, "
            f"耗时 {last_json.get('elapsed_sec', '?')}s"
        )
    else:
        info = f"培育完成: 生成 {len(wavs)} 个样本"

    sample_names = [Path(p).name for p in wavs]
    return gr.update(choices=sample_names, value=(sample_names[0] if sample_names else None)), (wavs[0] if wavs else None), wavs, info, str(out_dir)


def build_master_voice_tab():
    """构建「母版音色培育」Tab 内容（声音设计 + 声音克隆）。"""
    gr.Markdown(
        "### 母版音色培育（从无到有捏音色）\n\n"
        "参照「母版TTS生成工作流」的两条链路：\n"
        "- **声音设计**：填一段音色描述 → 从无到有捏出全新音色（无需参考音频）；\n"
        "- **声音克隆**：上传参考音色 → 以它为基底克隆（可选转录文本做极致克隆）。\n\n"
        "培育产物自动落入 `outputs/master_voice/`，Tab1 唱歌创作会自动扫描为参考音色。"
    )

    engine = gr.Dropdown(
        choices=[(label, key) for key, label in ENGINE_CHOICES],
        value="qwen3-design",
        label="培育引擎",
    )
    design = gr.Textbox(
        label="音色描述指令（声音设计引擎核心，从描述捏出音色）",
        placeholder="示例：清冷空灵的国风少女声线，带古筝余韵，语速舒缓，尾音轻微气声",
        lines=3,
    )
    corpus = gr.Textbox(
        label="合成文本（每行一条，逐条生成样本）",
        placeholder="示例：\n月光落进旧巷，风把故事吹向远方\n我在群山之巅，等一场雪落\n这一夜，月色温柔，星河滚烫",
        lines=8,
    )

    with gr.Row():
        ref_audio = gr.Audio(
            label="参考音色音频（声音克隆引擎使用）",
            type="filepath",
            sources=["upload"],
        )
        ref_text = gr.Textbox(
            label="参考音频转录文本（克隆可选，提供则极致克隆）",
            placeholder="参考音频里说的话/唱的词",
            lines=2,
        )

    with gr.Row():
        max_tokens = gr.Slider(256, 4096, value=2048, step=64, label="max_new_tokens（Qwen3）")
        temperature = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="temperature（Qwen3）")
        seed = gr.Number(value=42, precision=0, label="seed（Qwen3）")
    with gr.Row():
        cfg_value = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="cfg_value（VoxCPM）")
        inference_timesteps = gr.Slider(2, 30, value=10, step=1, label="inference_timesteps（VoxCPM）")

    run_btn = gr.Button("开始培育", variant="primary")
    status_text = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        sample_dropdown = gr.Dropdown(
            label="试听样本（下拉选择，页内直接播放）",
            choices=[],
            interactive=True,
            allow_custom_value=True,
        )
        player = gr.Audio(
            label="试听播放器（听完满意再下载）",
            type="filepath",
            interactive=False,
        )
    out_files = gr.Files(label="下载区（满意后再下载保存）", interactive=False)
    out_dir_box = gr.Textbox(label="输出目录", interactive=False)

    def on_sample_change(name, out_dir):
        if not name or not out_dir:
            return gr.update(value=None)
        return gr.update(value=str(Path(out_dir) / name))

    sample_dropdown.change(
        fn=on_sample_change,
        inputs=[sample_dropdown, out_dir_box],
        outputs=player,
    )

    run_btn.click(
        fn=do_master_voice,
        inputs=[engine, design, corpus, ref_audio, ref_text,
                max_tokens, temperature, seed, cfg_value, inference_timesteps],
        outputs=[sample_dropdown, player, out_files, status_text, out_dir_box],
    )
