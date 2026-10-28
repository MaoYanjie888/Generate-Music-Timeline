# -*- coding: utf-8 -*-
"""
YuE2 参考音色歌曲创作工具 (换声主通道 + VoxCPM 备用)
=====================================================
默认走「母版人声 -> 音色转换」主通道 (voice_convert):
  YuE2 生成母版 -> htdemucs 分离 vocal+acc ->
  T8 内置 Seed-VC 将母版 vocal 换成参考音色(保留母版旋律唱腔) -> 混音
  - 旋律/唱腔来自母版, 音色来自参考音频, 解决「纯朗读」问题

备用通道 (voxcpm): VoxCPM2 零样本歌声克隆(纯文本合成, 偏朗读腔)
  仅作为无 Seed-VC 场景的备选。

提供 build_singing_tab() 供顶层双子页面嵌入; 也可独立运行:
  F:\Generate-Music-Timeline\venv_timeline\Scripts\python.exe singing\webui.py
"""

import time
import traceback
from pathlib import Path

import gradio as gr

from pipeline import run_creation, yue2_health
from pipeline import DEFAULT_CFG_VALUE, DEFAULT_INFERENCE_TIMESTEPS
from pipeline import DEFAULT_SEEDVC_DIFFUSION_STEPS, DEFAULT_SEEDVC_CFG_RATE, DEFAULT_SEEDVC_AUTO_F0_ADJUST
from pipeline import DEFAULT_PITCH_CORRECT
from pipeline import PROJECT_ROOT

APP_TITLE = "唱歌创作（母版换声 · Seed-VC）"
APP_DESC = (
    "整合 YuE2（作曲/编曲）、htdemucs（人声/伴奏分离）与 Seed-VC（参考音色换声）：\n"
    "上传一段参考音色，配上歌词与风格提示词，生成由该音色演唱的完整歌曲。\n"
    "主通道流程：YuE2 生成母版 → 分离母版人声+伴奏 → Seed-VC 将母版人声换成参考音色"
    "（保留母版旋律唱腔，不再朗读腔）→ 与伴奏混音。\n"
    "另有 VoxCPM 文本合成备用通道（纯 TTS，无旋律条件，结果偏朗读腔）。"
)

PROJECT_OUTPUTS = PROJECT_ROOT.parent / "outputs"


def check_service():
    try:
        h = yue2_health()
        if not h.get("ok"):
            return False, "YuE2 服务未就绪"
        return True, f"YuE2 服务正常 (v{h.get('version')})"
    except Exception as exc:
        return False, f"无法连接 YuE2 服务 (127.0.0.1:8189): {exc}"


def ensure_yue2_service(timeout: int = 180):
    """YuE2 未就绪时自动拉起 deps/ 下的一键包后台服务; 返回 (ok, msg)。

    开源场景: 用户只需双击启动器, 无需手动先开一键包; 一键包位置按项目根相对推导。
    """
    ok, msg = check_service()
    if ok:
        return True, msg
    kit = None
    deps = PROJECT_ROOT / "deps"
    if deps.is_dir():
        for p in deps.iterdir():
            if p.is_dir() and p.name.startswith("YuE2-T8-Local"):
                kit = p
                break
    if kit is None:
        return False, f"未找到 YuE2 一键包({deps}\\YuE2-T8-Local-*), 无法自动启动"
    ps1 = kit / "启动本地整合包.ps1"
    if not ps1.is_file():
        return False, f"未找到一键包启动脚本: {ps1}"
    import subprocess
    print(f"[YuE2] 服务未运行, 正在自动启动 {kit.name} ...")
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), "-NoBrowser"],
        cwd=str(kit),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        ok, msg = check_service()
        if ok:
            return True, f"YuE2 服务已自动启动 (v{msg.split('v')[-1]})"
    return False, "YuE2 自动启动超时(180s), 请查看一键包 logs\\server.stderr.log"


def do_create(reference, lyrics, style, seed, candidates, cfg_scale,
              prompt_max_sec, vocal_gain_db, acc_gain_db,
              channel, engine,
              diffusion_steps, cfg_rate, auto_f0_adjust, apply_pitch_correct, clean_lyrics,
              use_master_mix,
              auto_transcribe, reference_text, cfg_value, inference_timesteps, denoise,
              progress=gr.Progress()):
    if reference is None:
        raise gr.Error("请先上传参考音色音频")
    if not (lyrics and lyrics.strip()):
        raise gr.Error("请填写歌词")
    engine = str(engine or "yue2")
    if not (style and style.strip()) and engine != "songbloom":
        raise gr.Error("请填写风格提示词")

    def cb(stage, pct, msg):
        progress(pct / 100.0, desc=f"{stage} · {msg}")

    try:
        files = run_creation(
            reference,
            lyrics.strip(),
            (style or "").strip(),
            seed=int(seed),
            candidates=int(candidates),
            cfg_scale=float(cfg_scale),
            prompt_max_sec=int(prompt_max_sec),
            vocal_gain_db=float(vocal_gain_db),
            acc_gain_db=float(acc_gain_db),
            channel=str(channel),
            engine=engine,
            diffusion_steps=int(diffusion_steps),
            cfg_rate=float(cfg_rate),
            auto_f0_adjust=bool(auto_f0_adjust),
            apply_pitch_correct=bool(apply_pitch_correct),
            clean_lyrics=bool(clean_lyrics),
            use_master_mix=bool(use_master_mix),
            auto_transcribe=bool(auto_transcribe),
            reference_text=(reference_text or "").strip() or None,
            cfg_value=float(cfg_value),
            inference_timesteps=int(inference_timesteps),
            denoise=bool(denoise),
            progress_cb=cb,
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"创作失败: {exc}")

    return (
        files["final"],
        files["vocal"],
        files["master"],
        files["accompaniment"],
        files["work_dir"],
    )


def build_singing_tab():
    """构建「唱歌创作」Tab 内容(供顶层双子页面嵌入; 也可独立运行)。"""
    gr.Markdown(f"### {APP_TITLE}\n\n{APP_DESC}")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 输入素材")
            reference_audio = gr.Audio(label="参考音色（上传人声/歌声片段，建议 10~25 秒）", type="filepath")
            lyrics = gr.Textbox(label="歌词", lines=6, placeholder="输入歌曲歌词，支持多段…")
            style = gr.Textbox(label="风格提示词", lines=2,
                               placeholder="例：古风女声、婉转悠扬，笛箫伴奏，中速")
            create_btn = gr.Button("开始创作", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 母版引擎")
            engine = gr.Radio(
                label="母版生成引擎",
                choices=["yue2", "songbloom"],
                value="yue2",
                info="yue2=YuE2 按歌词+风格提示生成母版（最长约 4 分钟）；songbloom=SongBloom 用参考音色前 10 秒+歌词生成母版（最长 150 秒，风格提示词可留空）",
            )
            gr.Markdown("### 合成通道")
            channel = gr.Radio(
                label="合成通道",
                choices=["voice_convert", "voxcpm"],
                value="voice_convert",
                info="voice_convert=母版人声→Seed-VC换参考音色（默认，保留旋律唱腔）；voxcpm=VoxCPM文本合成（备用，偏朗读腔）",
            )

            with gr.Group(visible=True) as vc_group:
                gr.Markdown("### Seed-VC 换声参数（主通道）")
                diffusion_steps = gr.Slider(
                    label="扩散步数", minimum=4, maximum=50, step=1,
                    value=int(DEFAULT_SEEDVC_DIFFUSION_STEPS),
                    info="越大音质越细腻但越慢（默认 50）",
                )
                cfg_rate = gr.Slider(
                    label="CFG 引导强度", minimum=0.0, maximum=1.5, step=0.05,
                    value=float(DEFAULT_SEEDVC_CFG_RATE),
                    info="越小越自然、电音/金属感越弱；越大越贴近参考音色（默认 0.5）",
                )
                auto_f0_adjust = gr.Checkbox(
                    label="自动音域适配（auto_f0_adjust）", value=bool(DEFAULT_SEEDVC_AUTO_F0_ADJUST),
                    info="关闭后换声直接保留母版旋律（音准由音准校正兜底），融合更自然（默认关）",
                )
                apply_pitch_correct = gr.Checkbox(
                    label="音准校正（以母版旋律为基准）", value=bool(DEFAULT_PITCH_CORRECT),
                    info="换声后按母版 vocal 旋律拉回逐音符跑调，仅在跑调段落局部重合成、保留原轨音色（默认开）",
                )
                clean_lyrics = gr.Checkbox(
                    label="歌词发音友好预处理", value=True,
                    info="提交 YuE2 前用 pypinyin 检测多音字/生僻字并替换为常用同音字，降低认错字概率（默认开）",
                )

            with gr.Group(visible=False) as vxc_group:
                gr.Markdown("### VoxCPM 参数（备用通道）")
                cfg_value = gr.Slider(
                    label="CFG 引导强度", minimum=0.5, maximum=5.0, step=0.1, value=float(DEFAULT_CFG_VALUE),
                    info="越大越贴近参考音色与文本，过高可能不自然（默认 2.0）",
                )
                inference_timesteps = gr.Slider(
                    label="扩散推理步数", minimum=4, maximum=50, step=1, value=int(DEFAULT_INFERENCE_TIMESTEPS),
                    info="越大音质越细腻但越慢（默认 10）",
                )
                denoise = gr.Checkbox(
                    label="降噪增强（需加载 denoiser，默认关闭）", value=False,
                    info="仅当 voxcpm_worker 加载 denoiser 时生效，否则自动忽略",
                )
                gr.Markdown("#### 参考音色转录（极致克隆）")
                auto_transcribe = gr.Checkbox(
                    label="自动转录参考音频（SenseVoice）", value=True,
                    info="未提供转录文本时自动识别参考音频内容，开启极致克隆提升音色还原",
                )
                reference_text = gr.Textbox(
                    label="参考音频转录文本（可选）", lines=2,
                    placeholder="若已知参考音频内容可手动填写，音色还原更精准；留空则自动转录",
                )

            def toggle_channel(value):
                show_vc = value == "voice_convert"
                return (
                    gr.update(visible=show_vc),
                    gr.update(visible=not show_vc),
                )

            channel.change(toggle_channel, inputs=channel, outputs=[vc_group, vxc_group])

            gr.Markdown("### 母版生成参数")
            seed = gr.Number(label="随机种子", value=831001, precision=0)
            candidates = gr.Number(label="候选数", value=1, precision=0, minimum=1)
            cfg_scale = gr.Slider(label="母版 CFG 引导强度", minimum=0.5, maximum=2.0, step=0.1, value=1.0)

            gr.Markdown("### 混音与输出")
            use_master_mix = gr.Checkbox(
                label="母带混音（低切+压缩+混响+响度对齐母版）", value=True,
                info="主通道混音默认走专业母带混音（人声低切、软拐点压缩、卷积混响匹配伴奏空间感、响度对齐母版），解决干声+伴奏直接相加的生硬感；关闭则回退简单相加混音",
            )
            prompt_max_sec = gr.Slider(label="参考音色最大时长(秒)", minimum=5, maximum=30, step=1, value=25)
            vocal_gain_db = gr.Slider(label="人声音量增益(dB)", minimum=-12, maximum=12, step=0.5, value=0.0)
            acc_gain_db = gr.Slider(label="伴奏音量增益(dB)", minimum=-12, maximum=12, step=0.5, value=0.0)

    gr.Markdown("---")
    with gr.Row():
        final_audio = gr.Audio(label="最终歌曲（参考音色演唱）", type="filepath")
        vocal_audio = gr.Audio(label="换声/合成干声", type="filepath")
    with gr.Row():
        master_audio = gr.Audio(label="母版歌曲（引擎产物）", type="filepath")
        acc_audio = gr.Audio(label="伴奏（母版分离）", type="filepath")
    work_dir_box = gr.Textbox(label="本次作品目录", interactive=False)

    create_btn.click(
        do_create,
        inputs=[reference_audio, lyrics, style, seed, candidates, cfg_scale,
                prompt_max_sec, vocal_gain_db, acc_gain_db,
                channel, engine,
                diffusion_steps, cfg_rate, auto_f0_adjust, apply_pitch_correct, clean_lyrics,
                use_master_mix,
                auto_transcribe, reference_text, cfg_value, inference_timesteps, denoise],
        outputs=[final_audio, vocal_audio, master_audio, acc_audio, work_dir_box],
    )

    return dict(
        reference_audio=reference_audio, lyrics=lyrics, style=style, create_btn=create_btn,
        seed=seed, candidates=candidates, cfg_scale=cfg_scale,
        prompt_max_sec=prompt_max_sec, vocal_gain_db=vocal_gain_db, acc_gain_db=acc_gain_db,
        channel=channel, engine=engine,
        diffusion_steps=diffusion_steps, cfg_rate=cfg_rate, auto_f0_adjust=auto_f0_adjust,
        apply_pitch_correct=apply_pitch_correct, clean_lyrics=clean_lyrics,
        use_master_mix=use_master_mix,
        auto_transcribe=auto_transcribe, reference_text=reference_text,
        cfg_value=cfg_value, inference_timesteps=inference_timesteps, denoise=denoise,
        final_audio=final_audio, vocal_audio=vocal_audio,
        master_audio=master_audio, acc_audio=acc_audio, work_dir_box=work_dir_box,
    )


def create_app():
    """独立运行入口(单页)"""
    with gr.Blocks(title=APP_TITLE) as demo:
        build_singing_tab()
    return demo


if __name__ == "__main__":
    ok, msg = ensure_yue2_service()
    print(f"[YuE2] {msg}")
    if not ok:
        print(f"提示: 请检查 {PROJECT_ROOT}\\deps 下的 YuE2 一键包是否完整")
    demo = create_app()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                        theme=gr.themes.Soft(),
                        allowed_paths=[str(PROJECT_OUTPUTS), str(PROJECT_ROOT.parent)])
