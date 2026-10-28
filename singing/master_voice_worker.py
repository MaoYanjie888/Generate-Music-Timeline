# -*- coding: utf-8 -*-
"""
母版音色培育 Worker (声音设计 + 声音克隆 四引擎)
==================================================
参照「母版TTS生成工作流.json」的两条链路实现:

  A. 声音设计(从无到有捏音色, 无需参考音频):
     A1 qwen3-design  : Qwen3-TTS VoiceDesign 模型, generate_voice_design(text, instruct=音色描述)
     A2 voxcpm-design : VoxCPM2 纯文本模式, voice_design 指令以 "(描述)" 前缀拼入文本
  B. 声音克隆(以参考音色为基底):
     B1 qwen3-clone   : Qwen3-TTS Base + create_voice_clone_prompt(参考音频 -> 克隆)
     B2 voxcpm-clone  : VoxCPM2 + reference_wav_path(+prompt_wav_path/prompt_text 极致克隆)

按引擎选择解释器运行(qwen_tts 只在 ComfyUI venv, voxcpm 只在项目 venv_voxcpm):
  F:\ComfyUI\venv\Scripts\python.exe            -> qwen3-design / qwen3-clone
  F:\Generate-Music-Timeline\venv_voxcpm\Scripts\python.exe -> voxcpm-design / voxcpm-clone

用法示例:
  F:\ComfyUI\venv\Scripts\python.exe singing\master_voice_worker.py ^
      --engine qwen3-design --design "清冷空灵的国风少女声线，语速舒缓" ^
      --lines "月光落进旧巷" --lines "风把故事吹向远方" ^
      --output_dir "outputs/master_voice/xxx"

  F:\Generate-Music-Timeline\venv_voxcpm\Scripts\python.exe singing\master_voice_worker.py ^
      --engine voxcpm-design --design "warm female voice with ancient chinese charm" ^
      --lines "hello world" --output_dir "outputs/master_voice/xxx"

  F:\ComfyUI\venv\Scripts\python.exe singing\master_voice_worker.py ^
      --engine qwen3-clone --reference "ref.wav" --ref_text "参考音频文本" ^
      --lines "克隆文本一" --lines "克隆文本二" --output_dir "outputs/master_voice/xxx"
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QWEN3_DESIGN_MODEL_DIR = r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN3_CLONE_MODEL_DIR = r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base"
VOXCPM_MODEL_DIR = str(PROJECT_ROOT / "models" / "VoxCPM2")
VOXCPM_SAMPLE_RATE = 48000

ENGINES = ("qwen3-design", "voxcpm-design", "qwen3-clone", "voxcpm-clone")


def load_qwen3_model(model_dir: str, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel
    print(f"[master-voice] 加载 Qwen3-TTS 模型: {model_dir} (device={device})", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model


def load_voxcpm_model(model_dir: str, device: str):
    from voxcpm import VoxCPM
    print(f"[master-voice] 加载 VoxCPM2 模型: {model_dir} (device={device})", flush=True)
    return VoxCPM.from_pretrained(model_dir, device=device, load_denoiser=False)


def qwen3_design(model, text: str, design: str, language: str,
                 max_new_tokens: int, temperature: float, seed: int):
    """A1: Qwen3 VoiceDesign 从无到有捏音色。返回 (wav: np.ndarray, sr: int)。"""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    wavs, fs = model.generate_voice_design(
        text=[text],
        instruct=[design if design.strip() else ""],
        language=language or "Auto",
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=True,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
    )
    return wavs[0], fs


def qwen3_clone(model, text: str, ref_path: str, ref_text: str,
                max_new_tokens: int, temperature: float, seed: int):
    """B1: Qwen3 Base + 参考音频克隆(无情绪指令, 纯净音色克隆)。返回 (wav, sr)。"""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    ref_text = (ref_text or "").strip()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=[str(ref_path)],
        ref_text=[ref_text if ref_text else None],
        x_vector_only_mode=[not ref_text],
    )
    vc_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)

    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    ref_ids = None
    if ref_text:
        ref_ids = [model._tokenize_texts([model._build_ref_text(ref_text)])[0]]

    gen_kwargs = model._merge_generate_kwargs(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=True,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
    )

    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        instruct_ids=None,
        voice_clone_prompt=vc_dict,
        languages=["Auto"],
        non_streaming_mode=False,
        **gen_kwargs,
    )

    codes_for_decode = []
    for i, codes in enumerate(talker_codes_list):
        ref_code_list = vc_dict.get("ref_code", None)
        if ref_code_list is not None and ref_code_list[i] is not None:
            codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
        else:
            codes_for_decode.append(codes)

    wavs_all, fs = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

    wavs_out = []
    for i, wav in enumerate(wavs_all):
        ref_code_list = vc_dict.get("ref_code", None)
        if ref_code_list is not None and ref_code_list[i] is not None:
            ref_len = int(ref_code_list[i].shape[0])
            total_len = int(codes_for_decode[i].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wavs_out.append(wav[cut:])
        else:
            wavs_out.append(wav)

    return wavs_out[0], fs


def build_final_text(text: str, control_instruction: str = None) -> str:
    """VoxCPM2 voice_design: 控制指令以 (描述) 前缀拼入合成文本。"""
    if control_instruction and control_instruction.strip():
        return f"({control_instruction.strip()}){text}"
    return text


def voxcpm_design(model, text: str, design: str, cfg_value: float, inference_timesteps: int):
    """A2: VoxCPM2 纯文本模式捏音色。返回 (wav: np.ndarray, sr: int)。"""
    final_text = build_final_text(text, design)
    wav = model.generate(
        text=final_text,
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
    )
    return wav, VOXCPM_SAMPLE_RATE


def voxcpm_clone(model, text: str, ref_16k: str, ref_text: str,
                 cfg_value: float, inference_timesteps: int):
    """B2: VoxCPM2 + 参考音频克隆。返回 (wav: np.ndarray, sr: int)。"""
    kwargs = dict(
        text=text,
        reference_wav_path=str(ref_16k),
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
    )
    if ref_text and ref_text.strip():
        kwargs.update(prompt_wav_path=str(ref_16k), prompt_text=ref_text.strip())
    wav = model.generate(**kwargs)
    return wav, VOXCPM_SAMPLE_RATE


def preprocess_reference(reference: Path, out16k: Path) -> Path:
    """参考音频预处理为 16kHz 单声道 wav(AudioVAE 要求 16kHz 输入)。"""
    import librosa
    import soundfile as sf
    y, sr = librosa.load(str(reference), sr=None, mono=True)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    out16k.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out16k), y, 16000, subtype="FLOAT")
    return out16k


def write_wav(wav, sr: int, out_path: Path):
    import numpy as np
    import soundfile as sf
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    if wav.dtype != np.float32 and wav.dtype != np.float64:
        wav = wav.astype(np.float32)
    sf.write(str(out_path), wav, int(sr))
    return round(len(wav) / sr, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="母版音色培育 Worker (声音设计 + 声音克隆 四引擎)")
    parser.add_argument("--engine", required=True, choices=ENGINES,
                        help="引擎: qwen3-design/voxcpm-design(捏音色) qwen3-clone/voxcpm-clone(克隆)")
    parser.add_argument("--lines", action="append", required=True,
                        help="合成文本(可重复传多次, 每条生成一个样本)")
    parser.add_argument("--design", default="", help="音色设计指令(design 引擎必填; clone 引擎忽略)")
    parser.add_argument("--reference", default=None, help="参考音频路径(clone 引擎必填)")
    parser.add_argument("--ref_text", default=None, help="参考音频转录文本(clone 引擎可选, 提供则极致克隆)")
    parser.add_argument("--output_dir", required=True, help="培育样本输出目录")
    parser.add_argument("--model_dir", default=None, help="模型目录(默认按引擎自动选择)")
    parser.add_argument("--device", default="cuda:0", help="推理设备(默认 cuda:0)")
    # Qwen3 参数
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成码元数(默认 2048)")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度(默认 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子(每条自动 +idx)")
    parser.add_argument("--language", default="Auto", help="Qwen3 语言(默认 Auto)")
    # VoxCPM 参数
    parser.add_argument("--cfg_value", type=float, default=2.0, help="CFG 引导强度(默认 2.0)")
    parser.add_argument("--inference_timesteps", type=int, default=10, help="扩散推理步数(默认 10)")
    args = parser.parse_args()

    engine = args.engine
    items = [(ln or "").strip() for ln in args.lines]
    items = [ln for ln in items if ln]
    if not items:
        print("error: 合成文本为空", file=sys.stderr)
        return 2

    design = (args.design or "").strip()
    if engine.endswith("design") and not design:
        print("error: 声音设计引擎必须提供 --design 音色设计指令", file=sys.stderr)
        return 2

    ref_path = Path(args.reference) if args.reference else None
    if engine.endswith("clone"):
        if ref_path is None or not ref_path.is_file():
            print(f"error: 克隆引擎必须提供有效的 --reference: {ref_path}", file=sys.stderr)
            return 2

    # 模型目录默认值
    model_dir = args.model_dir
    if not model_dir:
        if engine == "qwen3-design":
            model_dir = QWEN3_DESIGN_MODEL_DIR
        elif engine == "qwen3-clone":
            model_dir = QWEN3_CLONE_MODEL_DIR
        else:
            model_dir = VOXCPM_MODEL_DIR
    if not Path(model_dir).is_dir():
        print(f"error: 模型目录不存在: {model_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 克隆模式预处理参考音频 16kHz
    ref_16k = None
    ref_text = (args.ref_text or "").strip()
    if engine.endswith("clone"):
        ref_16k = out_dir / "master_voice_ref_16k.wav"
        preprocess_reference(ref_path, ref_16k)
        print(f"[master-voice] 参考音频已预处理: {ref_16k}", flush=True)

    # 加载模型
    if engine.startswith("qwen3"):
        model = load_qwen3_model(model_dir, args.device)
    else:
        model = load_voxcpm_model(model_dir, args.device)

    outputs = []
    for idx, text in enumerate(items, 1):
        head = "".join(ch for ch in text[:10] if ch not in '\\/:*?"<>|').strip() or "sample"
        out_wav = out_dir / f"{idx:02d}_{engine}_{head}.wav"
        print(f"[master-voice] 培育 {idx}/{len(items)}: engine={engine} 文本={text[:24]}{'…' if len(text) > 24 else ''}", flush=True)
        try:
            if engine == "qwen3-design":
                wav, sr = qwen3_design(model, text, design, args.language,
                                       args.max_new_tokens, args.temperature,
                                       int(args.seed) + idx)
            elif engine == "qwen3-clone":
                wav, sr = qwen3_clone(model, text, str(ref_path), ref_text,
                                      args.max_new_tokens, args.temperature,
                                      int(args.seed) + idx)
            elif engine == "voxcpm-design":
                wav, sr = voxcpm_design(model, text, design, args.cfg_value, args.inference_timesteps)
            else:  # voxcpm-clone
                wav, sr = voxcpm_clone(model, text, str(ref_16k), ref_text,
                                       args.cfg_value, args.inference_timesteps)
            duration = write_wav(wav, sr, out_wav)
            outputs.append({
                "index": idx, "engine": engine, "text": text,
                "output": str(out_wav), "sample_rate": int(sr),
                "duration": duration,
            })
            print(f"[master-voice] 完成: {out_wav.name} ({duration}s)", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[master-voice] 失败 {idx}: {exc}", flush=True)
            outputs.append({"index": idx, "engine": engine, "text": text, "error": str(exc)})

    ok_cnt = sum(1 for o in outputs if o.get("output"))
    elapsed = time.time() - t0
    print(json.dumps({
        "ok": ok_cnt > 0,
        "engine": engine,
        "total": len(items),
        "success": ok_cnt,
        "output_dir": str(out_dir),
        "elapsed_sec": round(elapsed, 1),
        "outputs": outputs,
    }, ensure_ascii=False), flush=True)
    return 0 if ok_cnt > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
