# -*- coding: utf-8 -*-
"""
母版音色培育 Worker (声音设计 + 声音克隆, Qwen3-TTS 双引擎)
==================================================
参照「母版TTS生成工作流.json」的两条链路实现:

  A. 声音设计(从无到有捏音色, 无需参考音频):
     A1 qwen3-design  : Qwen3-TTS VoiceDesign 模型, generate_voice_design(text, instruct=音色描述)
  B. 声音克隆(以参考音色为基底):
     B1 qwen3-clone   : Qwen3-TTS Base + create_voice_clone_prompt(参考音频 -> 克隆)

由 ComfyUI venv 解释器运行(qwen_tts 只在 ComfyUI venv):
  F:\ComfyUI\venv\Scripts\python.exe            -> qwen3-design / qwen3-clone

用法示例:
  F:\ComfyUI\venv\Scripts\python.exe singing\master_voice_worker.py ^
      --engine qwen3-design --design "清冷空灵的国风少女声线，语速舒缓" ^
      --lines "月光落进旧巷" --lines "风把故事吹向远方" ^
      --output_dir "outputs/master_voice/xxx"

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

ENGINES = ("qwen3-design", "qwen3-clone")


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
    parser = argparse.ArgumentParser(description="母版音色培育 Worker (声音设计 + 声音克隆, Qwen3-TTS)")
    parser.add_argument("--engine", required=True, choices=ENGINES,
                        help="引擎: qwen3-design(捏音色) / qwen3-clone(克隆)")
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
        model_dir = QWEN3_DESIGN_MODEL_DIR if engine == "qwen3-design" else QWEN3_CLONE_MODEL_DIR
    if not Path(model_dir).is_dir():
        print(f"error: 模型目录不存在: {model_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref_text = (args.ref_text or "").strip()

    # 加载模型
    model = load_qwen3_model(model_dir, args.device)

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
            else:  # qwen3-clone
                wav, sr = qwen3_clone(model, text, str(ref_path), ref_text,
                                      args.max_new_tokens, args.temperature,
                                      int(args.seed) + idx)
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
