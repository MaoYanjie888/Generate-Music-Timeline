# -*- coding: utf-8 -*-
"""
情绪语音 TTS Worker (Qwen3-TTS Base + 参考音色克隆 + 情绪指令)
================================================================
由 F:\\ComfyUI\\venv\\Scripts\\python.exe 独立运行的命令行脚本
(qwen_tts 包只装在 ComfyUI 虚拟环境)。

能力: 上传「母版音色」参考音频 + 输入文本 + 选择情绪标签
      -> 用参考音色的音色朗读文本, 且语气按所选情绪演绎。

原理: 加载本地 Qwen3-TTS-12Hz-1.7B-Base,
      create_voice_clone_prompt 提取参考音频音色(x-vector / ICL),
      通过 instruct_ids 注入情绪指令(<|im_start|>user ... <|im_end|>),
      再走 model.generate 完成带情绪的克隆语音合成。

用法示例:
  F:\\ComfyUI\\venv\\Scripts\\python.exe singing\\emotion_tts_worker.py ^
      --reference "ref.wav" --text "你好, 世界" --emotion 开心 ^
      --output "out.wav" [--ref_text "参考音频文本"] [--max_new_tokens 2048]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 情绪标签 -> 注入给模型的语气指令(留空=自然语气)
EMOTION_PROMPTS = {
    "自然": "",
    "开心": "用开心、轻松、愉悦的语气朗读，语气中带着笑意",
    "悲伤": "用悲伤、低沉、忧郁的语气朗读，语速稍慢",
    "温柔": "用温柔、亲切、舒缓的语气朗读",
    "愤怒": "用愤怒、激动、有力的语气朗读，语气加重",
    "威严": "用威严、庄重、沉稳的语气朗读",
    "俏皮": "用俏皮、活泼、灵动的语气朗读",
    "激昂": "用慷慨激昂、充满热情与力量感的语气朗读",
    "平静": "用平静、自然、从容的语气朗读",
    "撒娇": "用撒娇、可爱、甜甜的语气朗读",
    "忧虑": "用担忧、焦虑、不安的语气朗读",
    "惊喜": "用惊喜、兴奋、充满感染力的语气朗读",
}

DEFAULT_MODEL_DIR = r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base"


def load_model(model_dir: str, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"[emotion-tts] 加载模型: {model_dir} (device={device})", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # 不依赖 flash-attn 预编译
    )
    return model


def synthesize(model, text: str, emotion_key: str, ref_path: str,
               ref_text: str = "", max_new_tokens: int = 2048,
               temperature: float = 0.9, seed: int = 42):
    """以参考音色克隆 + 情绪指令合成语音, 返回 (wav: np.ndarray, sr: int)。"""
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    emotion_prompt = (EMOTION_PROMPTS.get(emotion_key) or "").strip()
    ref_text = (ref_text or "").strip()

    # 1) 音色 prompt: 有参考文本 -> ICL 极致克隆; 无 -> 仅 x-vector 音色嵌入
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=[str(ref_path)],
        ref_text=[ref_text if ref_text else None],
        x_vector_only_mode=[not ref_text],
    )
    vc_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)

    # 2) 组装输入: user(情绪指令) + assistant(参考文本) + assistant(目标文本)
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    ref_ids = None
    if ref_text:
        ref_ids = [model._tokenize_texts([model._build_ref_text(ref_text)])[0]]
    instruct_ids = None
    if emotion_prompt:
        instruct_ids = [model._tokenize_texts([model._build_instruct_text(emotion_prompt)])[0]]

    gen_kwargs = model._merge_generate_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.05,
    )

    print(f"[emotion-tts] 合成中 (情绪={emotion_key}, 文本 {len(text)} 字, max_new_tokens={max_new_tokens})", flush=True)
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        instruct_ids=instruct_ids,
        voice_clone_prompt=vc_dict,
        languages=["Auto"],
        non_streaming_mode=False,
        **gen_kwargs,
    )

    # 3) 拼接参考码流并解码(与官方 generate_voice_clone 一致, 裁剪参考段)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-TTS 情绪语音合成 Worker")
    parser.add_argument("--reference", required=True, help="参考音色(母版音色)音频路径")
    parser.add_argument("--text", required=True, help="要朗读/合成的文本")
    parser.add_argument("--emotion", default="自然", help="情绪标签, 可选: " + "/".join(EMOTION_PROMPTS.keys()))
    parser.add_argument("--output", required=True, help="输出 wav 路径")
    parser.add_argument("--ref_text", default=None, help="参考音频转录文本(可选, 提供则走 ICL 极致克隆)")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Qwen3-TTS Base 模型目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备(默认 cuda:0)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成码元数(默认 2048)")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度(默认 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    text = args.text.strip()
    reference = Path(args.reference)
    out = Path(args.output)
    if not text:
        print("error: 文本为空", file=sys.stderr)
        return 2
    if not reference.is_file():
        print(f"error: 参考音频不存在: {reference}", file=sys.stderr)
        return 2
    if args.emotion not in EMOTION_PROMPTS:
        print(f"error: 未知情绪标签 {args.emotion}, 可选: {list(EMOTION_PROMPTS.keys())}", file=sys.stderr)
        return 2
    if not Path(args.model_dir).is_dir():
        print(f"error: 模型目录不存在: {args.model_dir}", file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        model = load_model(args.model_dir, args.device)
        wav, sr = synthesize(
            model, text, args.emotion, str(reference),
            ref_text=args.ref_text or "",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1

    import numpy as np
    import soundfile as sf
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    sf.write(str(out), wav, sr)
    print(f"[emotion-tts] 完成: {out} ({sr}Hz, 时长 {len(wav) / sr:.2f}s, 耗时 {time.time() - t0:.1f}s)", flush=True)
    print(json.dumps({
        "ok": True,
        "output": str(out),
        "sample_rate": sr,
        "duration": round(len(wav) / sr, 2),
        "emotion": args.emotion,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
