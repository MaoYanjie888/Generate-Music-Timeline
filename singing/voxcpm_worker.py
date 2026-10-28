# -*- coding: utf-8 -*-
"""
VoxCPM 零样本歌声克隆 Worker
=============================
由 F:\\Generate-Music-Timeline\\venv_voxcpm\\Scripts\\python.exe 独立运行的命令行脚本。

功能: 加载本地 VoxCPM2 权重(VoxCPM.from_pretrained), 用「目标歌词文本 + 参考音色音频」
      直接合成干声, 写出采样率为 model.tts_model.sample_rate(48000) 的 wav。
      参考音频自动预处理为 16kHz 单声道(AudioVAE 输入要求)后经 reference_wav_path 传入。

用法示例:
  venv_voxcpm\\Scripts\\python.exe singing\\voxcpm_worker.py ^
      --lyrics "歌词文本" --reference ref.wav --output vocal.wav ^
      [--reference_text "参考音频转录文本"] [--cfg_value 2.0] ^
      [--inference_timesteps 10] [--denoise] [--model_dir ...]
"""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

# 项目根 = singing/ 的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "VoxCPM2"
DEFAULT_SAMPLE_RATE = 48000


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


def transcribe_reference(ref_wav: Path) -> str:
    """用 venv 内已装的 funasr(SenseVoice) 自动转录参考音频; 失败返回空串(降级为基础克隆)。"""
    try:
        from funasr import AutoModel

        asr = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0",
                        disable_update=True, disable_pbar=True)
        res = asr.generate(input=str(ref_wav), language="auto", use_itn=True, batch_size_s=60)
        text = ""
        if res:
            text = res[0].get("text", "") or ""
        # 清洗 SenseVoice 语言/事件标签, 如 <|zh|> <|NEUTRAL|> <|Speech|>
        text = re.sub(r"<\|[^|]*\|>", "", text).strip()
        return text
    except Exception as exc:
        print(f"[warn] SenseVoice 自动转录失败, 降级为仅参考音色克隆: {exc}", flush=True)
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxCPM 零样本歌声克隆 Worker")
    parser.add_argument("--lyrics", required=True, help="目标歌词文本(合成内容)")
    parser.add_argument("--reference", required=True, help="参考音色音频路径")
    parser.add_argument("--output", required=True, help="输出干声 wav 路径")
    parser.add_argument("--reference_text", default=None, help="参考音频转录文本(可选, 提供则走极致克隆)")
    parser.add_argument("--auto_transcribe", action="store_true", default=False,
                        help="参考文本未提供时用 SenseVoice 自动转录")
    parser.add_argument("--cfg_value", type=float, default=2.0, help="CFG 引导强度(默认 2.0)")
    parser.add_argument("--inference_timesteps", type=int, default=10, help="扩散推理步数(默认 10)")
    parser.add_argument("--denoise", action="store_true", default=False, help="启用外部降噪增强(需模型加载 denoiser)")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR), help="本地 VoxCPM2 权重目录")
    parser.add_argument("--device", default="cuda", help="推理设备(默认 cuda)")
    args = parser.parse_args()

    lyrics = args.lyrics.strip()
    reference = Path(args.reference)
    out = Path(args.output)
    if not lyrics:
        print("error: 歌词为空", file=sys.stderr)
        return 2
    if not reference.is_file():
        print(f"error: 参考音频不存在: {reference}", file=sys.stderr)
        return 2

    # 0) 参考音频预处理为 16kHz 单声道
    ref_16k = out.parent / "voxcpm_ref_16k.wav"
    preprocess_reference(reference, ref_16k)

    # 1) 参考文本: 显式提供则使用; 否则按需自动转录
    reference_text = (args.reference_text or "").strip()
    if not reference_text and args.auto_transcribe:
        print("[voxcpm] 未提供参考转录文本, 使用 SenseVoice 自动转录...", flush=True)
        reference_text = transcribe_reference(ref_16k)
        if reference_text:
            print(f"[voxcpm] 自动转录结果: {reference_text[:80]}", flush=True)

    # 2) 加载本地 VoxCPM2 权重
    print(f"[voxcpm] 加载模型: {args.model_dir} (device={args.device}, load_denoiser=False)", flush=True)
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained(args.model_dir, device=args.device, load_denoiser=False)

    # 3) 合成干声
    kwargs = dict(
        text=lyrics,
        reference_wav_path=str(ref_16k),
        cfg_value=float(args.cfg_value),
        inference_timesteps=int(args.inference_timesteps),
    )
    if reference_text:
        # 极致克隆: 参考音频+其文本, 模型以音频续写方式精准还原音色细节
        kwargs.update(prompt_wav_path=str(ref_16k), prompt_text=reference_text)
    if args.denoise:
        try:
            if "denoise" in inspect.signature(model.generate).parameters:
                kwargs["denoise"] = True
            else:
                print("[warn] 当前 voxcpm 版本 generate 不支持 denoise 参数, 已忽略", flush=True)
        except Exception:
            print("[warn] 无法检查 generate 签名, denoise 已忽略", flush=True)

    print(f"[voxcpm] 开始合成(文本 {len(lyrics)} 字, cfg={args.cfg_value}, steps={args.inference_timesteps})", flush=True)
    try:
        wav = model.generate(**kwargs)
    except TypeError:
        # 兜底: 去掉 denoise 重试(部分版本签名不一致)
        if "denoise" in kwargs:
            kwargs.pop("denoise")
            print("[warn] generate 报参数错误, 已去掉 denoise 重试", flush=True)
            wav = model.generate(**kwargs)
        else:
            raise

    # 4) 写出 wav(采样率取模型 tts_model.sample_rate, 48000)
    import soundfile as sf
    sample_rate = int(getattr(model.tts_model, "sample_rate", DEFAULT_SAMPLE_RATE))
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, sample_rate)
    print(f"[voxcpm] 完成: {out} ({sample_rate}Hz, 时长 {len(wav) / sample_rate:.2f}s)", flush=True)

    # 5) 结构化结果输出(供 pipeline 解析)
    print(json.dumps({
        "ok": True,
        "output": str(out),
        "sample_rate": sample_rate,
        "reference_text": reference_text,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
