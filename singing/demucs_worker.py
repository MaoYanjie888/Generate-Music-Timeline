# -*- coding: utf-8 -*-
"""htdemucs 人声/伴奏分离 worker
由 singing/pipeline.py 以 subprocess 调用 YuE2 runtime python 执行,
替代 SoulX preprocess 内置的 mel-band-roformer-karaoke 分离
(该模型对 AI 合成歌手分离失败: 人声漏分到伴奏, 造成伴奏含人声 + 干声断音)。

用法: python demucs_worker.py <src_audio> <vocal_out.wav> <acc_out.wav>
输出: 44100Hz float32 wav
"""
import os
import sys
import yaml
import torch
from pathlib import Path
import soundfile as sf
from demucs.api import Separator
from demucs.apply import BagOfModels
from demucs.hf import load_safetensors_model

# 一键包位置: 优先取 pipeline 通过环境变量传入的 YUE2_KIT, 兜底按相对路径推导(开源可移植)
YUE2_ROOT = Path(os.environ.get("YUE2_KIT") or (
    Path(__file__).resolve().parent.parent / "deps" / "YuE2-T8-Local-v1.6.0-Windows-NVIDIA-20260919"
))


def separate(src: str, vocal_out: str, acc_out: str) -> None:
    model_root = YUE2_ROOT / "models" / "Demucs"
    bag_config = yaml.safe_load((model_root / "htdemucs.yaml").read_text(encoding="utf-8"))
    model = load_safetensors_model(model_root / "955717e8.safetensors")
    bag = BagOfModels([model], bag_config.get("weights"), bag_config.get("segment"))
    separator = Separator.__new__(Separator)
    separator._model = bag
    separator._audio_channels = bag.audio_channels
    separator._samplerate = bag.samplerate
    separator.update_parameter(device="cuda", shifts=1, overlap=0.25, split=True, segment=None,
                               jobs=0, progress=False, callback=None, callback_arg=None)
    separator._callback = None
    origin, stems = separator.separate_audio_file(str(src))
    vocal = stems["vocals"].detach().float().cpu()
    # 伴奏 = drums + bass + other 直接求和(模型直估), 不用 origin - vocals 差值法
    # (差值法会把分离模型漏分的人声残留在伴奏里, 且产生"挖空/断断续续"伪影)
    acc = (stems["drums"] + stems["bass"] + stems["other"]).detach().float().cpu()
    sr = bag.samplerate
    Path(vocal_out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(vocal_out, vocal.numpy().T, sr, subtype="FLOAT")
    sf.write(acc_out, acc.numpy().T, sr, subtype="FLOAT")
    del stems, origin, separator, bag, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    separate(sys.argv[1], sys.argv[2], sys.argv[3])
    print("OK")
