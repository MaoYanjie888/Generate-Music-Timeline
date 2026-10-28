#!/usr/bin/env python3
"""SongBloom 母版生成 worker (venv_songbloom 独立环境)

Generate-Music-Timeline 集成: 用 SongBloom(2B) 替代 YuE2 生成母版歌曲。
输入: 10s 参考音色(prompt) + SongBloom 格式歌词(token 化结构歌词)
输出: 48kHz 完整歌曲 flac (最长 150s)

用法:
  venv_songbloom\\Scripts\\python.exe songbloom_worker.py \
      --lyrics "[intro] [intro] , [verse] 句1. 句2. , [chorus] ..." \
      --prompt <参考音色.wav> --output <母版.flac> \
      --local-dir <权重目录> --dtype bfloat16

由 singing/pipeline.py 的 songbloom_synthesize 以 subprocess 调用,
stdout 写进度, 退出码 0/1, 日志落 outputs/logs/songbloom_*.log。
"""
import argparse
import os
import sys
import time

os.environ["DISABLE_FLASH_ATTN"] = "1"


def resolve_model_files(local_dir: str, model_name: str) -> dict:
    """本地权重目录必须包含 songbloom.py 所需的 5 个文件, 不做联网下载。"""
    need = {
        "cfg": f"{model_name}.yaml",
        "ckpt": f"{model_name}.pt",
        "vae_cfg": "stable_audio_1920_vae.json",
        "vae_ckpt": "autoencoder_music_dsp1920.ckpt",
        "g2p": "vocab_g2p.yaml",
    }
    found = {}
    for key, fname in need.items():
        p = os.path.join(local_dir, fname)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"SongBloom 权重缺失: {p}")
        found[key] = p
    return found


def load_model_config(cfg_file, parent_dir="./"):
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver(
        "concat", lambda *x: [item for sublist in x for item in sublist]
    )
    OmegaConf.register_new_resolver(
        "get_fname", lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    OmegaConf.register_new_resolver("load_yaml", OmegaConf.load)
    OmegaConf.register_new_resolver(
        "dynamic_path", lambda x: x.replace("???", parent_dir)
    )
    return OmegaConf.load(cfg_file)


def main():
    ap = argparse.ArgumentParser(description="SongBloom 母版生成 worker")
    ap.add_argument("--lyrics", required=True, help="SongBloom 格式歌词(token 化结构歌词)")
    ap.add_argument("--prompt", required=True, help="参考音色音频(内部截取前10s, 48kHz)")
    ap.add_argument("--output", required=True, help="输出母版 flac 路径")
    ap.add_argument("--local-dir", required=True, help="SongBloom 权重本地目录")
    ap.add_argument("--model-name", default="songbloom_full_150s")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    # 代码仓库固定为项目 deps/SongBloom (含 SongBloom/models 包)
    cand = os.path.join(os.path.dirname(repo_root), "deps", "SongBloom")
    if not os.path.isdir(os.path.join(cand, "SongBloom", "models")):
        raise RuntimeError(f"找不到 SongBloom 代码仓库: {cand}")
    sys.path.insert(0, cand)

    t0 = time.time()
    files = resolve_model_files(args.local_dir, args.model_name)
    print(f"model files ok: {os.path.basename(files['ckpt'])} "
          f"({os.path.getsize(files['ckpt']) / 1e9:.2f}GB)", flush=True)

    import torch
    import torchaudio
    from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler

    cfg = load_model_config(files["cfg"], parent_dir=args.local_dir)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"loading SongBloom model ({args.dtype}) ...", flush=True)
    model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=dtype)
    model.set_generation_params(**cfg.inference)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    prompt_wav, sr = torchaudio.load(args.prompt)
    if sr != model.sample_rate:
        prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
    prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(dtype)
    prompt_wav = prompt_wav[..., : 10 * model.sample_rate]
    print(f"prompt: {prompt_wav.shape[-1] / model.sample_rate:.1f}s @ {model.sample_rate}Hz", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print("generating ...", flush=True)
    wav = model.generate(args.lyrics, prompt_wav)
    torchaudio.save(args.output, wav[0].cpu().float(), model.sample_rate)
    print(f"done in {time.time() - t0:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
