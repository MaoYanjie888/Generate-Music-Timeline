# -*- coding: utf-8 -*-
"""
音准校正模块 (Pitch Correction)
===============================
用于修复 Seed-VC 换声后"五音不全"的问题。

原理:
    Seed-VC 换声链路中, RMVPE 提取的 f0 直接决定输出音高; 残留伴奏/参考音色
    不干净会导致逐音符跑调, auto_f0_adjust 全局音域平移会导致整体移调。
    本模块以"母版 vocal"(旋律目标)的 f0 为基准, 对"换声后 vocal"做逐帧音准
    校正, 同时用 World 声码器重合成, 保留换声音色(频谱包络来自换声轨)。

校正策略:
    1. 两轨重采样对齐后分别用 pyworld.harvest 提取 f0 (抗噪, 优于 RMVPE)
    2. 仅在双方都是 voiced 的帧上计算偏差 log-ratio
    3. 对偏差序列做滑动平均平滑(默认 0.5s), 得到慢变校正曲线
       -> 只修整体/局部音高偏移, 不破坏颤音、滑音、转音等唱腔细节
    4. 校正量超过阈值(默认 0.15 半音)才应用, 避免微差过度拉扯
    5. 超过 max_correction 半音的部分截断, 防过度修正
    6. pyworld (harvest + cheapTrick + D4C) 重合成校正后 f0
    7. 仅对「确实需要校正的段落」局部重合成, 其余段落保留原始换声轨,
       段落边缘平滑过渡 —— 避免整轨 World 声码器重合成抹平气声/齿音/瞬态,
       产生塑料/电子机械感(20261022 换声质量优化)

依赖: numpy / soundfile / pyworld (venv_timeline 已具备)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import pyworld as pw

SR_WORLD = 44100  # World 处理采样率
FRAME_PERIOD = 5.0  # ms, pyworld 默认


def _load_mono(path: Path, sr: int) -> tuple[np.ndarray, int]:
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y[:, 0]
    if orig_sr != sr:
        import scipy.signal
        num = int(round(len(y) * sr / orig_sr))
        y = scipy.signal.resample_poly(y, sr, orig_sr)
        # resample_poly 可能引入极短尾部, 截断到精确长度
        y = y[:num]
    y = y.astype(np.float64)
    # 20261028: Seed-VC 扩散输出偶发 NaN/Inf 样本, pyworld 原生层遇非有限值
    # 会死循环/极慢(第二次生成卡死根因之一), 一律清为非有限值后按静音处理
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, sr


def _extract_f0(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """提取 f0, 返回 (f0, t_sec); 静音帧为 0。

    优先 harvest (对歌声谐波丰富者稳定); 若 voiced 帧过少(纯正弦/异常信号),
    自动退回 dio + stonemask 细化。20261028: 原生层异常(死循环/崩溃)在
    try/except 兜底后返回全零 f0, 由上层"无校正"分支直出原轨, 不再拖垮进程。
    """
    try:
        f0, t = pw.harvest(y.astype(np.float64), sr, frame_period=FRAME_PERIOD)
    except Exception:
        f0 = np.zeros(max(1, int(len(y) * 1000.0 / FRAME_PERIOD)), dtype=np.float64)
        t = np.arange(len(f0)) * (FRAME_PERIOD / 1000.0)
        return f0, t
    voiced_ratio = (f0 > 0).sum() / max(1, len(f0))
    if voiced_ratio < 0.10:
        try:
            f0_dio, t = pw.dio(y.astype(np.float64), sr, frame_period=FRAME_PERIOD)
            f0 = pw.stonemask(y.astype(np.float64), f0_dio, t, sr)
        except Exception:
            f0 = np.zeros(len(f0), dtype=np.float64)
    return f0, t


def _smooth_win(smooth_sec: float) -> int:
    """与 _smooth_log_ratio 一致的平滑窗帧数(奇数)。"""
    win = max(3, int(round(smooth_sec * 1000.0 / FRAME_PERIOD)))
    if win % 2 == 0:
        win += 1
    return win


def _smooth_log_ratio(src_f0: np.ndarray, ref_f0: np.ndarray, sr: int,
                      smooth_sec: float, apply_thresh_st: float,
                      max_corr_st: float) -> np.ndarray:
    """计算平滑后的逐帧校正曲线 (log 域, 单位: 自然对数半音差)。

    src_f0/ref_f0: f0 序列 (0=unvoiced), 长度可不一致, 内部线性对齐
    返回: corr = src_f0 * exp(k) 中的 k, 仅 voiced 帧非零
    """
    # 长度对齐 (Seed-VC 输出与源 vocal 通常同长, 防御性重采样)
    if len(ref_f0) != len(src_f0):
        n_src = len(src_f0)
        x = np.linspace(0.0, 1.0, len(ref_f0))
        xi = np.linspace(0.0, 1.0, n_src)
        ref_f0 = np.interp(xi, x, ref_f0)
    n = min(len(src_f0), len(ref_f0))
    src = src_f0[:n].copy()
    ref = ref_f0[:n].copy()

    voiced = (src > 0) & (ref > 0)
    k = np.zeros(n, dtype=np.float64)
    if voiced.sum() < 8:  # 有效对齐帧太少, 放弃逐帧校正
        return k

    log_ratio = np.zeros(n, dtype=np.float64)
    log_ratio[voiced] = np.log(ref[voiced] / src[voiced])

    # 滑动平均平滑 (半汉宁窗近似), 得到慢变校正曲线
    win = _smooth_win(smooth_sec)
    kernel = np.hanning(win)
    kernel /= kernel.sum()
    k_smooth = np.convolve(log_ratio, kernel, mode="same")
    # 未平滑前超出应用阈值才认为有"需要修的偏差", 防止把唱腔微颤当误差
    dev_st = np.abs(log_ratio) * 12.0 / np.log(2.0)
    active = voiced & (dev_st > apply_thresh_st)
    if active.sum() < 8:
        return k  # 偏差整体很小, 不校正

    k[active] = k_smooth[active]
    # 截断过大校正量 (半音)
    max_k = max_corr_st * np.log(2.0) / 12.0
    k[active] = np.clip(k[active], -max_k, max_k)
    # 边缘淡出, 避免起止处突变
    fade = min(win, max(8, n // 10))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        k[:fade] *= ramp
        k[-fade:] *= ramp[::-1]
    return k


def pitch_correct(source_vocal: Path, reference_vocal: Path, out: Path,
                  smooth_sec: float = 0.5,
                  apply_thresh_st: float = 0.15,
                  max_corr_st: float = 2.0) -> Path:
    """以 reference_vocal(母版/旋律目标)为基准, 校正 source_vocal(换声轨)音准。

    返回输出 wav 路径。
    """
    src, _ = _load_mono(Path(source_vocal), SR_WORLD)
    ref, _ = _load_mono(Path(reference_vocal), SR_WORLD)

    src_f0, src_t = _extract_f0(src, SR_WORLD)
    ref_f0, _ = _extract_f0(ref, SR_WORLD)

    k = _smooth_log_ratio(src_f0, ref_f0, SR_WORLD, smooth_sec,
                          apply_thresh_st, max_corr_st)

    # 若整体几乎无校正, 直接拷贝原文件, 避免无谓重合成
    if not np.any(k != 0.0):
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), src.astype(np.float32), SR_WORLD)
        return out

    n = min(len(src_f0), len(ref_f0))
    # 用源轨(换声轨)频谱包络/非周期, 保证音色不变; 仅替换 f0
    sp = pw.cheaptrick(src, src_f0[:n].copy(), src_t[:n], SR_WORLD)
    ap = pw.d4c(src, src_f0[:n].copy(), src_t[:n], SR_WORLD)

    f0_out = src_f0[:n].copy()
    voiced = f0_out > 0
    f0_out[voiced] *= np.exp(k[voiced])

    y_syn = pw.synthesize(f0_out, sp, ap, SR_WORLD, frame_period=FRAME_PERIOD)

    # ---- 局部混合: 仅"需要校正的段落"使用合成轨, 其余保留原始换声轨 ----
    # 帧级权重(k!=0 的帧) -> 用平滑窗做卷积扩张, 得到带过渡带的连续权重
    win = _smooth_win(smooth_sec)
    pad = win // 2 + 2
    w_frames = (np.abs(k) > 0.0).astype(np.float64)
    kernel = np.hanning(2 * pad + 1)
    kernel /= kernel.sum()
    w_frames = np.convolve(w_frames, kernel, mode="same")
    w_frames = np.clip(w_frames, 0.0, 1.0)

    hop = int(round(SR_WORLD * FRAME_PERIOD / 1000.0))
    n_samples = len(src)
    w = np.interp(np.arange(n_samples) / float(hop), np.arange(len(w_frames)), w_frames)
    w = np.clip(w, 0.0, 1.0)

    m = min(len(y_syn), n_samples)
    y_out = src[:m] * (1.0 - w[:m]) + y_syn[:m] * w[:m]
    if len(y_syn) < n_samples:  # 合成轨意外偏短时, 尾部保留原始
        y_out = np.concatenate([y_out, src[len(y_syn):]])

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y_out.astype(np.float32), SR_WORLD)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音准校正: 以旋律参考轨为基准校正换声轨")
    parser.add_argument("--source", required=True, help="待校正 vocal (Seed-VC 换声输出)")
    parser.add_argument("--reference", required=True, help="旋律参考 vocal (母版 vocal)")
    parser.add_argument("--output", required=True, help="输出 wav 路径")
    parser.add_argument("--smooth-sec", type=float, default=0.5, help="平滑窗秒数(默认0.5)")
    parser.add_argument("--apply-thresh-st", type=float, default=0.15, help="应用阈值半音(默认0.15)")
    parser.add_argument("--max-corr-st", type=float, default=2.0, help="最大校正半音(默认2.0)")
    args = parser.parse_args()

    out_p = pitch_correct(Path(args.source), Path(args.reference), Path(args.output),
                          smooth_sec=args.smooth_sec,
                          apply_thresh_st=args.apply_thresh_st,
                          max_corr_st=args.max_corr_st)
    print(f"pitch_correct done: {out_p}")
