# -*- coding: utf-8 -*-
"""
专业母带混音模块 (Mix Master v2.0)
====================================
解决「干声 + 伴奏直接相加」的生硬拼接感, 让换声/合成干声与伴奏
融合达到接近母版歌曲的效果。

v2.0 升级 (针对「混音后杂音/机械音变重」):
  1. 软噪声门: 估计换声轨底噪, 低电平段(呼吸/垫底噪)软衰减, 不再被压缩器/响度对齐放大
  2. De-esser 去嘶: 8kHz 分频, 高频齿音/Seed-VC 机械毛刺动态衰减
  3. 混响 IR 暖化: 指数衰减噪声 IR 先低通 5kHz 再卷积, 消除白噪嘶声; L/R 不同种子出立体声扩散
  4. 立体声保留: 伴奏不再压成单声道, 人声 mono 补成双声道, 输出立体声
  5. 总线 16kHz 温和低通 + 响度对齐增益上限收紧 (+8dB), 避免抬噪

混音链:
  人声: 噪声门 -> 高通80Hz -> de-esser -> 软拐点压缩 -> 暖卷积混响(L/R异IR)
  伴奏: 高通25Hz(去次声)
  总线: 混音 -> 响度对齐母版 -> 16k低通 -> tanh软限幅

设计原则: 处理温和, 不引入新的机械/塑料感; 纯 numpy/scipy, 无外部音频工具依赖。

依赖: numpy / soundfile / scipy (venv_timeline 已具备)
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly

SR_MASTER = 44100  # 混音统一采样率


# ---------------- 基础工具 ----------------

def _resample(y: np.ndarray, orig_sr: int, sr: int) -> np.ndarray:
    g = gcd(orig_sr, sr)
    return resample_poly(y, sr // g, orig_sr // g)


def _load_audio(path: Path, sr: int = SR_MASTER, want_stereo: bool = True) -> np.ndarray:
    """读取音频 -> float64 (2, N) 立体声(单声道自动复制为双声道)。"""
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] == 1 and want_stereo:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    if orig_sr != sr:
        y = np.stack([_resample(y[:, i], orig_sr, sr) for i in range(y.shape[1])], axis=1)
    return y.astype(np.float64).T  # -> (2, N)


def _hp_filter(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """二阶 Butterworth 高通, 去除低频浑浊/次声。"""
    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def _lp_filter(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """二阶 Butterworth 低通, 温和滚降高频噪声。"""
    sos = butter(2, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def _frame_rms_db(x: np.ndarray, sr: int, block_sec: float = 0.010) -> tuple[np.ndarray, int]:
    """帧级 RMS(dB), 返回 (env_db, block 样本数)。"""
    block = max(1, int(sr * block_sec))
    nb = (len(x) + block - 1) // block
    env = np.array([float(np.sqrt(np.mean(x[i * block:(i + 1) * block] ** 2) + 1e-12))
                    for i in range(nb)])
    return 20.0 * np.log10(env + 1e-12), block


def _db_smooth(env_db: np.ndarray, sr: int, block: int,
               attack_ms: float = 5.0, release_ms: float = 200.0) -> np.ndarray:
    """dB 域一阶平滑(attack 快、release 慢), 插值到样本级。"""
    tau_a = max(1, int(round(sr * attack_ms / 1000.0 / block)))
    tau_r = max(1, int(round(sr * release_ms / 1000.0 / block)))
    ca = 1.0 - np.exp(-1.0 / tau_a)
    cr = 1.0 - np.exp(-1.0 / tau_r)
    smooth = np.zeros_like(env_db)
    g = 0.0
    for i in range(len(env_db)):
        if env_db[i] < g:
            g = g + cr * (env_db[i] - g)
        else:
            g = g + ca * (env_db[i] - g)
        smooth[i] = g
    idx = np.arange(len(env_db) * block) / float(block)
    return np.interp(idx, np.arange(len(env_db)), smooth)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 1e-12
    return float(np.sqrt(np.mean(np.clip(x, -1.0, 1.0) ** 2)))


def _lufs_integrated(x: np.ndarray, sr: int) -> float:
    """EBU R128 集成响度 (LUFS), 简化 K-weighting:
    高通 38Hz(RLB 近似) + 低通 1681Hz(高频搁架近似), 再 -0.691 偏移。
    用于母版响度对齐, 比 RMS 更符合人耳感知。
    """
    from scipy.signal import butter as _butter
    sos_hp = _butter(2, 38.0, btype="highpass", fs=sr, output="sos")
    sos_lp = _butter(2, 1681.0, btype="lowpass", fs=sr, output="sos")
    y = sosfilt(sos_hp, x)
    y = sosfilt(sos_lp, y)
    m = float(np.mean(y ** 2) + 1e-12)
    return 10.0 * np.log10(m) - 0.691


def _soft_clip(x: np.ndarray, ceiling: float = 0.97, knee_db: float = 6.0,
               attack_ms: float = 2.0, release_ms: float = 90.0) -> np.ndarray:
    """软拐点峰值限幅: 帧级包络在 knee 以下完全直通, 仅接近/超过 ceiling 时
    按 attack/release 平滑衰减, 取代 v2 的 tanh 全程饱和。

    修复「成品更平更闷」: tanh(x*1.2) 对全信号持续压动态; 新实现低电平
    完全不动, 只有峰值瞬态被轻压回 ceiling, 动态范围与瞬态得以保留。
    """
    peak = float(np.abs(x).max())
    if peak <= ceiling:
        return x  # 未达限幅阈值, 完全不动

    block = max(1, int(SR_MASTER * 0.005))           # 5ms 帧
    nb = (len(x) + block - 1) // block
    env = np.array([float(np.abs(x[i * block:(i + 1) * block]).max() + 1e-12)
                    for i in range(nb)])
    thr = ceiling / (10.0 ** (knee_db / 20.0))       # 提前 knee_db 开始介入(软拐点)
    gain = np.ones_like(env)
    over = env > thr
    gain[over] = ceiling / env[over]                 # 压回 ceiling

    # 时间平滑: attack 快(防过冲)、release 慢(防抽吸/咔哒)
    tau_a = max(1, int(round(SR_MASTER * attack_ms / 1000.0 / block)))
    tau_r = max(1, int(round(SR_MASTER * release_ms / 1000.0 / block)))
    ca = 1.0 - np.exp(-1.0 / tau_a)
    cr = 1.0 - np.exp(-1.0 / tau_r)
    gs = np.zeros_like(gain)
    g0 = 1.0
    for i in range(len(gain)):
        if gain[i] < g0:
            g0 = g0 + cr * (gain[i] - g0)
        else:
            g0 = g0 + ca * (gain[i] - g0)
        gs[i] = g0
    idx = np.arange(len(x)) / float(block)
    gain_s = np.interp(idx, np.arange(len(env)), gs)
    return x * gain_s


# ---------------- 噪声治理 ----------------

def _noise_gate(x: np.ndarray, sr: int, head_db: float = 6.0,
                ratio: float = 3.0, max_reduce_db: float = 24.0) -> np.ndarray:
    """软噪声门: 估计底噪, 低电平段软衰减, 高于门限 0dB。

    底噪估计取前 0.25s(无演唱段), 门限 = 底噪 + head_db。
    低于门限的帧按 (env - gate) * (1 - 1/ratio) 衰减, 最多 max_reduce_db,
    避免把呼吸声/微弱尾音硬切出咔哒声。
    """
    probe = int(sr * 0.25)
    floor = float(np.sqrt(np.mean(x[:probe] ** 2) + 1e-12))
    gate_lin = floor * (10.0 ** (head_db / 20.0))
    env_db, block = _frame_rms_db(x, sr)
    gate_db = 20.0 * np.log10(gate_lin + 1e-12)

    gain_db = np.where(env_db < gate_db,
                       np.clip(env_db - gate_db - 12.0, -max_reduce_db, 0.0),
                       0.0)
    gain_db = _db_smooth(gain_db, sr, block, attack_ms=8.0, release_ms=180.0)
    gain_db = gain_db[:x.shape[0]]  # 帧上采样可能多出末尾不足一帧的样本, 截断对齐
    return x * (10.0 ** (gain_db / 20.0))


def _de_esser(x: np.ndarray, sr: int, crossover_hz: float = 8000.0,
              thresh: float = 0.58, max_reduce_db: float = 6.0,
              gate_db: float = -40.0) -> np.ndarray:
    """动态去嘶: 8kHz 分频, 高频段能量占比过高时动态压低高频支路。

    对 Seed-VC 扩散伪影/齿音毛刺有效; 阈值内不动作, 温和不伤音色。
    仅在实际演唱段(env > gate_db)生效, 安静段不动避免误触发。
    """
    lp = _lp_filter(x, sr, crossover_hz)
    hp = x - lp
    env_db, block = _frame_rms_db(x, sr)
    hp_db, _ = _frame_rms_db(hp, sr)
    ratio = 10.0 ** ((hp_db - env_db) / 20.0)  # 高频占比
    # 超阈值部分按比例衰减, 平滑过渡
    over = np.maximum(ratio - thresh, 0.0)
    red_db = np.clip(over * 20.0, 0.0, max_reduce_db)
    red_db = np.where(env_db < gate_db, 0.0, red_db)  # 安静段保护
    red_db = _db_smooth(red_db, sr, block, attack_ms=3.0, release_ms=120.0)
    red_db = red_db[:x.shape[0]]  # 帧上采样可能多出末尾不足一帧的样本, 截断对齐
    return lp + hp * (10.0 ** (-red_db / 20.0))


def _soft_knee_compressor(x: np.ndarray, sr: int,
                          threshold_db: float = -12.0, ratio: float = 1.8,
                          knee_db: float = 8.0,
                          attack_ms: float = 10.0, release_ms: float = 160.0) -> np.ndarray:
    """软拐点动态压缩器 (纯 numpy 实现)。

    v2.1 放宽: 阈值 -18dB/2.5 -> -12dB/1.8, 只压明显过大的段落, 保留更多动态。
    """
    env_db, block = _frame_rms_db(x, sr)
    th = threshold_db
    over = env_db - th
    gain_db = np.zeros_like(env_db)
    below = over <= -knee_db / 2.0
    inside = (over > -knee_db / 2.0) & (over < knee_db / 2.0)
    above = over >= knee_db / 2.0
    gain_db[below] = 0.0
    gain_db[above] = (1.0 / ratio - 1.0) * over[above]
    k = knee_db / 2.0
    gain_db[inside] = (1.0 / ratio - 1.0) * (over[inside] + k) ** 2 / (4.0 * k)
    gain_db = _db_smooth(gain_db, sr, block, attack_ms=attack_ms, release_ms=release_ms)
    gain_db = gain_db[:x.shape[0]]  # 帧上采样可能多出末尾不足一帧的样本, 截断对齐
    return x * (10.0 ** (gain_db / 20.0))


def _convolution_reverb(x: np.ndarray, sr: int,
                        ir_sec: float = 1.3, wet: float = 0.10,
                        pre_delay_ms: float = 12.0,
                        ir_seed: int = 20261022,
                        ir_lowpass_hz: float = 5000.0) -> np.ndarray:
    """暖卷积混响: 指数衰减噪声 IR 先低通 5kHz, 消除白噪嘶声; L/R 用不同种子。

    混响只补空间感, 不喧宾夺主(wet 默认 10%)。
    """
    n_ir = max(1024, int(sr * ir_sec))
    t = np.arange(n_ir) / sr
    rng = np.random.default_rng(ir_seed)
    ir = rng.standard_normal(n_ir) * np.exp(-t / (ir_sec / 3.0))
    ir = _lp_filter(ir, sr, ir_lowpass_hz)          # 暖化: 滚掉高频嘶声
    ir = ir / np.sqrt(np.sum(ir ** 2) + 1e-12)
    predelay = int(sr * pre_delay_ms / 1000.0)
    x_pad = np.concatenate([np.zeros(predelay), x])
    wet_sig = fftconvolve(x_pad, ir)[:len(x)]
    return x * (1.0 - wet) + wet_sig * wet


# ---------------- 主混音 ----------------

def mix_master(vocal: Path, acc: Path, out: Path,
               vocal_gain_db: float = 0.0, acc_gain_db: float = 0.0,
               master_ref: Path | None = None,
               enable_compressor: bool = True,
               reverb_wet: float = 0.10,
               loudness_match: bool = True,
               enable_noise_gate: bool = True,
               enable_deesser: bool = True,
               final_lowpass_hz: float = 0.0,
               ceiling: float = 0.97) -> Path:
    """专业母带混音 v2.1: vocal + acc -> out (wav, 44100Hz, 立体声)。

    v2.1 修复「成品与母版波形差距大」:
      - 总线 tanh 全程饱和 -> 软拐点峰值限幅(仅峰值介入, 保留动态)
      - 压缩器阈值 -18dB/2.5 -> -12dB/1.8(温和化)
      - de-esser 阈值 0.42 -> 0.58(只压明显齿音)
      - 16k 总线低通默认关闭(成品发闷主因之一, 保留参数可手动开)
      - 响度对齐 RMS -> LUFS 感知对齐

    参数:
      master_ref: 母版歌曲路径; 提供时混音输出响度对齐到母版 LUFS(整体听感一致)
      enable_compressor: 是否启用软拐点压缩(默认开)
      reverb_wet: 人声混响湿声比例(0~1, 默认 0.10)
      loudness_match: 是否响度对齐母版(默认开)
      enable_noise_gate: 软噪声门去底噪(默认开)
      enable_deesser: 8kHz 去嘶压机械毛刺(默认开)
      final_lowpass_hz: 总线温和低通(默认 0=关闭, >0 时启用)
    """
    vocal = Path(vocal)
    acc = Path(acc)
    out = Path(out)
    if master_ref is not None:
        master_ref = Path(master_ref)

    v2 = _load_audio(vocal)          # (2, N)
    a2 = _load_audio(acc)            # (2, N)

    # ---- 人声链: 取 L 声道做主处理, 混响阶段 L/R 异 IR 出立体感 ----
    v = v2[0].copy()
    if enable_noise_gate:
        v = _noise_gate(v, SR_MASTER)
    v = _hp_filter(v, SR_MASTER, 80.0)
    if enable_deesser:
        v = _de_esser(v, SR_MASTER)
    if enable_compressor:
        v = _soft_knee_compressor(v, SR_MASTER)
    if reverb_wet > 0:
        vL = _convolution_reverb(v, SR_MASTER, wet=min(0.5, reverb_wet), ir_seed=20261022)
        vR = _convolution_reverb(v, SR_MASTER, wet=min(0.5, reverb_wet), ir_seed=20261023)
        v_out = np.stack([vL, vR])
    else:
        v_out = np.stack([v, v])

    # ---- 伴奏链: 低切去次声(htdemucs 分离低频偶有残留) ----
    a_out = np.stack([_hp_filter(a2[i], SR_MASTER, 25.0) for i in range(2)])

    v_out = v_out * (10.0 ** (vocal_gain_db / 20.0))
    a_out = a_out * (10.0 ** (acc_gain_db / 20.0))

    length = max(v_out.shape[1], a_out.shape[1])
    mix = np.zeros((2, length), dtype=np.float64)
    mix[:, :v_out.shape[1]] += v_out
    mix[:, :a_out.shape[1]] += a_out

    # ---- 响度对齐母版(两声道总 LUFS 感知对齐, 增益上限 +8dB 防抬噪) ----
    if loudness_match and master_ref is not None and master_ref.exists():
        ref2 = _load_audio(master_ref)
        ref_lufs = _lufs_integrated(ref2, SR_MASTER)
        cur_lufs = _lufs_integrated(mix, SR_MASTER)
        gain_db = float(np.clip(ref_lufs - cur_lufs, -8.0, 8.0))
        mix = mix * (10.0 ** (gain_db / 20.0))
        print(f"[mix_master] LUFS 对齐: {cur_lufs:.1f} -> {ref_lufs:.1f} dB (增益 {gain_db:+.1f} dB)")

    # ---- 总线: 16k 温和低通(压高频噪声) -> 软限幅 ----
    if final_lowpass_hz > 0:
        mix = np.stack([_lp_filter(mix[i], SR_MASTER, final_lowpass_hz) for i in range(2)])
    mix = np.stack([_soft_clip(mix[i], ceiling) for i in range(2)])

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mix.T.astype(np.float32), SR_MASTER)  # (N, 2)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="专业母带混音 v2: 降噪+去嘶+压缩+暖混响+响度对齐")
    parser.add_argument("--vocal", required=True, help="换声/合成干声")
    parser.add_argument("--acc", required=True, help="伴奏")
    parser.add_argument("--output", required=True, help="输出 wav")
    parser.add_argument("--master-ref", default=None, help="母版歌曲(响度参考, 可选)")
    parser.add_argument("--vocal-gain-db", type=float, default=0.0)
    parser.add_argument("--acc-gain-db", type=float, default=0.0)
    parser.add_argument("--no-compressor", dest="compressor", action="store_false")
    parser.add_argument("--compressor", action="store_true", default=True)
    parser.add_argument("--reverb-wet", type=float, default=0.10)
    parser.add_argument("--no-loudness-match", dest="loudness_match", action="store_false")
    parser.add_argument("--loudness-match", action="store_true", default=True)
    parser.add_argument("--no-noise-gate", dest="noise_gate", action="store_false")
    parser.add_argument("--noise-gate", action="store_true", default=True)
    parser.add_argument("--no-deesser", dest="deesser", action="store_false")
    parser.add_argument("--deesser", action="store_true", default=True)
    parser.add_argument("--lowpass-hz", type=float, default=0.0,
                        help="总线低通频率(默认0=关闭, 需要时如16000)")
    args = parser.parse_args()

    out_p = mix_master(
        Path(args.vocal), Path(args.acc), Path(args.output),
        vocal_gain_db=args.vocal_gain_db, acc_gain_db=args.acc_gain_db,
        master_ref=Path(args.master_ref) if args.master_ref else None,
        enable_compressor=args.compressor,
        reverb_wet=args.reverb_wet,
        loudness_match=args.loudness_match,
        enable_noise_gate=args.noise_gate,
        enable_deesser=args.deesser,
        final_lowpass_hz=args.lowpass_hz,
    )
    print(f"mix_master done: {out_p}")
