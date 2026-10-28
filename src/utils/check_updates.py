# -*- coding: utf-8 -*-
"""
模型/组件更新检测 (联网查询 GitHub Releases + HuggingFace lastModified)
======================================================================
用于 WebUI「检查更新」按钮: 对比本地模型目录状态与远端最新版本,
输出可读文本, 帮助判断哪些模型可能已发布新版本。

远端来源:
  - GitHub Releases: YuE (RVC-Boss/YuE), VoxCPM (PlayVoice/VoxCPM), Qwen3-TTS (QwenLM/Qwen3-TTS)
  - HuggingFace lastModified: Qwen3-ASR, Qwen3-ForcedAligner, Qwen3-TTS Base

本机走 Clash 代理(127.0.0.1:7897), urllib 默认跟随系统代理; 若直连失败
自动显式走本地代理重试一次。
"""

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_PROXY = "http://127.0.0.1:7897"

# 本地模型目录探测(存在即视为已安装, 记录安装时间)
LOCAL_MODELS = {
    "YuE2 后端": PROJECT_ROOT / "deps",
    "VoxCPM2": PROJECT_ROOT / "models" / "VoxCPM2",
    "Qwen3-TTS Base": Path(r"F:\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-Base"),
    "Qwen3-ASR-1.7B": PROJECT_ROOT / "models" / "Qwen3-ASR-1.7B-hf",
    "Qwen3-ForcedAligner-0.6B": PROJECT_ROOT / "models" / "Qwen3-ForcedAligner-0.6B-hf",
}

# 远端版本源
GITHUB_REPOS = [
    ("YuE2 后端", "RVC-Boss/YuE"),
    ("VoxCPM2", "PlayVoice/VoxCPM"),
    ("Qwen3-TTS", "QwenLM/Qwen3-TTS"),
]
HF_MODELS = [
    ("Qwen3-TTS Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    ("Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B-hf"),
    ("Qwen3-ForcedAligner-0.6B", "Qwen/Qwen3-ForcedAligner-0.6B-hf"),
]


def _open(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "marvis-update-check/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except Exception:
        # 兜底: 显式走 Clash 本地代理
        proxy_handler = urllib.request.ProxyHandler({"http": LOCAL_PROXY, "https": LOCAL_PROXY})
        opener = urllib.request.build_opener(proxy_handler)
        return opener.open(req, timeout=timeout)


def _gh_latest(repo: str) -> str:
    try:
        with _open(f"https://api.github.com/repos/{repo}/releases/latest") as r:
            d = json.loads(r.read().decode("utf-8", errors="replace"))
        return d.get("tag_name") or d.get("name") or "未知"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # 仓库无 Release: 兜底取最新 Tag
            try:
                with _open(f"https://api.github.com/repos/{repo}/tags") as r:
                    tags = json.loads(r.read().decode("utf-8", errors="replace"))
                if tags:
                    return f"{tags[0].get('name', '?')} (Tag)"
                return "暂无发布"
            except Exception:
                return "暂无发布"
        raise


def _hf_last_modified(model_id: str) -> str:
    with _open(f"https://huggingface.co/api/models/{model_id}") as r:
        d = json.loads(r.read().decode("utf-8", errors="replace"))
    ts = d.get("lastModified") or ""
    if ts:
        return ts[:10]
    return "未知"


def _local_state(path: Path) -> str:
    if not path.exists():
        return "未安装"
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime.strftime("%Y-%m-%d")


def run_check() -> str:
    """执行更新检测, 返回多行可读结果文本。"""
    lines = []
    lines.append("=== 模型更新检测 ===")

    # 本地安装状态
    lines.append("\n[本地状态]")
    for name, path in LOCAL_MODELS.items():
        lines.append(f"  {name}: {_local_state(path)}")

    # GitHub 最新版本
    lines.append("\n[GitHub 最新版本]")
    for name, repo in GITHUB_REPOS:
        try:
            ver = _gh_latest(repo)
            lines.append(f"  {name}: {ver} (repo: {repo})")
        except Exception as exc:
            lines.append(f"  {name}: 查询失败({type(exc).__name__}) — 请检查网络/代理后重试")

    # HF 模型更新时间
    lines.append("\n[HuggingFace 更新时间]")
    for name, mid in HF_MODELS:
        try:
            upd = _hf_last_modified(mid)
            lines.append(f"  {name}: 远端更新 {upd}")
        except Exception as exc:
            lines.append(f"  {name}: 查询失败({type(exc).__name__})")

    lines.append("\n提示: 一键包(YuE2/VoxCPM)本地版本以包内版本文件为准, 远端版本仅供对照;")
    lines.append("      若远端明显更新可前往对应仓库 Release 页核对后手动更新。")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run_check())
