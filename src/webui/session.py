"""会话持久化模块 — MVGR WebUI 素材/状态快照与断点续跑支持。

- session.json 保存 UI 状态（音频/照片/歌词/角色/时间轴/设置/上次批次）
- uploads/ 目录保存持久化后的素材副本（gradio 临时文件在会话结束后会被清理）
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

SESSION_FILENAME = "session.json"
UPLOADS_DIRNAME = "uploads"


def get_workspace() -> Path:
    """获取 workspace 目录（默认 ./output）。"""
    try:
        cfg = ConfigManager.build_from_ui(
            audio_path="", portrait_path="", style_preset="", segments=[]
        )
        ws = cfg.get("project", {}).get("workspace", "./output")
    except Exception:
        ws = "./output"
    return Path(ws)


def session_path() -> Path:
    """session.json 绝对路径。"""
    return get_workspace() / SESSION_FILENAME


def uploads_dir() -> Path:
    """uploads/ 素材持久化目录（自动创建）。"""
    d = get_workspace() / UPLOADS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def persist_upload(src: str, category: str) -> str:
    """把 gradio 临时文件复制到 uploads/，返回持久路径。

    Args:
        src: 源文件路径（gradio 临时文件）。
        category: audio / lyrics / portrait / role_image。
    Returns:
        持久化后的绝对路径字符串；失败返回原路径。
    """
    if not src:
        return ""
    src_p = Path(src)
    if not src_p.exists():
        return src
    # 已在 uploads/ 内则直接返回
    try:
        if src_p.resolve().is_relative_to(uploads_dir().resolve()):
            return str(src_p)
    except Exception:
        pass
    ext = src_p.suffix or ".bin"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = uploads_dir() / f"{category}_{ts}{ext}"
    try:
        shutil.copy2(src_p, dest)
        logger.info(f"素材持久化: {src} -> {dest}")
        return str(dest)
    except Exception:
        logger.exception(f"素材持久化失败: {src}")
        return src


def persist_lyrics(rows: list[list]) -> str:
    """把歌词时间轴写入 uploads/lyrics_*.lrc，返回路径。"""
    if not rows:
        return ""
    lines = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            start = float(row[0])
        except (TypeError, ValueError):
            continue
        text = str(row[2])
        mm = int(start // 60)
        ss = start - mm * 60
        lines.append(f"[{mm:02d}:{ss:05.2f}] {text}")
    if not lines:
        return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = uploads_dir() / f"lyrics_{ts}.lrc"
    try:
        dest.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"歌词时间轴落盘: {dest}")
        return str(dest)
    except Exception:
        logger.exception("歌词落盘失败")
        return ""


def save_session(data: dict[str, Any]) -> None:
    """写入 session.json（原子写：先写临时文件再替换）。"""
    try:
        data = dict(data)
        data["version"] = 1
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p = session_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(p)
    except Exception:
        logger.exception("保存 session.json 失败")


def load_session() -> Optional[dict[str, Any]]:
    """读取 session.json；不存在或损坏返回 None。"""
    p = session_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        logger.exception("读取 session.json 失败")
        return None


def clear_session() -> None:
    """删除 session.json。"""
    try:
        p = session_path()
        if p.exists():
            p.unlink()
    except Exception:
        pass
