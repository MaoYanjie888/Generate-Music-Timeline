"""定位运行资源（源码运行 / PyInstaller 打包后）的公共工具。"""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """项目根目录。

    - 源码运行：仓库根目录（webui.py 所在目录）；
    - PyInstaller onedir 打包后：EXE 所在目录（dist/MVGR）。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def resource_path(rel: str) -> Path:
    """返回相对资源（configs / models 等）的绝对路径。

    源码运行：仓库根目录（webui.py 所在目录）；
    PyInstaller onedir 打包后：
      1. 优先 EXE 同级目录（dist/MVGR）——手动分发时的传统布局；
      2. 其次 sys._MEIPASS（PyInstaller 6.x 把 datas 收集到 _internal）；
      3. 都找不到时回退 EXE 同级目录，由调用方决定是否在线回退。
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        p = exe_dir / rel
        if p.exists():
            return p
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            p2 = Path(meipass) / rel
            if p2.exists():
                return p2
        return p
    return Path(__file__).resolve().parent.parent.parent / rel
