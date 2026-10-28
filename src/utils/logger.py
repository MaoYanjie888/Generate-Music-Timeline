"""日志系统模块。

提供统一的日志记录能力：
- 控制台彩色输出
- 文件按日期轮转
- 不同级别的日志过滤
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器。

    为不同日志级别显示不同颜色：
    - DEBUG: 蓝色
    - INFO: 绿色
    - WARNING: 黄色
    - ERROR: 红色
    - CRITICAL: 红色加粗
    """

    # ANSI 颜色码
    COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",      # 青色
        "INFO": "\033[32m",       # 绿色
        "WARNING": "\033[33m",    # 黄色
        "ERROR": "\033[31m",      # 红色
        "CRITICAL": "\033[1;31m", # 红色加粗
    }
    RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录。

        Args:
            record: 日志记录对象。

        Returns:
            格式化后的日志字符串。
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "mvgr",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """设置并返回全局日志器。

    Args:
        name: 日志器名称。
        level: 日志级别。
        log_dir: 日志文件目录。默认为当前目录下的 logs/。
        log_to_file: 是否输出到文件。
        log_to_console: 是否输出到控制台。

    Returns:
        配置完成的 Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式
    fmt = (
        "%(asctime)s | %(levelname)-18s | "
        "%(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    # ── 控制台处理器 ──
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(fmt, datefmt=datefmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # ── 文件处理器 ──
    if log_to_file:
        log_dir_path = Path(log_dir or "logs")
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # 主日志文件（按日期轮转）
        log_file = log_dir_path / f"{name}.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 错误日志单独文件
        error_log = log_dir_path / f"{name}_error.log"
        error_handler = RotatingFileHandler(
            filename=str(error_log),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

    return logger


# 全局默认日志器
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "mvgr") -> logging.Logger:
    """获取日志器实例。

    优先返回已设置的日志器，否则创建默认实例。

    Args:
        name: 日志器名称。

    Returns:
        Logger 实例。
    """
    global _logger

    if _logger is not None:
        child = _logger.getChild(name) if name != "mvgr" else _logger
        return child

    _logger = setup_logger(name="mvgr")
    return _logger
