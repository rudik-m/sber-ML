"""Настройка логирования через rich."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console()
_CONFIGURED = False


def get_console() -> Console:
    """Вернуть общий экземпляр консоли."""

    return _CONSOLE


def configure_logging(level: int = logging.INFO) -> None:
    """Один раз настроить стандартный логгер."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Вернуть настроенный логгер."""

    configure_logging()
    return logging.getLogger(name)
