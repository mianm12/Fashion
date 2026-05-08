from __future__ import annotations

import os
import sys
from typing import Final, TextIO

DEFAULT_LOG_LEVEL: Final = "INFO"
LOG_LEVELS: Final = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}
_default_source: str | None = None


def set_source(source: str | None) -> None:
    """设置当前进程默认日志来源名称。

    参数:
        source (str | None): 默认日志来源名称；传入 None 时清空默认来源。

    返回:
        None: 本函数仅更新日志模块上下文，不返回业务数据。
    """
    global _default_source
    _default_source = source


def current_source() -> str | None:
    """获取当前默认日志来源名称。

    参数:
        None: 本函数不接收外部参数。

    返回:
        str | None: 当前默认日志来源名称；未设置时返回 None。
    """
    return _default_source


def resolve_source(source: str | None) -> str | None:
    """解析单条日志实际使用的来源名称。

    参数:
        source (str | None): 单条日志显式传入的来源名称。

    返回:
        str | None: 显式来源优先；未传入时使用默认来源。
    """
    if source is not None:
        return source

    return current_source()


def current_log_level() -> str:
    """获取当前运行环境配置的日志级别。

    参数:
        None: 本函数直接读取 `FASHION_TREND_LOG_LEVEL` 环境变量。

    返回:
        str: 当前生效的日志级别名称；非法配置会回退到默认 `INFO`。
    """
    level_name = os.getenv("FASHION_TREND_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if level_name not in LOG_LEVELS:
        return DEFAULT_LOG_LEVEL

    return level_name


def is_enabled(level_name: str) -> bool:
    """判断指定日志级别当前是否允许输出。

    参数:
        level_name (str): 待判断的日志级别名称。

    返回:
        bool: 当指定级别不低于当前配置级别时返回 True，否则返回 False。

    异常:
        ValueError: 当传入未知日志级别名称时抛出。
    """
    normalized_level_name = level_name.upper()
    if normalized_level_name not in LOG_LEVELS:
        raise ValueError(f"未知日志级别: {level_name}")

    configured_level_name = current_log_level()
    return LOG_LEVELS[normalized_level_name] >= LOG_LEVELS[configured_level_name]


def format_message(level_name: str, message: str, source: str | None = None) -> str:
    """格式化标准日志文本。

    参数:
        level_name (str): 日志级别名称。
        message (str): 需要输出的日志正文。
        source (str | None，可选): 日志来源名称，默认为 None。

    返回:
        str: 拼接级别、来源和正文后的日志文本。
    """
    normalized_level_name = level_name.upper()
    if source is None:
        return f"[{normalized_level_name}] {message}"

    return f"[{normalized_level_name}] [{source}] {message}"


def emit(
    level_name: str,
    message: str,
    source: str | None,
    stream: TextIO,
) -> None:
    """按日志级别和目标流输出日志。

    参数:
        level_name (str): 日志级别名称。
        message (str): 需要输出的日志正文。
        source (str | None): 日志来源名称。
        stream (TextIO): 日志输出目标流。

    返回:
        None: 本函数仅负责输出日志，不返回业务数据。
    """
    normalized_level_name = level_name.upper()
    if not is_enabled(normalized_level_name):
        return

    print(
        format_message(normalized_level_name, message, source=resolve_source(source)),
        file=stream,
        flush=True,
    )


def debug(message: str, source: str | None = None) -> None:
    """输出 debug 级别日志到标准输出。

    参数:
        message (str): 需要输出的日志正文。
        source (str | None，可选): 日志来源名称，默认为 None。

    返回:
        None: 本函数仅负责输出日志，不返回业务数据。
    """
    emit("DEBUG", message, source, sys.stdout)


def info(message: str, source: str | None = None) -> None:
    """输出 info 级别日志到标准输出。

    参数:
        message (str): 需要输出的日志正文。
        source (str | None，可选): 日志来源名称，默认为 None。

    返回:
        None: 本函数仅负责输出日志，不返回业务数据。
    """
    emit("INFO", message, source, sys.stdout)


def warning(message: str, source: str | None = None) -> None:
    """输出 warning 级别日志到标准错误。

    参数:
        message (str): 需要输出的日志正文。
        source (str | None，可选): 日志来源名称，默认为 None。

    返回:
        None: 本函数仅负责输出日志，不返回业务数据。
    """
    emit("WARNING", message, source, sys.stderr)


def warn(message: str, source: str | None = None) -> None:
    """输出 warning 级别日志到标准错误，作为 `warning` 的简短别名。

    参数:
        message (str): 需要输出的日志正文。
        source (str | None，可选): 日志来源名称，默认为 None。

    返回:
        None: 本函数仅负责输出日志，不返回业务数据。
    """
    warning(message, source=source)


def error(message: str, source: str | None = None) -> None:
    """输出 error 级别日志到标准错误。

    参数:
        message (str): 需要输出的日志正文。
        source (str | None，可选): 日志来源名称，默认为 None。

    返回:
        None: 本函数仅负责输出日志，不返回业务数据。
    """
    emit("ERROR", message, source, sys.stderr)
