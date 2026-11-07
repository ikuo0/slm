"""プロセスのメモリ使用量（/proc/self/status の VmHWM / VmRSS / VmPeak 等）を取得する共通統計モジュール。

Linux の `/proc/self/status` からカーネルが記録しているプロセス全体のメモリ統計
（TensorFlow の C++ ランタイムや NumPy / SciPy のネイティブメモリを含む）を
オーバーヘッドなしで取得する。
`/proc/self/status` が参照できない環境では標準ライブラリ `resource.getrusage` にフォールバックする。
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

__all__ = [
    "read_proc_memory_kb",
    "get_memory_stats",
]


def read_proc_memory_kb(status_file: str | Path = "/proc/self/status") -> dict[str, int]:
    """`/proc/self/status` からメモリ関連項目 (VmHWM, VmRSS, VmPeak, VmSize 等) を KiB 単位で読み取る。"""
    path = Path(status_file)
    result: dict[str, int] = {}
    if not path.is_file():
        return result

    target_keys = ("VmPeak", "VmSize", "VmHWM", "VmRSS", "VmData", "VmStk", "VmExe", "VmLib")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, val_part = line.split(":", 1)
            key = key.strip()
            if key not in target_keys:
                continue
            tokens = val_part.strip().split()
            if tokens and tokens[0].isdigit():
                result[key] = int(tokens[0])
    except OSError:
        pass
    return result


def _fallback_peak_rss_kb() -> int:
    """`resource.getrusage` から最大物理メモリ (ru_maxrss) を KiB 単位で取得する。"""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS のみ bytes、Linux 等は KiB
    if sys.platform == "darwin":
        return int(ru.ru_maxrss // 1024)
    return int(ru.ru_maxrss)


def get_memory_stats(prev_stats: dict | None = None) -> dict:
    """現在のプロセスのメモリ統計（ピーク物理メモリ VmHWM・現在物理メモリ VmRSS・ピーク仮想メモリ VmPeak 等）を返す。

    Args:
        prev_stats: 中断再開時など、前回実行時のメモリ統計辞書がある場合に渡すと、
            ピーク値 (`peak_rss_kb`, `peak_vm_kb`) について前回と今回の最大値を維持する。

    Returns:
        メモリ使用量統計の辞書:
        - `peak_rss_kb`: 最大物理メモリ使用量 (VmHWM, KiB)
        - `peak_rss_mb`: 最大物理メモリ使用量 (MiB, 小数第2位)
        - `current_rss_kb`: 現在の物理メモリ使用量 (VmRSS, KiB)
        - `current_rss_mb`: 現在の物理メモリ使用量 (MiB, 小数第2位)
        - `peak_vm_kb`: 最大仮想メモリ使用量 (VmPeak, KiB)
        - `peak_vm_mb`: 最大仮想メモリ使用量 (MiB, 小数第2位)
        - `current_vm_kb`: 現在の仮想メモリ使用量 (VmSize, KiB)
        - `current_vm_mb`: 現在の仮想メモリ使用量 (MiB, 小数第2位)
    """
    proc_mem = read_proc_memory_kb()
    fallback_rss_kb = _fallback_peak_rss_kb()

    peak_rss_kb = int(proc_mem.get("VmHWM", fallback_rss_kb))
    current_rss_kb = int(proc_mem.get("VmRSS", peak_rss_kb))
    peak_vm_kb = int(proc_mem.get("VmPeak", 0))
    current_vm_kb = int(proc_mem.get("VmSize", 0))

    if isinstance(prev_stats, dict):
        prev_peak_rss = prev_stats.get("peak_rss_kb")
        if prev_peak_rss is not None:
            peak_rss_kb = max(peak_rss_kb, int(prev_peak_rss))
        prev_peak_vm = prev_stats.get("peak_vm_kb")
        if prev_peak_vm is not None:
            peak_vm_kb = max(peak_vm_kb, int(prev_peak_vm))

    return {
        "peak_rss_kb": peak_rss_kb,
        "peak_rss_mb": round(peak_rss_kb / 1024.0, 2),
        "current_rss_kb": current_rss_kb,
        "current_rss_mb": round(current_rss_kb / 1024.0, 2),
        "peak_vm_kb": peak_vm_kb,
        "peak_vm_mb": round(peak_vm_kb / 1024.0, 2),
        "current_vm_kb": current_vm_kb,
        "current_vm_mb": round(current_vm_kb / 1024.0, 2),
    }
