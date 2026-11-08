"""青空文庫（aozora）オリジナルデータ単体の統計集計スクリプト。

どのカレントディレクトリからでも直接実行可能:
    python /workspaces/slm/sandbox/sample_stats/aozora_stats.py
    python /workspaces/slm/sandbox/sample_stats/aozora_stats.py --aozora-directory=samples/aozora/cards/000005
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sample_stats.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--corpus=aozora", *sys.argv[1:]]))
