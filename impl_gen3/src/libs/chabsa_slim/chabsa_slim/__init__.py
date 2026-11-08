"""chABSA、livedoor コーパス、および青空文庫（aozora）から本文テキストを抽出し、共通フォーマット JSON に変換するモジュール。

出力される共通フォーマット JSON 構造:
{
  "sentences": [
    {
      "original": "chabsa/e00008_ann.json",
      "lines": [
        "元文章の各行..."
      ]
    },
    {
      "original": "livedoor/topic-news/topic-news-xxx.txt",
      "lines": [
        "記事タイトル",
        "記事本文の各行..."
      ]
    },
    {
      "original": "aozora/cards/000005/files/5_ruby_21311/5_ruby_21311.txt",
      "lines": [
        "青空文庫の本文各行..."
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = [
    "EXCLUDED_LIVEDOOR_FILENAMES",
    "extract_chabsa_entry",
    "extract_chabsa_file",
    "collect_chabsa_entries",
    "extract_livedoor_entry",
    "extract_livedoor_file",
    "collect_livedoor_entries",
    "decode_aozora_bytes",
    "resolve_aozora_gaiji",
    "extract_aozora_entry",
    "extract_aozora_file",
    "collect_aozora_entries",
    "build_slim_dataset",
    "slim_document",
    "slim_file",
    "slim_directory",
]

EXCLUDED_LIVEDOOR_FILENAMES = frozenset({"LICENSE.TXT", "README.TXT", "CHANGES.TXT"})

# 青空文庫の正規表現パターン
_AOZORA_SEPARATOR_RE = re.compile(r"^-{15,}\s*$")
_AOZORA_FOOTER_RE = re.compile(
    r"^(?:底本[：:]|初出[：:]|入力[：:]|校正[：:]|翻訳[：:]|青空文庫作成ファイル[：:]|このファイルは、インターネットの図書館、青空文庫)"
)
_AOZORA_GAIJI_NOTE_RE = re.compile(r"※［＃([^］]+)］")
_AOZORA_UNICODE_RE = re.compile(r"(?:U\+|UCS-)([0-9A-Fa-f]{4,6})")
_AOZORA_JIS_MENKUTEN_RE = re.compile(r"(?:第[1-4]水準\s*)?([12])-(\d{1,2})-(\d{1,2})\b")
_AOZORA_NOTE_RE = re.compile(r"［＃[^］]*］")
_AOZORA_RUBY_RE = re.compile(r"《[^》]*》")
# クリーニング後に残っていたら変換失敗とみなす記号群
_AOZORA_LEFTOVER_MARKERS = ("［＃", "《", "》", "｜", "※", "〖", "〗")


def extract_chabsa_entry(doc: dict, original: str) -> dict | None:
    """chABSA の origin JSON 辞書 1 件から共通フォーマットの 1 エントリ {"original", "lines"} を生成する。"""
    lines: list[str] = []
    for s in doc.get("sentences", []):
        text = s.get("sentence", "")
        if not text:
            continue
        for subline in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            cleaned = subline.strip()
            if cleaned:
                lines.append(cleaned)
    if not lines:
        return None
    return {
        "original": original,
        "lines": lines,
    }


def extract_chabsa_file(
    in_path: str | Path,
    *,
    dataset_prefix: str = "chabsa",
) -> dict | None:
    """chABSA の JSON ファイル 1 件を読み込み、共通フォーマットの 1 エントリ {"original", "lines"} を返す。"""
    path = Path(in_path)
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    original = f"{dataset_prefix}/{path.name}" if dataset_prefix else path.name
    return extract_chabsa_entry(doc, original)


def collect_chabsa_entries(
    chabsa_dir: str | Path,
    *,
    dataset_prefix: str = "chabsa",
) -> list[dict]:
    """chABSA ディレクトリ内の *.json を列挙し、共通フォーマットの sentences エントリ一覧を返す。"""
    dir_path = Path(chabsa_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"chABSA ディレクトリが存在しません: {dir_path}")

    entries: list[dict] = []
    for in_path in sorted(dir_path.glob("*.json")):
        entry = extract_chabsa_file(in_path, dataset_prefix=dataset_prefix)
        if entry is not None:
            entries.append(entry)
    return entries


def extract_livedoor_entry(text: str, original: str) -> dict | None:
    """livedoor コーパスの記事テキスト 1 件から 3 行目以降（タイトル＋本文）を抽出し、空行を除いたエントリを返す。

    livedoor コーパス記事フォーマット:
        1行目: 記事URL
        2行目: 記事の日付
        3行目: 記事タイトル
        4行目以降: 記事本文
    """
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if len(raw_lines) < 3:
        return None

    lines = [line.strip() for line in raw_lines[2:] if line.strip()]
    if not lines:
        return None
    return {
        "original": original,
        "lines": lines,
    }


def extract_livedoor_file(
    in_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    dataset_prefix: str = "livedoor",
) -> dict | None:
    """livedoor コーパスの記事ファイル 1 件を読み込み、共通フォーマットの 1 エントリ {"original", "lines"} を返す。"""
    path = Path(in_path)
    if path.name.upper() in EXCLUDED_LIVEDOOR_FILENAMES:
        return None

    if base_dir is not None:
        try:
            rel_path = path.relative_to(Path(base_dir)).as_posix()
        except ValueError:
            rel_path = path.name
    else:
        rel_path = path.name

    original = f"{dataset_prefix}/{rel_path}" if dataset_prefix else rel_path
    text = path.read_text(encoding="utf-8", errors="replace")
    return extract_livedoor_entry(text, original)


def collect_livedoor_entries(
    livedoor_dir: str | Path,
    *,
    dataset_prefix: str = "livedoor",
) -> list[dict]:
    """livedoor コーパスディレクトリ（samples/livedoor/text 等）配下の記事 *.txt を列挙し、エントリ一覧を返す。

    README.txt / CHANGES.txt / LICENSE.txt は自動的に除外する。
    """
    dir_path = Path(livedoor_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"livedoor ディレクトリが存在しません: {dir_path}")

    entries: list[dict] = []
    for in_path in sorted(dir_path.rglob("*.txt")):
        if not in_path.is_file():
            continue
        if in_path.name.upper() in EXCLUDED_LIVEDOOR_FILENAMES:
            continue
        entry = extract_livedoor_file(
            in_path,
            base_dir=dir_path,
            dataset_prefix=dataset_prefix,
        )
        if entry is not None:
            entries.append(entry)
    return entries


def decode_aozora_bytes(raw_bytes: bytes) -> str | None:
    """青空文庫テキストのバイト列を UTF-8 または CP932 (Shift_JIS) としてデコードし UTF-8 文字列にする。"""
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _menkuten_to_char(plane: int, row: int, cell: int) -> str | None:
    """JIS X 0213 の面区点番号 (plane-row-cell) を EUC-JIS-2004 経由で対応する Unicode 1文字へ変換する。"""
    if plane not in (1, 2) or not (1 <= row <= 94) or not (1 <= cell <= 94):
        return None
    b1 = row + 0xA0
    b2 = cell + 0xA0
    raw = bytes([b1, b2]) if plane == 1 else bytes([0x8F, b1, b2])
    try:
        return raw.decode("euc_jis_2004")
    except UnicodeDecodeError:
        return None


def resolve_aozora_gaiji(text: str) -> str:
    """青空文庫の外字注記 `※［＃...］` を可能な限り対応する Unicode 文字に置換する。

    変換できない外字は `※` を残し、後段の検査で検出・除外できるようにする。
    """
    def _replace(match: re.Match[str]) -> str:
        note = match.group(1)
        m_uni = _AOZORA_UNICODE_RE.search(note)
        if m_uni:
            try:
                return chr(int(m_uni.group(1), 16))
            except ValueError:
                pass
        m_jis = _AOZORA_JIS_MENKUTEN_RE.search(note)
        if m_jis:
            ch = _menkuten_to_char(
                int(m_jis.group(1)),
                int(m_jis.group(2)),
                int(m_jis.group(3)),
            )
            if ch is not None:
                return ch
        # 変換不能な場合は未解決マーカーとして ※ を残す
        return "※"

    return _AOZORA_GAIJI_NOTE_RE.sub(_replace, text)


def _strip_aozora_header_and_footer(raw_lines: list[str]) -> list[str]:
    """青空文庫テキストから冒頭（作品名・著者名・記号説明区切り線）と末尾（底本等の書誌情報）を除去する。"""
    n = len(raw_lines)
    if n == 0:
        return []

    # 1. 冒頭メタデータの除去
    # 先頭 80 行以内に `-------------------------------------------------------` が 2 本あれば、2 本目の直後から本文とする
    header_scan_limit = min(n, 80)
    sep_indices = [
        i for i in range(header_scan_limit) if _AOZORA_SEPARATOR_RE.match(raw_lines[i])
    ]
    if len(sep_indices) >= 2:
        body_start = sep_indices[1] + 1
    else:
        # 区切り線ブロックがない作品は、先頭の非空行ブロック（作品名・著者名・訳者名）の後の最初の空行までをスキップ
        idx = 0
        while idx < n and not raw_lines[idx].strip():
            idx += 1
        while idx < n and raw_lines[idx].strip():
            idx += 1
        body_start = idx

    # 2. 末尾書誌情報（底本：、入力：、校正：、青空文庫作成ファイル：等）の除去
    # 末尾 80 行の範囲で最初に書誌情報ヘッダが現れた位置から後ろを切り落とす
    body_end = n
    footer_scan_start = max(body_start, n - 80)
    for i in range(footer_scan_start, n):
        stripped = raw_lines[i].strip()
        if _AOZORA_FOOTER_RE.match(stripped):
            body_end = i
            break

    return raw_lines[body_start:body_end]


def extract_aozora_entry(text: str, original: str) -> dict | None:
    """青空文庫のテキスト 1 件からメタデータ・ルビ・注記を除去し、本文のみのエントリ {"original", "lines"} を返す。

    変換不能な外字や未処理の注記記号（`［＃`, `《`, `》`, `｜`, `※` 等）が残った場合は None を返し、学習対象から除外する。
    """
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    body_lines = _strip_aozora_header_and_footer(raw_lines)
    if not body_lines:
        return None

    cleaned_lines: list[str] = []
    for line in body_lines:
        if not line.strip():
            continue
        # 1. 外字 `※［＃...］` を Unicode 文字へ変換
        s = resolve_aozora_gaiji(line)
        # 2. 残りの入力者注記 `［＃...］` を除去
        s = _AOZORA_NOTE_RE.sub("", s)
        # 3. ルビ `《...》` とルビ開始記号 `｜` を除去
        s = _AOZORA_RUBY_RE.sub("", s)
        s = s.replace("｜", "")
        s = s.strip()
        if not s:
            continue

        # 4. 変換後に注記やルビ・未解決外字が残っていないか検査
        if any(marker in s for marker in _AOZORA_LEFTOVER_MARKERS):
            return None

        cleaned_lines.append(s)

    if not cleaned_lines:
        return None

    return {
        "original": original,
        "lines": cleaned_lines,
    }


def extract_aozora_file(
    in_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    dataset_prefix: str = "aozora",
) -> dict | None:
    """青空文庫の .txt ファイル 1 件を読み込み、UTF-8 変換・本文抽出を行って 1 エントリ {"original", "lines"} を返す。"""
    path = Path(in_path)
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return None

    text = decode_aozora_bytes(raw_bytes)
    if text is None:
        return None

    if base_dir is not None:
        try:
            rel_path = path.relative_to(Path(base_dir)).as_posix()
        except ValueError:
            rel_path = path.name
    else:
        rel_path = path.name

    original = f"{dataset_prefix}/{rel_path}" if dataset_prefix else rel_path
    return extract_aozora_entry(text, original)


def collect_aozora_entries(
    aozora_dir: str | Path,
    *,
    dataset_prefix: str = "aozora",
) -> tuple[list[dict], int]:
    """青空文庫ディレクトリ（samples/aozora 等）配下の作品 *.txt を列挙し、(有効エントリ一覧, 除外ファイル数) を返す。"""
    dir_path = Path(aozora_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"aozora ディレクトリが存在しません: {dir_path}")

    entries: list[dict] = []
    skipped_count = 0
    for in_path in sorted(dir_path.rglob("*.txt")):
        if not in_path.is_file():
            continue
        entry = extract_aozora_file(
            in_path,
            base_dir=dir_path,
            dataset_prefix=dataset_prefix,
        )
        if entry is not None:
            entries.append(entry)
        else:
            skipped_count += 1
    return entries, skipped_count


def build_slim_dataset(
    out_slim_file: str | Path,
    *,
    chabsa_dir: str | Path | None = None,
    livedoor_dir: str | Path | None = None,
    aozora_dir: str | Path | None = None,
) -> dict:
    """chABSA・livedoor・青空文庫（aozora）から共通フォーマット JSON を構築して UTF-8 で 1 ファイルに保存する。

    Args:
        out_slim_file: 出力先 JSON ファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/slim.json)
        chabsa_dir: chABSA origin JSON ディレクトリ (任意)
        livedoor_dir: livedoor コーパス text ディレクトリ (任意)
        aozora_dir: 青空文庫ディレクトリ (例: samples/aozora, 任意)

    Returns:
        統計情報の辞書
    """
    if chabsa_dir is None and livedoor_dir is None and aozora_dir is None:
        raise ValueError(
            "chabsa_dir, livedoor_dir, aozora_dir の少なくとも 1 つを指定してください"
        )

    out_path = Path(out_slim_file)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "slim.json"

    chabsa_entries: list[dict] = []
    livedoor_entries: list[dict] = []
    aozora_entries: list[dict] = []
    aozora_skipped = 0

    if chabsa_dir is not None:
        chabsa_entries = collect_chabsa_entries(chabsa_dir)
    if livedoor_dir is not None:
        livedoor_entries = collect_livedoor_entries(livedoor_dir)
    if aozora_dir is not None:
        aozora_entries, aozora_skipped = collect_aozora_entries(aozora_dir)

    all_entries = [*chabsa_entries, *livedoor_entries, *aozora_entries]
    payload = {
        "sentences": all_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    chabsa_lines = sum(len(e["lines"]) for e in chabsa_entries)
    livedoor_lines = sum(len(e["lines"]) for e in livedoor_entries)
    aozora_lines = sum(len(e["lines"]) for e in aozora_entries)
    total_lines = chabsa_lines + livedoor_lines + aozora_lines

    return {
        "out_slim_file": str(out_path),
        "num_documents": len(all_entries),
        "num_chabsa_documents": len(chabsa_entries),
        "num_livedoor_documents": len(livedoor_entries),
        "num_aozora_documents": len(aozora_entries),
        "num_aozora_skipped_documents": aozora_skipped,
        "num_sentences": total_lines,
        "num_chabsa_sentences": chabsa_lines,
        "num_livedoor_sentences": livedoor_lines,
        "num_aozora_sentences": aozora_lines,
    }


# 後方互換用ヘルパー
def slim_document(doc: dict, original: str = "chabsa/unknown.json") -> dict:
    """chABSA ドキュメント 1 件を共通フォーマットに変換する。"""
    entry = extract_chabsa_entry(doc, original)
    return {"sentences": [entry] if entry is not None else []}


def slim_file(in_path: str | Path, out_path: str | Path) -> dict:
    """chABSA JSON ファイル 1 件を共通フォーマットに変換して書き出す。"""
    in_path = Path(in_path)
    out_path = Path(out_path)
    entry = extract_chabsa_file(in_path)
    slimmed = {"sentences": [entry] if entry is not None else []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(slimmed, f, ensure_ascii=False, indent=2)
    return slimmed


def slim_directory(
    chabsa_dir: str | Path,
    out_dir_or_file: str | Path,
    *,
    livedoor_dir: str | Path | None = None,
    aozora_dir: str | Path | None = None,
) -> Path:
    """chABSA (および任意の livedoor / aozora) ディレクトリから共通フォーマット JSON ファイルを作成してそのパスを返す。"""
    stats = build_slim_dataset(
        out_dir_or_file,
        chabsa_dir=chabsa_dir,
        livedoor_dir=livedoor_dir,
        aozora_dir=aozora_dir,
    )
    return Path(stats["out_slim_file"])
