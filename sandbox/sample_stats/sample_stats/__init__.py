"""chABSA、livedoor ニュースコーパス、および青空文庫（aozora）のオリジナルデータ（正規化前・SentencePiece前）の統計集計モジュール。

集計項目:
- 文書（ファイル）単位の統計: 文書数、1文書あたりの行数・文字数（合計・平均・標準偏差・最小・中央値・パーセンタイル・最大）
- 文（行）単位の統計: 総行数（文数）、1行あたりの文字数（合計・平均・標準偏差・最小・中央値・パーセンタイル・最大・長さ分布）
- 文字種・表記特性の統計: 総文字数、ユニーク文字数（異なり文字数）、文字種別（ひらがな・全角カナ・半角カナ・漢字・半角英数・全角英数・空白・記号等）の内訳、NFKC非正規文字の割合
- livedoor カテゴリ別の内訳統計
- 青空文庫（aozora）の文字コード内訳・外字変換数・ルビ/注記除去数・スキップ文書内訳
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

__all__ = [
    "EXCLUDED_LIVEDOOR_FILENAMES",
    "compute_numeric_summary",
    "compute_length_histogram",
    "analyze_char_types",
    "analyze_chabsa_directory",
    "analyze_livedoor_directory",
    "analyze_aozora_directory",
    "analyze_corpora",
    "format_summary_markdown",
]

EXCLUDED_LIVEDOOR_FILENAMES = frozenset({"LICENSE.TXT", "README.TXT", "CHANGES.TXT"})

# 青空文庫の正規表現パターン（impl_gen3 chabsa_slim と同一基準）
_AOZORA_SEPARATOR_RE = re.compile(r"^-{15,}\s*$")
_AOZORA_FOOTER_RE = re.compile(
    r"^(?:底本[：:]|初出[：:]|入力[：:]|校正[：:]|翻訳[：:]|青空文庫作成ファイル[：:]|このファイルは、インターネットの図書館、青空文庫)"
)
_AOZORA_GAIJI_NOTE_RE = re.compile(r"※［＃([^］]+)］")
_AOZORA_UNICODE_RE = re.compile(r"(?:U\+|UCS-)([0-9A-Fa-f]{4,6})")
_AOZORA_JIS_MENKUTEN_RE = re.compile(r"(?:第[1-4]水準\s*)?([12])-(\d{1,2})-(\d{1,2})\b")
_AOZORA_NOTE_RE = re.compile(r"［＃[^］]*］")
_AOZORA_RUBY_RE = re.compile(r"《[^》]*》")
_AOZORA_LEFTOVER_MARKERS = ("［＃", "《", "》", "｜", "※", "〖", "〗")

HISTOGRAM_BINS: list[tuple[str, int, int | None]] = [
    ("1-10文字", 1, 10),
    ("11-25文字", 11, 25),
    ("26-50文字", 26, 50),
    ("51-100文字", 51, 100),
    ("101-200文字", 101, 200),
    ("201-500文字", 201, 500),
    ("501文字以上", 501, None),
]


def _percentile(sorted_vals: list[int], q: float) -> float:
    """昇順ソート済み整数リストからパーセンタイル値 (0.0 <= q <= 100.0) を線形補間で計算する。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = (n - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 2)


def compute_numeric_summary(values: list[int]) -> dict:
    """整数値のリスト（文字数や行数など）から基本統計量（件数・合計・平均・標準偏差・最小・最大・分位数）を算出する。"""
    n = len(values)
    if n == 0:
        return {
            "count": 0,
            "total": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "p5": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }

    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    mean = total / n
    variance = sum((v - mean) ** 2 for v in sorted_vals) / n
    std = math.sqrt(variance)

    return {
        "count": n,
        "total": total,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": sorted_vals[0],
        "p5": _percentile(sorted_vals, 5.0),
        "p25": _percentile(sorted_vals, 25.0),
        "median": _percentile(sorted_vals, 50.0),
        "p75": _percentile(sorted_vals, 75.0),
        "p90": _percentile(sorted_vals, 90.0),
        "p95": _percentile(sorted_vals, 95.0),
        "p99": _percentile(sorted_vals, 99.0),
        "max": sorted_vals[-1],
    }


def compute_length_histogram(lengths: list[int]) -> list[dict]:
    """文字数リストから文字数帯別の件数・構成比ヒストグラムを作成する。"""
    total = len(lengths)
    result: list[dict] = []
    for label, lo, hi in HISTOGRAM_BINS:
        if hi is None:
            cnt = sum(1 for v in lengths if v >= lo)
        else:
            cnt = sum(1 for v in lengths if lo <= v <= hi)
        ratio = round((cnt / total) * 100.0, 2) if total > 0 else 0.0
        result.append(
            {
                "range": label,
                "count": cnt,
                "percentage": ratio,
            }
        )
    return result


def _classify_char(ch: str) -> str:
    """1文字を文字種カテゴリに分類する（正規化前のオリジナル表記揺れを判別可能にする）。"""
    cp = ord(ch)
    # 空白類
    if ch == "\u3000":
        return "fullwidth_space"
    if ch in (" ", "\t", "\n", "\r", "\v", "\f") or unicodedata.category(ch).startswith("Z"):
        return "ascii_or_other_space"

    # 半角英数字・記号 (ASCII)
    if 0x0030 <= cp <= 0x0039:
        return "ascii_digit"
    if 0x0041 <= cp <= 0x005A:
        return "ascii_upper"
    if 0x0061 <= cp <= 0x007A:
        return "ascii_lower"
    if 0x0021 <= cp <= 0x007E:
        return "ascii_symbol"

    # 全角英数字
    if 0xFF10 <= cp <= 0xFF19:
        return "fullwidth_digit"
    if 0xFF21 <= cp <= 0xFF3A:
        return "fullwidth_upper"
    if 0xFF41 <= cp <= 0xFF5A:
        return "fullwidth_lower"

    # 半角カタカナ (U+FF65 - U+FF9F)
    if 0xFF65 <= cp <= 0xFF9F:
        return "halfwidth_katakana"

    # ひらがな (U+3040 - U+309F)
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"

    # 全角カタカナ (U+30A0 - U+30FF, U+31F0 - U+31FF)
    if (0x30A0 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF):
        return "fullwidth_katakana"

    # 漢字 (CJK統合漢字・拡張・互換漢字・々〇等)
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2FA1F)
        or (0xF900 <= cp <= 0xFAFF)
        or cp in (0x3005, 0x3007)
    ):
        return "kanji"

    # 句読点・全角記号類
    cat = unicodedata.category(ch)
    if cat.startswith(("P", "S")):
        return "cjk_or_other_symbol"

    return "other"


CHAR_CATEGORY_LABELS: dict[str, str] = {
    "kanji": "漢字",
    "hiragana": "ひらがな",
    "fullwidth_katakana": "全角カタカナ",
    "halfwidth_katakana": "半角カタカナ",
    "ascii_upper": "半角英字（大文字）",
    "ascii_lower": "半角英字（小文字）",
    "fullwidth_upper": "全角英字（大文字）",
    "fullwidth_lower": "全角英字（小文字）",
    "ascii_digit": "半角数字",
    "fullwidth_digit": "全角数字",
    "ascii_symbol": "半角記号",
    "cjk_or_other_symbol": "全角句読点・記号",
    "fullwidth_space": "全角スペース",
    "ascii_or_other_space": "半角スペース・空白",
    "other": "その他文字",
}


def analyze_char_types(lines: Iterable[str]) -> dict:
    """文字列イテラブルから文字種分布・ユニーク文字数・NFKC非正規文字数を集計する。"""
    char_counter: Counter[str] = Counter()
    for line in lines:
        char_counter.update(line)

    total_chars = sum(char_counter.values())
    unique_chars = len(char_counter)

    cat_total: Counter[str] = Counter()
    cat_unique: Counter[str] = Counter()
    nfkc_diff_total = 0
    nfkc_diff_unique = 0
    upper_total = 0

    for ch, count in char_counter.items():
        cat = _classify_char(ch)
        cat_total[cat] += count
        cat_unique[cat] += 1
        if unicodedata.normalize("NFKC", ch) != ch:
            nfkc_diff_total += count
            nfkc_diff_unique += 1
        if ch.isupper():
            upper_total += count

    categories_breakdown: dict[str, dict] = {}
    for cat_key, label in CHAR_CATEGORY_LABELS.items():
        c_tot = cat_total.get(cat_key, 0)
        c_uni = cat_unique.get(cat_key, 0)
        categories_breakdown[cat_key] = {
            "label": label,
            "total_chars": c_tot,
            "percentage": round((c_tot / total_chars) * 100.0, 2) if total_chars > 0 else 0.0,
            "unique_chars": c_uni,
        }

    return {
        "total_chars": total_chars,
        "unique_chars": unique_chars,
        "nfkc_convertible_chars": {
            "total_chars": nfkc_diff_total,
            "percentage": round((nfkc_diff_total / total_chars) * 100.0, 2) if total_chars > 0 else 0.0,
            "unique_chars": nfkc_diff_unique,
        },
        "uppercase_chars": {
            "total_chars": upper_total,
            "percentage": round((upper_total / total_chars) * 100.0, 2) if total_chars > 0 else 0.0,
        },
        "categories": categories_breakdown,
    }


def _excerpt(text: str, max_len: int = 60) -> str:
    """長すぎる文字列を表示用に省略する。"""
    cleaned = text.replace("\n", "\\n")
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len] + "..."


def analyze_chabsa_directory(chabsa_dir: str | Path) -> dict:
    """chABSA のオリジナル JSON ディレクトリ (`samples/chabsa_data/data`) を集計する。"""
    dir_path = Path(chabsa_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"chABSA ディレクトリが見つかりません: {dir_path}")

    json_files = sorted(dir_path.glob("*.json"))

    doc_sentence_counts: list[int] = []
    doc_line_counts: list[int] = []
    doc_char_counts: list[int] = []

    raw_sentence_lengths: list[int] = []
    line_lengths: list[int] = []
    all_lines: list[str] = []

    empty_files = 0
    empty_sentences = 0
    multiline_sentences = 0

    min_doc_info: dict | None = None
    max_doc_info: dict | None = None
    min_line_info: dict | None = None
    max_line_info: dict | None = None

    for path in json_files:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)

        raw_sents = doc.get("sentences", [])
        valid_raw_sent_count = 0
        file_lines: list[str] = []

        for s in raw_sents:
            text = s.get("sentence", "")
            if not text or not text.strip():
                empty_sentences += 1
                continue
            valid_raw_sent_count += 1
            raw_sentence_lengths.append(len(text))

            sublines = [
                sub.strip()
                for sub in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                if sub.strip()
            ]
            if len(sublines) > 1:
                multiline_sentences += 1

            for line in sublines:
                file_lines.append(line)
                l_len = len(line)
                line_lengths.append(l_len)
                if min_line_info is None or l_len < min_line_info["length"]:
                    min_line_info = {
                        "file": path.name,
                        "length": l_len,
                        "excerpt": _excerpt(line),
                    }
                if max_line_info is None or l_len > max_line_info["length"]:
                    max_line_info = {
                        "file": path.name,
                        "length": l_len,
                        "excerpt": _excerpt(line),
                    }

        if not file_lines:
            empty_files += 1
            continue

        doc_chars = sum(len(line) for line in file_lines)
        doc_sentence_counts.append(valid_raw_sent_count)
        doc_line_counts.append(len(file_lines))
        doc_char_counts.append(doc_chars)
        all_lines.extend(file_lines)

        if min_doc_info is None or doc_chars < min_doc_info["chars"]:
            min_doc_info = {
                "file": path.name,
                "lines": len(file_lines),
                "chars": doc_chars,
            }
        if max_doc_info is None or doc_chars > max_doc_info["chars"]:
            max_doc_info = {
                "file": path.name,
                "lines": len(file_lines),
                "chars": doc_chars,
            }

    return {
        "corpus": "chABSA",
        "directory": str(dir_path.resolve()),
        "files": {
            "total_json_files": len(json_files),
            "valid_documents": len(doc_line_counts),
            "empty_documents": empty_files,
            "empty_sentence_entries": empty_sentences,
            "multiline_sentence_entries": multiline_sentences,
        },
        "document_stats": {
            "sentences_per_doc": compute_numeric_summary(doc_sentence_counts),
            "lines_per_doc": compute_numeric_summary(doc_line_counts),
            "chars_per_doc": compute_numeric_summary(doc_char_counts),
            "min_doc": min_doc_info,
            "max_doc": max_doc_info,
        },
        "sentence_and_line_stats": {
            "raw_sentence_chars": compute_numeric_summary(raw_sentence_lengths),
            "line_chars": compute_numeric_summary(line_lengths),
            "line_length_histogram": compute_length_histogram(line_lengths),
            "min_line": min_line_info,
            "max_line": max_line_info,
        },
        "char_stats": analyze_char_types(all_lines),
        "_raw_lines": all_lines,
        "_doc_line_counts": doc_line_counts,
        "_doc_char_counts": doc_char_counts,
        "_line_lengths": line_lengths,
    }


def analyze_livedoor_directory(livedoor_dir: str | Path) -> dict:
    """livedoor ニュースコーパスのオリジナルテキストディレクトリ (`samples/livedoor/text`) を集計する。"""
    dir_path = Path(livedoor_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"livedoor ディレクトリが見つかりません: {dir_path}")

    all_txt_files = sorted(p for p in dir_path.rglob("*.txt") if p.is_file())
    excluded_files: list[str] = []
    article_files: list[Path] = []

    for p in all_txt_files:
        if p.name.upper() in EXCLUDED_LIVEDOOR_FILENAMES:
            excluded_files.append(str(p.relative_to(dir_path).as_posix()))
        else:
            article_files.append(p)

    doc_line_counts: list[int] = []
    doc_body_line_counts: list[int] = []
    doc_empty_line_counts: list[int] = []
    doc_char_counts: list[int] = []
    doc_raw_file_char_counts: list[int] = []

    title_lengths: list[int] = []
    body_line_lengths: list[int] = []
    all_line_lengths: list[int] = []
    all_lines: list[str] = []

    min_doc_info: dict | None = None
    max_doc_info: dict | None = None
    min_line_info: dict | None = None
    max_line_info: dict | None = None

    # カテゴリ別集計用
    cat_buffers: dict[str, dict[str, list]] = {}

    for path in article_files:
        rel_path = path.relative_to(dir_path).as_posix()
        category = path.parent.name if path.parent != dir_path else "root"

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        doc_raw_file_char_counts.append(len(raw_text))

        raw_lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if len(raw_lines) < 3:
            continue

        # 3行目: タイトル、4行目以降: 本文
        content_raw_lines = raw_lines[2:]
        empty_line_cnt = sum(1 for l in content_raw_lines if not l.strip())
        valid_lines = [l.strip() for l in content_raw_lines if l.strip()]
        if not valid_lines:
            continue

        title_str = raw_lines[2].strip()
        if title_str:
            title_lengths.append(len(title_str))
            body_lines = [l.strip() for l in raw_lines[3:] if l.strip()]
        else:
            body_lines = valid_lines

        for bl in body_lines:
            body_line_lengths.append(len(bl))

        for line in valid_lines:
            l_len = len(line)
            all_line_lengths.append(l_len)
            if min_line_info is None or l_len < min_line_info["length"]:
                min_line_info = {
                    "file": rel_path,
                    "length": l_len,
                    "excerpt": _excerpt(line),
                }
            if max_line_info is None or l_len > max_line_info["length"]:
                max_line_info = {
                    "file": rel_path,
                    "length": l_len,
                    "excerpt": _excerpt(line),
                }

        doc_chars = sum(len(l) for l in valid_lines)
        doc_line_counts.append(len(valid_lines))
        doc_body_line_counts.append(len(body_lines))
        doc_empty_line_counts.append(empty_line_cnt)
        doc_char_counts.append(doc_chars)
        all_lines.extend(valid_lines)

        if min_doc_info is None or doc_chars < min_doc_info["chars"]:
            min_doc_info = {
                "file": rel_path,
                "lines": len(valid_lines),
                "chars": doc_chars,
            }
        if max_doc_info is None or doc_chars > max_doc_info["chars"]:
            max_doc_info = {
                "file": rel_path,
                "lines": len(valid_lines),
                "chars": doc_chars,
            }

        if category not in cat_buffers:
            cat_buffers[category] = {
                "doc_lines": [],
                "doc_chars": [],
                "line_lengths": [],
            }
        cat_buffers[category]["doc_lines"].append(len(valid_lines))
        cat_buffers[category]["doc_chars"].append(doc_chars)
        cat_buffers[category]["line_lengths"].extend(len(l) for l in valid_lines)

    category_stats: dict[str, dict] = {}
    for cat_name in sorted(cat_buffers.keys()):
        buf = cat_buffers[cat_name]
        category_stats[cat_name] = {
            "num_documents": len(buf["doc_chars"]),
            "num_lines": len(buf["line_lengths"]),
            "total_chars": sum(buf["doc_chars"]),
            "chars_per_doc": compute_numeric_summary(buf["doc_chars"]),
            "lines_per_doc": compute_numeric_summary(buf["doc_lines"]),
            "chars_per_line": compute_numeric_summary(buf["line_lengths"]),
        }

    return {
        "corpus": "livedoor",
        "directory": str(dir_path.resolve()),
        "files": {
            "total_txt_files": len(all_txt_files),
            "excluded_files_count": len(excluded_files),
            "excluded_files": excluded_files,
            "valid_documents": len(doc_line_counts),
            "num_categories": len(category_stats),
        },
        "document_stats": {
            "lines_per_doc": compute_numeric_summary(doc_line_counts),
            "body_lines_per_doc": compute_numeric_summary(doc_body_line_counts),
            "empty_lines_per_doc": compute_numeric_summary(doc_empty_line_counts),
            "chars_per_doc": compute_numeric_summary(doc_char_counts),
            "raw_file_chars_per_doc": compute_numeric_summary(doc_raw_file_char_counts),
            "min_doc": min_doc_info,
            "max_doc": max_doc_info,
        },
        "sentence_and_line_stats": {
            "line_chars": compute_numeric_summary(all_line_lengths),
            "title_chars": compute_numeric_summary(title_lengths),
            "body_line_chars": compute_numeric_summary(body_line_lengths),
            "line_length_histogram": compute_length_histogram(all_line_lengths),
            "min_line": min_line_info,
            "max_line": max_line_info,
        },
        "category_stats": category_stats,
        "char_stats": analyze_char_types(all_lines),
        "_raw_lines": all_lines,
        "_doc_line_counts": doc_line_counts,
        "_doc_char_counts": doc_char_counts,
        "_line_lengths": all_line_lengths,
    }


def _decode_aozora_bytes_with_encoding(raw_bytes: bytes) -> tuple[str | None, str | None]:
    """青空文庫テキストのバイト列をデコードし、(デコード済み文字列, エンコーディング名) を返す。"""
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None


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


def _resolve_aozora_gaiji_with_count(text: str) -> tuple[str, int, int]:
    """青空文庫の外字注記 `※［＃...］` を Unicode 文字へ置換し、(置換後文字列, 解決成功数, 未解決数) を返す。"""
    resolved = 0
    unresolved = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal resolved, unresolved
        note = match.group(1)
        m_uni = _AOZORA_UNICODE_RE.search(note)
        if m_uni:
            try:
                ch = chr(int(m_uni.group(1), 16))
                resolved += 1
                return ch
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
                resolved += 1
                return ch
        unresolved += 1
        return "※"

    out = _AOZORA_GAIJI_NOTE_RE.sub(_replace, text)
    return out, resolved, unresolved


def _strip_aozora_header_and_footer(raw_lines: list[str]) -> list[str]:
    """青空文庫テキストから冒頭（作品名・著者名・記号説明区切り線）と末尾（底本等の書誌情報）を除去する。"""
    n = len(raw_lines)
    if n == 0:
        return []

    header_scan_limit = min(n, 80)
    sep_indices = [
        i for i in range(header_scan_limit) if _AOZORA_SEPARATOR_RE.match(raw_lines[i])
    ]
    if len(sep_indices) >= 2:
        body_start = sep_indices[1] + 1
    else:
        idx = 0
        while idx < n and not raw_lines[idx].strip():
            idx += 1
        while idx < n and raw_lines[idx].strip():
            idx += 1
        body_start = idx

    body_end = n
    footer_scan_start = max(body_start, n - 80)
    for i in range(footer_scan_start, n):
        stripped = raw_lines[i].strip()
        if _AOZORA_FOOTER_RE.match(stripped):
            body_end = i
            break

    return raw_lines[body_start:body_end]


def analyze_aozora_directory(
    aozora_dir: str | Path,
    *,
    max_files: int | None = None,
) -> dict:
    """青空文庫（aozora）のオリジナルテキストディレクトリ (`samples/aozora`) を集計する。

    Args:
        aozora_dir: 青空文庫テキストのルートディレクトリ（`samples/aozora` または `samples/aozora/cards/000005` 等）
        max_files: 集計対象の最大ファイル数（None の場合は全ファイルを対象）
    """
    dir_path = Path(aozora_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"aozora ディレクトリが見つかりません: {dir_path}")

    all_txt_files = sorted(p for p in dir_path.rglob("*.txt") if p.is_file())
    target_files = all_txt_files[:max_files] if max_files is not None else all_txt_files

    encoding_counts: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()

    total_gaiji_resolved = 0
    total_notes_removed = 0
    total_ruby_removed = 0

    doc_line_counts: list[int] = []
    doc_raw_line_counts: list[int] = []
    doc_empty_line_counts: list[int] = []
    doc_char_counts: list[int] = []
    doc_raw_file_char_counts: list[int] = []

    all_line_lengths: list[int] = []
    all_lines: list[str] = []

    min_doc_info: dict | None = None
    max_doc_info: dict | None = None
    min_line_info: dict | None = None
    max_line_info: dict | None = None

    for path in target_files:
        rel_path = path.relative_to(dir_path).as_posix()
        try:
            raw_bytes = path.read_bytes()
        except OSError:
            skipped_reasons["read_error"] += 1
            continue

        raw_text, enc = _decode_aozora_bytes_with_encoding(raw_bytes)
        if raw_text is None or enc is None:
            skipped_reasons["decode_error"] += 1
            continue

        encoding_counts[enc] += 1
        raw_lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        body_lines = _strip_aozora_header_and_footer(raw_lines)
        if not body_lines:
            skipped_reasons["empty_body"] += 1
            continue

        cleaned_lines: list[str] = []
        doc_gaiji = 0
        doc_notes = 0
        doc_ruby = 0
        has_leftover = False
        empty_line_cnt = 0

        for line in body_lines:
            if not line.strip():
                empty_line_cnt += 1
                continue
            s, g_ok, g_ng = _resolve_aozora_gaiji_with_count(line)
            if g_ng > 0:
                has_leftover = True
                break
            doc_gaiji += g_ok

            s, n_cnt = _AOZORA_NOTE_RE.subn("", s)
            doc_notes += n_cnt

            s, r_cnt = _AOZORA_RUBY_RE.subn("", s)
            doc_ruby += r_cnt

            s = s.replace("｜", "").strip()
            if not s:
                continue

            if any(marker in s for marker in _AOZORA_LEFTOVER_MARKERS):
                has_leftover = True
                break

            cleaned_lines.append(s)

        if has_leftover:
            skipped_reasons["unresolved_markup_or_gaiji"] += 1
            continue
        if not cleaned_lines:
            skipped_reasons["empty_body"] += 1
            continue

        total_gaiji_resolved += doc_gaiji
        total_notes_removed += doc_notes
        total_ruby_removed += doc_ruby

        doc_raw_file_char_counts.append(len(raw_text))
        doc_raw_line_counts.append(len(raw_lines))
        doc_empty_line_counts.append(empty_line_cnt)

        for line in cleaned_lines:
            l_len = len(line)
            all_line_lengths.append(l_len)
            if min_line_info is None or l_len < min_line_info["length"]:
                min_line_info = {
                    "file": rel_path,
                    "length": l_len,
                    "excerpt": _excerpt(line),
                }
            if max_line_info is None or l_len > max_line_info["length"]:
                max_line_info = {
                    "file": rel_path,
                    "length": l_len,
                    "excerpt": _excerpt(line),
                }

        doc_chars = sum(len(l) for l in cleaned_lines)
        doc_line_counts.append(len(cleaned_lines))
        doc_char_counts.append(doc_chars)
        all_lines.extend(cleaned_lines)

        if min_doc_info is None or doc_chars < min_doc_info["chars"]:
            min_doc_info = {
                "file": rel_path,
                "lines": len(cleaned_lines),
                "chars": doc_chars,
            }
        if max_doc_info is None or doc_chars > max_doc_info["chars"]:
            max_doc_info = {
                "file": rel_path,
                "lines": len(cleaned_lines),
                "chars": doc_chars,
            }

    total_skipped = sum(skipped_reasons.values())

    return {
        "corpus": "aozora",
        "directory": str(dir_path.resolve()),
        "files": {
            "total_txt_files": len(all_txt_files),
            "analyzed_txt_files": len(target_files),
            "valid_documents": len(doc_line_counts),
            "skipped_documents": total_skipped,
            "skipped_reasons": dict(skipped_reasons),
            "encodings": {
                "utf-8-sig": encoding_counts.get("utf-8-sig", 0),
                "cp932": encoding_counts.get("cp932", 0),
            },
            "markup_cleaning": {
                "gaiji_resolved_count": total_gaiji_resolved,
                "notes_removed_count": total_notes_removed,
                "ruby_removed_count": total_ruby_removed,
            },
        },
        "document_stats": {
            "lines_per_doc": compute_numeric_summary(doc_line_counts),
            "raw_lines_per_doc": compute_numeric_summary(doc_raw_line_counts),
            "empty_lines_per_doc": compute_numeric_summary(doc_empty_line_counts),
            "chars_per_doc": compute_numeric_summary(doc_char_counts),
            "raw_file_chars_per_doc": compute_numeric_summary(doc_raw_file_char_counts),
            "min_doc": min_doc_info,
            "max_doc": max_doc_info,
        },
        "sentence_and_line_stats": {
            "line_chars": compute_numeric_summary(all_line_lengths),
            "line_length_histogram": compute_length_histogram(all_line_lengths),
            "min_line": min_line_info,
            "max_line": max_line_info,
        },
        "char_stats": analyze_char_types(all_lines),
        "_raw_lines": all_lines,
        "_doc_line_counts": doc_line_counts,
        "_doc_char_counts": doc_char_counts,
        "_line_lengths": all_line_lengths,
    }


def _strip_private_keys(d: dict) -> dict:
    """内部受け渡し用の `_` 始まりキーを除外した辞書を返す。"""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def analyze_corpora(
    *,
    chabsa_dir: str | Path | None = None,
    livedoor_dir: str | Path | None = None,
    aozora_dir: str | Path | None = None,
    aozora_max_files: int | None = None,
) -> dict:
    """chABSA、livedoor、および青空文庫（aozora）コーパスをそれぞれ集計し、個別統計および合算統計を返す。"""
    chabsa_res = analyze_chabsa_directory(chabsa_dir) if chabsa_dir is not None else None
    livedoor_res = analyze_livedoor_directory(livedoor_dir) if livedoor_dir is not None else None
    aozora_res = (
        analyze_aozora_directory(aozora_dir, max_files=aozora_max_files)
        if aozora_dir is not None
        else None
    )

    combined_lines: list[str] = []
    combined_doc_lines: list[int] = []
    combined_doc_chars: list[int] = []
    combined_line_lengths: list[int] = []

    active_names: list[str] = []
    for name, res in (("chABSA", chabsa_res), ("livedoor", livedoor_res), ("aozora", aozora_res)):
        if res is not None:
            active_names.append(name)
            combined_lines.extend(res["_raw_lines"])
            combined_doc_lines.extend(res["_doc_line_counts"])
            combined_doc_chars.extend(res["_doc_char_counts"])
            combined_line_lengths.extend(res["_line_lengths"])

    combined_label = f"combined ({' + '.join(active_names)})" if active_names else "combined"

    combined_summary = {
        "corpus": combined_label,
        "valid_documents": len(combined_doc_lines),
        "document_stats": {
            "lines_per_doc": compute_numeric_summary(combined_doc_lines),
            "chars_per_doc": compute_numeric_summary(combined_doc_chars),
        },
        "sentence_and_line_stats": {
            "line_chars": compute_numeric_summary(combined_line_lengths),
            "line_length_histogram": compute_length_histogram(combined_line_lengths),
        },
        "char_stats": analyze_char_types(combined_lines),
    }

    return {
        "chabsa": _strip_private_keys(chabsa_res) if chabsa_res else None,
        "livedoor": _strip_private_keys(livedoor_res) if livedoor_res else None,
        "aozora": _strip_private_keys(aozora_res) if aozora_res else None,
        "combined": combined_summary,
    }


def format_summary_markdown(report: dict) -> str:
    """集計結果辞書を見やすい Markdown レポート文字列に整形する。"""
    lines: list[str] = []
    lines.append("# コーパス オリジナルデータ統計レポート（正規化前・SentencePiece前）")
    lines.append("")

    chabsa = report.get("chabsa")
    livedoor = report.get("livedoor")
    aozora = report.get("aozora")
    combined = report.get("combined")

    # 1. 全体サマリ比較表
    lines.append("## 1. 全体比較サマリ（chABSA vs livedoor vs aozora vs 合計）")
    lines.append("")
    lines.append("| 指標 | chABSA | livedoor | aozora | 合計 (Combined) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")

    def _fmt_int(val: int | float | None) -> str:
        return f"{int(val):,}" if val is not None else "-"

    def _fmt_flt(val: float | int | None) -> str:
        return f"{float(val):,.2f}" if val is not None else "-"

    c_docs = chabsa["files"]["valid_documents"] if chabsa else None
    l_docs = livedoor["files"]["valid_documents"] if livedoor else None
    a_docs = aozora["files"]["valid_documents"] if aozora else None
    t_docs = combined["valid_documents"] if combined else 0

    c_l_stat = chabsa["sentence_and_line_stats"]["line_chars"] if chabsa else {}
    l_l_stat = livedoor["sentence_and_line_stats"]["line_chars"] if livedoor else {}
    a_l_stat = aozora["sentence_and_line_stats"]["line_chars"] if aozora else {}
    t_l_stat = combined["sentence_and_line_stats"]["line_chars"] if combined else {}

    c_d_chars = chabsa["document_stats"]["chars_per_doc"] if chabsa else {}
    l_d_chars = livedoor["document_stats"]["chars_per_doc"] if livedoor else {}
    a_d_chars = aozora["document_stats"]["chars_per_doc"] if aozora else {}
    t_d_chars = combined["document_stats"]["chars_per_doc"] if combined else {}

    c_d_lines = chabsa["document_stats"]["lines_per_doc"] if chabsa else {}
    l_d_lines = livedoor["document_stats"]["lines_per_doc"] if livedoor else {}
    a_d_lines = aozora["document_stats"]["lines_per_doc"] if aozora else {}
    t_d_lines = combined["document_stats"]["lines_per_doc"] if combined else {}

    c_c_stat = chabsa["char_stats"] if chabsa else {}
    l_c_stat = livedoor["char_stats"] if livedoor else {}
    a_c_stat = aozora["char_stats"] if aozora else {}
    t_c_stat = combined["char_stats"] if combined else {}

    def _fmt_min_med_max(stat: dict) -> str:
        if not stat or stat.get("count", 0) == 0:
            return "-"
        return f"{_fmt_int(stat.get('min'))} / {_fmt_flt(stat.get('median'))} / {_fmt_int(stat.get('max'))}"

    def _fmt_mean_std(stat: dict) -> str:
        if not stat or stat.get("count", 0) == 0:
            return "-"
        return f"{_fmt_flt(stat.get('mean'))} ± {_fmt_flt(stat.get('std'))}"

    def _fmt_quantiles(stat: dict) -> str:
        if not stat or stat.get("count", 0) == 0:
            return "-"
        return f"{_fmt_flt(stat.get('p25'))} / {_fmt_flt(stat.get('p75'))} / {_fmt_flt(stat.get('p95'))} / {_fmt_flt(stat.get('p99'))}"

    lines.append(
        f"| **有効文書（ファイル）数** | {_fmt_int(c_docs)} | {_fmt_int(l_docs)} | {_fmt_int(a_docs)} | **{_fmt_int(t_docs)}** |"
    )
    lines.append(
        f"| **総行数（文数）** | {_fmt_int(c_l_stat.get('count') if chabsa else None)} | {_fmt_int(l_l_stat.get('count') if livedoor else None)} | {_fmt_int(a_l_stat.get('count') if aozora else None)} | **{_fmt_int(t_l_stat.get('count'))}** |"
    )
    lines.append(
        f"| **総文字数（改行除く）** | {_fmt_int(c_l_stat.get('total') if chabsa else None)} | {_fmt_int(l_l_stat.get('total') if livedoor else None)} | {_fmt_int(a_l_stat.get('total') if aozora else None)} | **{_fmt_int(t_l_stat.get('total'))}** |"
    )
    lines.append(
        f"| **ユニーク文字数（異なり文字数）** | {_fmt_int(c_c_stat.get('unique_chars') if chabsa else None)} | {_fmt_int(l_c_stat.get('unique_chars') if livedoor else None)} | {_fmt_int(a_c_stat.get('unique_chars') if aozora else None)} | **{_fmt_int(t_c_stat.get('unique_chars'))}** |"
    )
    lines.append(
        f"| **1文書あたり平均行数** | {_fmt_flt(c_d_lines.get('mean') if chabsa else None)} | {_fmt_flt(l_d_lines.get('mean') if livedoor else None)} | {_fmt_flt(a_d_lines.get('mean') if aozora else None)} | **{_fmt_flt(t_d_lines.get('mean'))}** |"
    )
    lines.append(
        f"| **1文書あたり行数（最小 / 中央値 / 最大）** | {_fmt_min_med_max(c_d_lines)} | {_fmt_min_med_max(l_d_lines)} | {_fmt_min_med_max(a_d_lines)} | {_fmt_min_med_max(t_d_lines)} |"
    )
    lines.append(
        f"| **1文書あたり平均文字数** | {_fmt_flt(c_d_chars.get('mean') if chabsa else None)} | {_fmt_flt(l_d_chars.get('mean') if livedoor else None)} | {_fmt_flt(a_d_chars.get('mean') if aozora else None)} | **{_fmt_flt(t_d_chars.get('mean'))}** |"
    )
    lines.append(
        f"| **1文書あたり文字数（最小 / 中央値 / 最大）** | {_fmt_min_med_max(c_d_chars)} | {_fmt_min_med_max(l_d_chars)} | {_fmt_min_med_max(a_d_chars)} | {_fmt_min_med_max(t_d_chars)} |"
    )
    lines.append(
        f"| **1行（1文）あたり平均文字数 ± 標準偏差** | {_fmt_mean_std(c_l_stat)} | {_fmt_mean_std(l_l_stat)} | {_fmt_mean_std(a_l_stat)} | **{_fmt_mean_std(t_l_stat)}** |"
    )
    lines.append(
        f"| **1行あたり文字数（最小 / 中央値 / 最大）** | {_fmt_min_med_max(c_l_stat)} | {_fmt_min_med_max(l_l_stat)} | {_fmt_min_med_max(a_l_stat)} | {_fmt_min_med_max(t_l_stat)} |"
    )
    lines.append(
        f"| **1行あたり文字数（25% / 75% / 95% / 99%点）** | {_fmt_quantiles(c_l_stat)} | {_fmt_quantiles(l_l_stat)} | {_fmt_quantiles(a_l_stat)} | {_fmt_quantiles(t_l_stat)} |"
    )
    lines.append("")

    # 2. 文字数帯ヒストグラム比較
    lines.append("## 2. 1行（1文）あたりの文字数分布")
    lines.append("")
    lines.append("| 文字数帯 | chABSA 行数 (割合) | livedoor 行数 (割合) | aozora 行数 (割合) | 合計 行数 (割合) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    c_hist = {h["range"]: h for h in chabsa["sentence_and_line_stats"]["line_length_histogram"]} if chabsa else {}
    l_hist = {h["range"]: h for h in livedoor["sentence_and_line_stats"]["line_length_histogram"]} if livedoor else {}
    a_hist = {h["range"]: h for h in aozora["sentence_and_line_stats"]["line_length_histogram"]} if aozora else {}
    t_hist = {h["range"]: h for h in combined["sentence_and_line_stats"]["line_length_histogram"]} if combined else {}

    def _fmt_hist_cell(h_map: dict, label: str, active: bool) -> str:
        if not active:
            return "-"
        item = h_map.get(label, {"count": 0, "percentage": 0.0})
        return f"{item['count']:,} ({item['percentage']:.2f}%)"

    for label, _, _ in HISTOGRAM_BINS:
        lines.append(
            f"| **{label}** | {_fmt_hist_cell(c_hist, label, bool(chabsa))} | {_fmt_hist_cell(l_hist, label, bool(livedoor))} | {_fmt_hist_cell(a_hist, label, bool(aozora))} | **{_fmt_hist_cell(t_hist, label, True)}** |"
        )
    lines.append("")

    # 3. 文字種・表記揺れ統計（正規化前）
    lines.append("## 3. 正規化前の文字種・表記揺れ統計")
    lines.append("")
    lines.append("| 文字種カテゴリ | chABSA 文字数 (割合 / 異なり) | livedoor 文字数 (割合 / 異なり) | aozora 文字数 (割合 / 異なり) | 合計 文字数 (割合 / 異なり) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")

    def _fmt_cat_cell(c_stat: dict, cat_key: str, active: bool) -> str:
        if not active:
            return "-"
        item = c_stat.get("categories", {}).get(cat_key, {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
        return f"{item['total_chars']:,} ({item['percentage']:.2f}% / {item['unique_chars']:,}種)"

    for cat_key, label in CHAR_CATEGORY_LABELS.items():
        lines.append(
            f"| **{label}** | {_fmt_cat_cell(c_c_stat, cat_key, bool(chabsa))} | {_fmt_cat_cell(l_c_stat, cat_key, bool(livedoor))} | {_fmt_cat_cell(a_c_stat, cat_key, bool(aozora))} | **{_fmt_cat_cell(t_c_stat, cat_key, True)}** |"
        )

    def _fmt_nfkc_cell(c_stat: dict, active: bool) -> str:
        if not active:
            return "-"
        item = c_stat.get("nfkc_convertible_chars", {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
        return f"{item['total_chars']:,} ({item['percentage']:.2f}% / {item['unique_chars']:,}種)"

    lines.append(
        f"| **（内）NFKC正規化対象文字** | {_fmt_nfkc_cell(c_c_stat, bool(chabsa))} | {_fmt_nfkc_cell(l_c_stat, bool(livedoor))} | {_fmt_nfkc_cell(a_c_stat, bool(aozora))} | **{_fmt_nfkc_cell(t_c_stat, True)}** |"
    )
    lines.append("")

    # 4. livedoor カテゴリ別内訳
    if livedoor and livedoor.get("category_stats"):
        lines.append("## 4. livedoor ニュース カテゴリ別統計")
        lines.append("")
        lines.append("| カテゴリ | 記事数 | 総行数 | 総文字数 | 1記事平均文字数 (最小〜最大) | 1行平均文字数 (最小〜最大) |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for cat_name, cstat in livedoor["category_stats"].items():
            cpd = cstat["chars_per_doc"]
            cpl = cstat["chars_per_line"]
            lines.append(
                f"| `{cat_name}` | {cstat['num_documents']:,} | {cstat['num_lines']:,} | {cstat['total_chars']:,} | {cpd['mean']:,.1f} ({cpd['min']:,}〜{cpd['max']:,}) | {cpl['mean']:,.1f} ({cpl['min']:,}〜{cpl['max']:,}) |"
            )
        lines.append("")

    # 5. 青空文庫（aozora）抽出・クリーニング内訳
    if aozora:
        af = aozora["files"]
        enc = af["encodings"]
        mc = af["markup_cleaning"]
        lines.append("## 5. 青空文庫（aozora）ファイル・記法クリーニング統計")
        lines.append("")
        lines.append(f"- **対象ディレクトリ**: `{aozora['directory']}`")
        lines.append(
            f"- **ファイル数**: 発見 {af['total_txt_files']:,} 件 / 解析対象 {af['analyzed_txt_files']:,} 件 / 有効文書 **{af['valid_documents']:,}** 件 / スキップ（除外） **{af['skipped_documents']:,}** 件"
        )
        if af["skipped_reasons"]:
            reasons_str = ", ".join(f"`{k}`: {v:,}件" for k, v in af["skipped_reasons"].items())
            lines.append(f"  - スキップ理由内訳: {reasons_str}")
        lines.append(
            f"- **文字コード内訳**: Shift_JIS (`cp932`) **{enc.get('cp932', 0):,}** 件 / UTF-8 (`utf-8-sig`) **{enc.get('utf-8-sig', 0):,}** 件"
        )
        lines.append(
            f"- **青空文庫記法の変換・除去数（有効文書内）**: 外字 Unicode 変換 **{mc['gaiji_resolved_count']:,}** 箇所 / ルビ除去 **{mc['ruby_removed_count']:,}** 箇所 / 編集者注記除去 **{mc['notes_removed_count']:,}** 箇所"
        )
        lines.append("")

    # 6. 最小・最大サンプルの詳細
    lines.append("## 6. 最小・最大サンプルの詳細")
    lines.append("")
    if chabsa and chabsa["document_stats"]["min_doc"]:
        lines.append("### chABSA")
        lines.append(f"- **最短文書**: `{chabsa['document_stats']['min_doc']['file']}` ({chabsa['document_stats']['min_doc']['chars']:,} 文字, {chabsa['document_stats']['min_doc']['lines']:,} 行)")
        lines.append(f"- **最長文書**: `{chabsa['document_stats']['max_doc']['file']}` ({chabsa['document_stats']['max_doc']['chars']:,} 文字, {chabsa['document_stats']['max_doc']['lines']:,} 行)")
        lines.append(f"- **最短行**: `{chabsa['sentence_and_line_stats']['min_line']['file']}` ({chabsa['sentence_and_line_stats']['min_line']['length']:,} 文字) — `{chabsa['sentence_and_line_stats']['min_line']['excerpt']}`")
        lines.append(f"- **最長行**: `{chabsa['sentence_and_line_stats']['max_line']['file']}` ({chabsa['sentence_and_line_stats']['max_line']['length']:,} 文字) — `{chabsa['sentence_and_line_stats']['max_line']['excerpt']}`")
        lines.append("")
    if livedoor and livedoor["document_stats"]["min_doc"]:
        t_stat = livedoor["sentence_and_line_stats"]["title_chars"]
        b_stat = livedoor["sentence_and_line_stats"]["body_line_chars"]
        lines.append("### livedoor")
        lines.append(f"- **記事タイトル文字数**: 平均 {t_stat['mean']:,.2f} 文字 (最小 {t_stat['min']:,} / 中央値 {t_stat['median']:,.1f} / 最大 {t_stat['max']:,})")
        lines.append(f"- **記事本文1行文字数**: 平均 {b_stat['mean']:,.2f} 文字 (最小 {b_stat['min']:,} / 中央値 {b_stat['median']:,.1f} / 最大 {b_stat['max']:,})")
        lines.append(f"- **最短記事**: `{livedoor['document_stats']['min_doc']['file']}` ({livedoor['document_stats']['min_doc']['chars']:,} 文字, {livedoor['document_stats']['min_doc']['lines']:,} 行)")
        lines.append(f"- **最長記事**: `{livedoor['document_stats']['max_doc']['file']}` ({livedoor['document_stats']['max_doc']['chars']:,} 文字, {livedoor['document_stats']['max_doc']['lines']:,} 行)")
        lines.append(f"- **最短行**: `{livedoor['sentence_and_line_stats']['min_line']['file']}` ({livedoor['sentence_and_line_stats']['min_line']['length']:,} 文字) — `{livedoor['sentence_and_line_stats']['min_line']['excerpt']}`")
        lines.append(f"- **最長行**: `{livedoor['sentence_and_line_stats']['max_line']['file']}` ({livedoor['sentence_and_line_stats']['max_line']['length']:,} 文字) — `{livedoor['sentence_and_line_stats']['max_line']['excerpt']}`")
        lines.append("")
    if aozora and aozora["document_stats"]["min_doc"]:
        lines.append("### aozora（青空文庫）")
        lines.append(f"- **最短作品**: `{aozora['document_stats']['min_doc']['file']}` ({aozora['document_stats']['min_doc']['chars']:,} 文字, {aozora['document_stats']['min_doc']['lines']:,} 行)")
        lines.append(f"- **最長作品**: `{aozora['document_stats']['max_doc']['file']}` ({aozora['document_stats']['max_doc']['chars']:,} 文字, {aozora['document_stats']['max_doc']['lines']:,} 行)")
        lines.append(f"- **最短行**: `{aozora['sentence_and_line_stats']['min_line']['file']}` ({aozora['sentence_and_line_stats']['min_line']['length']:,} 文字) — `{aozora['sentence_and_line_stats']['min_line']['excerpt']}`")
        lines.append(f"- **最長行**: `{aozora['sentence_and_line_stats']['max_line']['file']}` ({aozora['sentence_and_line_stats']['max_line']['length']:,} 文字) — `{aozora['sentence_and_line_stats']['max_line']['excerpt']}`")
        lines.append("")

    return "\n".join(lines)
