"""chABSA および livedoor ニュースコーパスのオリジナルデータ（正規化前・SentencePiece前）の統計集計モジュール。

集計項目:
- 文書（ファイル）単位の統計: 文書数、1文書あたりの行数・文字数（合計・平均・標準偏差・最小・中央値・パーセンタイル・最大）
- 文（行）単位の統計: 総行数（文数）、1行あたりの文字数（合計・平均・標準偏差・最小・中央値・パーセンタイル・最大・長さ分布）
- 文字種・表記特性の統計: 総文字数、ユニーク文字数（異なり文字数）、文字種別（ひらがな・全角カナ・半角カナ・漢字・半角英数・全角英数・空白・記号等）の内訳、NFKC非正規文字の割合
- livedoor カテゴリ別の内訳統計
"""

from __future__ import annotations

import json
import math
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
    "analyze_corpora",
    "format_summary_markdown",
]

EXCLUDED_LIVEDOOR_FILENAMES = frozenset({"LICENSE.TXT", "README.TXT", "CHANGES.TXT"})

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


def _strip_private_keys(d: dict) -> dict:
    """内部受け渡し用の `_` 始まりキーを除外した辞書を返す。"""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def analyze_corpora(
    *,
    chabsa_dir: str | Path | None = None,
    livedoor_dir: str | Path | None = None,
) -> dict:
    """chABSA と livedoor コーパスをそれぞれ集計し、個別統計および合算統計を返す。"""
    chabsa_res = analyze_chabsa_directory(chabsa_dir) if chabsa_dir is not None else None
    livedoor_res = analyze_livedoor_directory(livedoor_dir) if livedoor_dir is not None else None

    combined_lines: list[str] = []
    combined_doc_lines: list[int] = []
    combined_doc_chars: list[int] = []
    combined_line_lengths: list[int] = []

    for res in (chabsa_res, livedoor_res):
        if res is not None:
            combined_lines.extend(res["_raw_lines"])
            combined_doc_lines.extend(res["_doc_line_counts"])
            combined_doc_chars.extend(res["_doc_char_counts"])
            combined_line_lengths.extend(res["_line_lengths"])

    combined_summary = {
        "corpus": "combined (chABSA + livedoor)",
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
        "combined": combined_summary,
    }


def format_summary_markdown(report: dict) -> str:
    """集計結果辞書を見やすい Markdown レポート文字列に整形する。"""
    lines: list[str] = []
    lines.append("# コーパス オリジナルデータ統計レポート（正規化前・SentencePiece前）")
    lines.append("")

    chabsa = report.get("chabsa")
    livedoor = report.get("livedoor")
    combined = report.get("combined")

    # 1. 全体サマリ比較表
    lines.append("## 1. 全体比較サマリ（chABSA vs livedoor vs 合計）")
    lines.append("")
    lines.append("| 指標 | chABSA | livedoor | 合計 (Combined) |")
    lines.append("| :--- | ---: | ---: | ---: |")

    def _fmt_int(val: int | float | None) -> str:
        return f"{int(val):,}" if val is not None else "-"

    def _fmt_flt(val: float | int | None) -> str:
        return f"{float(val):,.2f}" if val is not None else "-"

    c_docs = chabsa["files"]["valid_documents"] if chabsa else 0
    l_docs = livedoor["files"]["valid_documents"] if livedoor else 0
    t_docs = combined["valid_documents"] if combined else 0

    c_l_stat = chabsa["sentence_and_line_stats"]["line_chars"] if chabsa else {}
    l_l_stat = livedoor["sentence_and_line_stats"]["line_chars"] if livedoor else {}
    t_l_stat = combined["sentence_and_line_stats"]["line_chars"] if combined else {}

    c_d_chars = chabsa["document_stats"]["chars_per_doc"] if chabsa else {}
    l_d_chars = livedoor["document_stats"]["chars_per_doc"] if livedoor else {}
    t_d_chars = combined["document_stats"]["chars_per_doc"] if combined else {}

    c_d_lines = chabsa["document_stats"]["lines_per_doc"] if chabsa else {}
    l_d_lines = livedoor["document_stats"]["lines_per_doc"] if livedoor else {}
    t_d_lines = combined["document_stats"]["lines_per_doc"] if combined else {}

    c_c_stat = chabsa["char_stats"] if chabsa else {}
    l_c_stat = livedoor["char_stats"] if livedoor else {}
    t_c_stat = combined["char_stats"] if combined else {}

    lines.append(f"| **有効文書（ファイル）数** | {_fmt_int(c_docs)} | {_fmt_int(l_docs)} | **{_fmt_int(t_docs)}** |")
    lines.append(f"| **総行数（文数）** | {_fmt_int(c_l_stat.get('count'))} | {_fmt_int(l_l_stat.get('count'))} | **{_fmt_int(t_l_stat.get('count'))}** |")
    lines.append(f"| **総文字数（改行除く）** | {_fmt_int(c_l_stat.get('total'))} | {_fmt_int(l_l_stat.get('total'))} | **{_fmt_int(t_l_stat.get('total'))}** |")
    lines.append(f"| **ユニーク文字数（異なり文字数）** | {_fmt_int(c_c_stat.get('unique_chars'))} | {_fmt_int(l_c_stat.get('unique_chars'))} | **{_fmt_int(t_c_stat.get('unique_chars'))}** |")
    lines.append(f"| **1文書あたり平均行数** | {_fmt_flt(c_d_lines.get('mean'))} | {_fmt_flt(l_d_lines.get('mean'))} | **{_fmt_flt(t_d_lines.get('mean'))}** |")
    lines.append(f"| **1文書あたり行数（最小 / 中央値 / 最大）** | {_fmt_int(c_d_lines.get('min'))} / {_fmt_flt(c_d_lines.get('median'))} / {_fmt_int(c_d_lines.get('max'))} | {_fmt_int(l_d_lines.get('min'))} / {_fmt_flt(l_d_lines.get('median'))} / {_fmt_int(l_d_lines.get('max'))} | {_fmt_int(t_d_lines.get('min'))} / {_fmt_flt(t_d_lines.get('median'))} / {_fmt_int(t_d_lines.get('max'))} |")
    lines.append(f"| **1文書あたり平均文字数** | {_fmt_flt(c_d_chars.get('mean'))} | {_fmt_flt(l_d_chars.get('mean'))} | **{_fmt_flt(t_d_chars.get('mean'))}** |")
    lines.append(f"| **1文書あたり文字数（最小 / 中央値 / 最大）** | {_fmt_int(c_d_chars.get('min'))} / {_fmt_flt(c_d_chars.get('median'))} / {_fmt_int(c_d_chars.get('max'))} | {_fmt_int(l_d_chars.get('min'))} / {_fmt_flt(l_d_chars.get('median'))} / {_fmt_int(l_d_chars.get('max'))} | {_fmt_int(t_d_chars.get('min'))} / {_fmt_flt(t_d_chars.get('median'))} / {_fmt_int(t_d_chars.get('max'))} |")
    lines.append(f"| **1行（1文）あたり平均文字数 ± 標準偏差** | {_fmt_flt(c_l_stat.get('mean'))} ± {_fmt_flt(c_l_stat.get('std'))} | {_fmt_flt(l_l_stat.get('mean'))} ± {_fmt_flt(l_l_stat.get('std'))} | **{_fmt_flt(t_l_stat.get('mean'))} ± {_fmt_flt(t_l_stat.get('std'))}** |")
    lines.append(f"| **1行あたり文字数（最小 / 中央値 / 最大）** | {_fmt_int(c_l_stat.get('min'))} / {_fmt_flt(c_l_stat.get('median'))} / {_fmt_int(c_l_stat.get('max'))} | {_fmt_int(l_l_stat.get('min'))} / {_fmt_flt(l_l_stat.get('median'))} / {_fmt_int(l_l_stat.get('max'))} | {_fmt_int(t_l_stat.get('min'))} / {_fmt_flt(t_l_stat.get('median'))} / {_fmt_int(t_l_stat.get('max'))} |")
    lines.append(f"| **1行あたり文字数（25% / 75% / 95% / 99%点）** | {_fmt_flt(c_l_stat.get('p25'))} / {_fmt_flt(c_l_stat.get('p75'))} / {_fmt_flt(c_l_stat.get('p95'))} / {_fmt_flt(c_l_stat.get('p99'))} | {_fmt_flt(l_l_stat.get('p25'))} / {_fmt_flt(l_l_stat.get('p75'))} / {_fmt_flt(l_l_stat.get('p95'))} / {_fmt_flt(l_l_stat.get('p99'))} | {_fmt_flt(t_l_stat.get('p25'))} / {_fmt_flt(t_l_stat.get('p75'))} / {_fmt_flt(t_l_stat.get('p95'))} / {_fmt_flt(t_l_stat.get('p99'))} |")
    lines.append("")

    # 2. 文字数帯ヒストグラム比較
    lines.append("## 2. 1行（1文）あたりの文字数分布")
    lines.append("")
    lines.append("| 文字数帯 | chABSA 行数 (割合) | livedoor 行数 (割合) | 合計 行数 (割合) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    c_hist = {h["range"]: h for h in chabsa["sentence_and_line_stats"]["line_length_histogram"]} if chabsa else {}
    l_hist = {h["range"]: h for h in livedoor["sentence_and_line_stats"]["line_length_histogram"]} if livedoor else {}
    t_hist = {h["range"]: h for h in combined["sentence_and_line_stats"]["line_length_histogram"]} if combined else {}
    for label, _, _ in HISTOGRAM_BINS:
        ch_h = c_hist.get(label, {"count": 0, "percentage": 0.0})
        lv_h = l_hist.get(label, {"count": 0, "percentage": 0.0})
        cb_h = t_hist.get(label, {"count": 0, "percentage": 0.0})
        lines.append(
            f"| **{label}** | {ch_h['count']:,} ({ch_h['percentage']:.2f}%) | {lv_h['count']:,} ({lv_h['percentage']:.2f}%) | **{cb_h['count']:,} ({cb_h['percentage']:.2f}%)** |"
        )
    lines.append("")

    # 3. 文字種・表記揺れ統計（正規化前）
    lines.append("## 3. 正規化前の文字種・表記揺れ統計")
    lines.append("")
    lines.append("| 文字種カテゴリ | chABSA 文字数 (割合 / 異なり) | livedoor 文字数 (割合 / 異なり) | 合計 文字数 (割合 / 異なり) |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for cat_key, label in CHAR_CATEGORY_LABELS.items():
        c_cat = c_c_stat.get("categories", {}).get(cat_key, {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
        l_cat = l_c_stat.get("categories", {}).get(cat_key, {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
        t_cat = t_c_stat.get("categories", {}).get(cat_key, {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
        lines.append(
            f"| **{label}** | {c_cat['total_chars']:,} ({c_cat['percentage']:.2f}% / {c_cat['unique_chars']:,}種) | {l_cat['total_chars']:,} ({l_cat['percentage']:.2f}% / {l_cat['unique_chars']:,}種) | **{t_cat['total_chars']:,} ({t_cat['percentage']:.2f}% / {t_cat['unique_chars']:,}種)** |"
        )
    c_nfkc = c_c_stat.get("nfkc_convertible_chars", {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
    l_nfkc = l_c_stat.get("nfkc_convertible_chars", {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
    t_nfkc = t_c_stat.get("nfkc_convertible_chars", {"total_chars": 0, "percentage": 0.0, "unique_chars": 0})
    lines.append(
        f"| **（内）NFKC正規化対象文字** | {c_nfkc['total_chars']:,} ({c_nfkc['percentage']:.2f}% / {c_nfkc['unique_chars']:,}種) | {l_nfkc['total_chars']:,} ({l_nfkc['percentage']:.2f}% / {l_nfkc['unique_chars']:,}種) | **{t_nfkc['total_chars']:,} ({t_nfkc['percentage']:.2f}% / {t_nfkc['unique_chars']:,}種)** |"
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

    # 5. 最小・最大サンプルの詳細
    lines.append("## 5. 最小・最大サンプルの詳細")
    lines.append("")
    if chabsa:
        lines.append("### chABSA")
        lines.append(f"- **最短文書**: `{chabsa['document_stats']['min_doc']['file']}` ({chabsa['document_stats']['min_doc']['chars']:,} 文字, {chabsa['document_stats']['min_doc']['lines']:,} 行)")
        lines.append(f"- **最長文書**: `{chabsa['document_stats']['max_doc']['file']}` ({chabsa['document_stats']['max_doc']['chars']:,} 文字, {chabsa['document_stats']['max_doc']['lines']:,} 行)")
        lines.append(f"- **最短行**: `{chabsa['sentence_and_line_stats']['min_line']['file']}` ({chabsa['sentence_and_line_stats']['min_line']['length']:,} 文字) — `{chabsa['sentence_and_line_stats']['min_line']['excerpt']}`")
        lines.append(f"- **最長行**: `{chabsa['sentence_and_line_stats']['max_line']['file']}` ({chabsa['sentence_and_line_stats']['max_line']['length']:,} 文字) — `{chabsa['sentence_and_line_stats']['max_line']['excerpt']}`")
        lines.append("")
    if livedoor:
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

    return "\n".join(lines)
