"""chABSA および livedoor コーパスから本文テキストを抽出し、共通フォーマット JSON に変換するモジュール。

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
    }
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "EXCLUDED_LIVEDOOR_FILENAMES",
    "extract_chabsa_entry",
    "extract_chabsa_file",
    "collect_chabsa_entries",
    "extract_livedoor_entry",
    "extract_livedoor_file",
    "collect_livedoor_entries",
    "build_slim_dataset",
    "slim_document",
    "slim_file",
    "slim_directory",
]

EXCLUDED_LIVEDOOR_FILENAMES = frozenset({"LICENSE.TXT", "README.TXT", "CHANGES.TXT"})


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


def build_slim_dataset(
    out_slim_file: str | Path,
    *,
    chabsa_dir: str | Path | None = None,
    livedoor_dir: str | Path | None = None,
) -> dict:
    """chABSA と livedoor コーパスから共通フォーマット JSON を構築して 1 ファイルに保存する。

    Args:
        out_slim_file: 出力先 JSON ファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/slim.json)
        chabsa_dir: chABSA origin JSON ディレクトリ (任意)
        livedoor_dir: livedoor コーパス text ディレクトリ (任意)

    Returns:
        統計情報の辞書 (num_documents, num_chabsa_documents, num_livedoor_documents, num_lines, out_slim_file)
    """
    if chabsa_dir is None and livedoor_dir is None:
        raise ValueError("chabsa_dir または livedoor_dir の少なくとも一方を指定してください")

    out_path = Path(out_slim_file)
    if out_path.suffix.lower() != ".json":
        out_path = out_path / "slim.json"

    chabsa_entries: list[dict] = []
    livedoor_entries: list[dict] = []

    if chabsa_dir is not None:
        chabsa_entries = collect_chabsa_entries(chabsa_dir)
    if livedoor_dir is not None:
        livedoor_entries = collect_livedoor_entries(livedoor_dir)

    all_entries = [*chabsa_entries, *livedoor_entries]
    payload = {
        "sentences": all_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    chabsa_lines = sum(len(e["lines"]) for e in chabsa_entries)
    livedoor_lines = sum(len(e["lines"]) for e in livedoor_entries)
    total_lines = chabsa_lines + livedoor_lines

    return {
        "out_slim_file": str(out_path),
        "num_documents": len(all_entries),
        "num_chabsa_documents": len(chabsa_entries),
        "num_livedoor_documents": len(livedoor_entries),
        "num_sentences": total_lines,
        "num_chabsa_sentences": chabsa_lines,
        "num_livedoor_sentences": livedoor_lines,
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
) -> Path:
    """chABSA (および任意の livedoor) ディレクトリから共通フォーマット JSON ファイルを作成してそのパスを返す。"""
    stats = build_slim_dataset(
        out_dir_or_file,
        chabsa_dir=chabsa_dir,
        livedoor_dir=livedoor_dir,
    )
    return Path(stats["out_slim_file"])
