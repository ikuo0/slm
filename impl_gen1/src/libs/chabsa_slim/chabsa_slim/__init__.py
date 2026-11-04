"""chABSA のアノテーション JSON から余計なデータを間引くモジュール。

origin の JSON (header / sentences + opinions) を
metadata / sentences (sentence_id, sentence のみ) の形に変換する。
"""

import json
from pathlib import Path

__all__ = ["slim_document", "slim_file", "slim_directory"]


def slim_document(doc: dict) -> dict:
    """chABSA ドキュメント 1 件を間引いた形式に変換する。

    Args:
        doc: origin JSON をパースした辞書 (header / sentences を持つ)

    Returns:
        {"metadata": {...}, "sentences": [{"sentence_id", "sentence"}, ...]}
    """
    metadata = dict(doc.get("header", {}))
    sentences = [
        {"sentence_id": s["sentence_id"], "sentence": s["sentence"]}
        for s in doc.get("sentences", [])
    ]
    return {"metadata": metadata, "sentences": sentences}


def slim_file(in_path: str | Path, out_path: str | Path) -> dict:
    """JSON ファイル 1 件を変換して書き出す。変換結果を返す。"""
    in_path = Path(in_path)
    out_path = Path(out_path)
    with in_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    slimmed = slim_document(doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(slimmed, f, ensure_ascii=False, indent=2)
    return slimmed


def slim_directory(chabsa_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """ディレクトリ内の *.json を列挙して一括変換する。

    出力ファイル名は元のファイル名を踏襲する。書き出したパスの一覧を返す。
    """
    chabsa_dir = Path(chabsa_dir)
    out_dir = Path(out_dir)
    written: list[Path] = []
    for in_path in sorted(chabsa_dir.glob("*.json")):
        out_path = out_dir / in_path.name
        slim_file(in_path, out_path)
        written.append(out_path)
    return written
