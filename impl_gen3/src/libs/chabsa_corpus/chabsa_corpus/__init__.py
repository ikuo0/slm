"""共通フォーマット JSON (またはディレクトリ) から SentencePiece 用コーパスを作るモジュール (impl_gen3)。

入力 JSON の形式:
{
  "sentences": [
    {
      "original": "chabsa/e00008_ann.json",
      "lines": [
        "文1",
        "文2"
      ]
    }
  ]
}

- sentences 配列内の 1 要素（original 単位）を 1 文書として扱い、1 文書全体を <s> と </s> で囲む
- 文書内の文区切り（lines の各要素間）は <n> + 改行 (\\n) とし、1 文 = 1 行 で出力する
  例:
    <s>文1<n>
    文2<n>
    文3</s>
- <s>, </s>, <n> は SentencePiece 学習時に bos_id=-1, eos_id=-1 とした上で
  user_defined_symbols へ渡すことで単一トークンとして扱われる
  (get_special_tokens() で一覧を取得できる)
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_NEWLINE_TOKEN",
    "get_special_tokens",
    "lines_to_corpus_lines",
    "document_to_lines",
    "document_to_line",
    "iter_document_lines_from_json",
    "build_corpus",
]

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_NEWLINE_TOKEN = "<n>"


def get_special_tokens(
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[str]:
    """SentencePiece の user_defined_symbols に渡す特殊記号の一覧を返す。"""
    tokens: list[str] = []
    for t in (bos_token, eos_token, newline_token):
        if t and t not in tokens:
            tokens.append(t)
    return tokens


def lines_to_corpus_lines(
    raw_lines: list[str],
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[str]:
    """1文書分の行リストを、<s>...<n>\\n...</s> の行リストに変換する。"""
    cleaned_sentences: list[str] = []
    for raw in raw_lines:
        text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", newline_token).strip()
        if text:
            cleaned_sentences.append(text)

    if not cleaned_sentences:
        return []

    joined = f"{newline_token}\n".join(cleaned_sentences)
    wrapped = f"{bos_token}{joined}{eos_token}"
    return wrapped.split("\n")


def iter_document_lines_from_json(doc: dict) -> list[list[str]]:
    """JSON 辞書から 1 文書ごとの行リスト (list[str]) の一覧を取り出す。

    - impl_gen3 共通フォーマット: {"sentences": [{"original": "...", "lines": ["...", ...]}, ...]}
      -> sentences の各要素が 1 文書
    - impl_gen1 旧フォーマット互換: {"metadata": {...}, "sentences": [{"sentence_id": 0, "sentence": "..."}, ...]}
      -> ファイル全体が 1 文書
    """
    sentences = doc.get("sentences", [])
    if not sentences:
        return []

    first = sentences[0]
    if isinstance(first, dict) and "lines" in first:
        docs: list[list[str]] = []
        for entry in sentences:
            if not isinstance(entry, dict):
                continue
            entry_lines = [str(line) for line in entry.get("lines", []) if str(line).strip()]
            if entry_lines:
                docs.append(entry_lines)
        return docs

    # 旧形式フォールバック
    legacy_lines = [
        str(s.get("sentence", ""))
        for s in sentences
        if isinstance(s, dict) and str(s.get("sentence", "")).strip()
    ]
    return [legacy_lines] if legacy_lines else []


def document_to_lines(
    doc: dict,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[str]:
    """JSON 辞書（共通フォーマットまたは単一文書）を、1文1行の行リストに変換する。"""
    if "lines" in doc and isinstance(doc["lines"], list):
        return lines_to_corpus_lines(
            doc["lines"],
            newline_token=newline_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    all_corpus_lines: list[str] = []
    for doc_lines in iter_document_lines_from_json(doc):
        all_corpus_lines.extend(
            lines_to_corpus_lines(
                doc_lines,
                newline_token=newline_token,
                bos_token=bos_token,
                eos_token=eos_token,
            )
        )
    return all_corpus_lines


def document_to_line(
    doc: dict,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> str:
    """JSON 辞書を <s>...<n>\\n...</s> 形式の文字列に変換する。"""
    return "\n".join(
        document_to_lines(
            doc,
            newline_token=newline_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
    )


def build_corpus(
    json_path_or_dir: str | Path,
    out_corpus_file: str | Path,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> dict:
    """共通フォーマット JSON ファイル（または JSON ディレクトリ）からコーパスファイルを生成する。

    1文書（sentences 内の 1 original エントリ）を <s>...</s> で囲み、
    文区切りを <n>\\n とした 1文1行形式で書き出す。

    Returns:
        文書数・文数・行数・特殊記号の辞書
    """
    in_path = Path(json_path_or_dir)
    out_corpus_file = Path(out_corpus_file)

    if in_path.is_file():
        json_files = [in_path]
    elif in_path.is_dir():
        json_files = sorted(in_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"入力パスが存在しません: {in_path}")

    num_documents = 0
    num_sentences = 0
    num_lines = 0
    out_corpus_file.parent.mkdir(parents=True, exist_ok=True)
    with out_corpus_file.open("w", encoding="utf-8") as fout:
        for json_path in json_files:
            with json_path.open(encoding="utf-8") as fin:
                doc = json.load(fin)
            for doc_lines in iter_document_lines_from_json(doc):
                corpus_lines = lines_to_corpus_lines(
                    doc_lines,
                    newline_token=newline_token,
                    bos_token=bos_token,
                    eos_token=eos_token,
                )
                if not corpus_lines:
                    continue
                for line in corpus_lines:
                    fout.write(line + "\n")
                num_documents += 1
                num_sentences += len(doc_lines)
                num_lines += len(corpus_lines)

    return {
        "num_documents": num_documents,
        "num_sentences": num_sentences,
        "num_lines": num_lines,
        "special_tokens": get_special_tokens(
            newline_token=newline_token,
            bos_token=bos_token,
            eos_token=eos_token,
        ),
    }
