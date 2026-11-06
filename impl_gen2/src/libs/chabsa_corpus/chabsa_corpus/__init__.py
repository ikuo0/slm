"""chABSA simplified 形式の JSON から SentencePiece 用コーパスを作るモジュール。

指定ディレクトリの *.json (metadata / sentences 形式) を列挙し、
sentence キーの値を連結して SentencePiece の学習入力テキストを生成する。

- JSON 1ファイルを1文書として扱い、1文書全体を <s> と </s> で囲む
- 文書内の文区切りは <n> + 改行 (\\n) とし、1文 = 1行 で出力する
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
    "document_to_lines",
    "document_to_line",
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


def document_to_lines(
    doc: dict,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[str]:
    """simplified 形式の文書 1 件を、1文1行の行リストに変換する。

    文書全体を bos_token (<s>) と eos_token (</s>) で囲み、
    文と文の区切りには行末に newline_token (<n>) を付与して改行する。
    sentence 内に生の改行が含まれる場合も newline_token に置き換える。
    """
    cleaned_sentences: list[str] = []
    for s in doc.get("sentences", []):
        text = s.get("sentence", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", newline_token)
        if text:
            cleaned_sentences.append(text)

    if not cleaned_sentences:
        return []

    joined = f"{newline_token}\n".join(cleaned_sentences)
    wrapped = f"{bos_token}{joined}{eos_token}"
    return wrapped.split("\n")


def document_to_line(
    doc: dict,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> str:
    """simplified 形式の文書 1 件を <s>...<n>\\n...</s> 形式の文字列に変換する。"""
    return "\n".join(
        document_to_lines(
            doc,
            newline_token=newline_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
    )


def build_corpus(
    json_dir: str | Path,
    out_corpus_file: str | Path,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> dict:
    """ディレクトリ内の *.json を列挙してコーパスファイルを生成する。

    1文書を <s>...</s> で囲み、文区切りを <n>\\n とした 1文1行形式で書き出す。

    Returns:
        文書数・文数・行数・特殊記号の辞書
    """
    json_dir = Path(json_dir)
    out_corpus_file = Path(out_corpus_file)

    num_documents = 0
    num_sentences = 0
    num_lines = 0
    out_corpus_file.parent.mkdir(parents=True, exist_ok=True)
    with out_corpus_file.open("w", encoding="utf-8") as fout:
        for json_path in sorted(json_dir.glob("*.json")):
            with json_path.open(encoding="utf-8") as fin:
                doc = json.load(fin)
            lines = document_to_lines(
                doc,
                newline_token=newline_token,
                bos_token=bos_token,
                eos_token=eos_token,
            )
            if not lines:
                continue
            for line in lines:
                fout.write(line + "\n")
            num_documents += 1
            num_sentences += len(doc.get("sentences", []))
            num_lines += len(lines)

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

