"""SentencePiece によるトークン化モジュール。

入力テキストから SentencePiece モデルを学習し、テキストをトークン化する。
- コーパス内に <s>, </s>, <n> が埋め込まれている前提で、デフォルトでは
  bos_id=-1, eos_id=-1 として組み込み予約を解除し、user_defined_symbols として
  単一トークン登録する（対策A）。
- モデル・語彙・トークン列・メタデータはすべて個別のファイルパスで指定して出力する。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm

__all__ = ["DEFAULT_USER_DEFINED_SYMBOLS", "train_model", "tokenize_file", "train_and_tokenize"]

DEFAULT_USER_DEFINED_SYMBOLS = ("<s>", "</s>", "<n>")


def train_model(
    input_text_file: str | Path,
    out_model_file: str | Path,
    out_vocab_file: str | Path,
    *,
    vocab_size: int = 8000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    user_defined_symbols: list[str] | None = None,
    bos_id: int = -1,
    eos_id: int = -1,
) -> dict:
    """SentencePiece モデルを学習し、モデル・語彙を指定パスへ出力する。

    デフォルトでは bos_id=-1, eos_id=-1 を指定して <s>, </s> の制御記号予約を解除し、
    user_defined_symbols として <s>, </s>, <n> を単一トークン登録できるようにする。

    Returns:
        学習に使用した SentencePiece パラメータの辞書
    """
    input_text_file = Path(input_text_file)
    out_model_file = Path(out_model_file)
    out_vocab_file = Path(out_vocab_file)

    symbols = (
        list(user_defined_symbols)
        if user_defined_symbols is not None
        else list(DEFAULT_USER_DEFINED_SYMBOLS)
    )

    params: dict = {
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }
    if symbols:
        params["user_defined_symbols"] = symbols

    # SentencePiece は model_prefix からの .model / .vocab 固定名で出力するため、
    # 一時ディレクトリに学習してから指定パスへ移動する
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = str(Path(tmp_dir) / "sp")
        spm.SentencePieceTrainer.train(
            input=str(input_text_file),
            model_prefix=prefix,
            **params,
        )
        out_model_file.parent.mkdir(parents=True, exist_ok=True)
        out_vocab_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(prefix + ".model", out_model_file)
        shutil.move(prefix + ".vocab", out_vocab_file)

    return params


def tokenize_file(
    model_file: str | Path,
    input_text_file: str | Path,
    out_tokens_file: str | Path,
    *,
    output_format: str = "piece",
    add_bos: bool = False,
    add_eos: bool = False,
) -> dict:
    """学習済みモデルでテキストを1行ずつトークン化し、1行1文の形式で出力する。

    Args:
        output_format: "piece" はトークン文字列、"id" はトークン ID を空白区切りで出力
        add_bos: 各行の先頭に BOS を付与する (bos_id が有効な場合のみ)
        add_eos: 各行の末尾に EOS を付与する (eos_id が有効な場合のみ)

    Returns:
        行数・トークン総数などの統計の辞書
    """
    if output_format not in ("piece", "id"):
        raise ValueError(f"Unknown output_format: {output_format}")

    model_file = Path(model_file)
    input_text_file = Path(input_text_file)
    out_tokens_file = Path(out_tokens_file)

    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    use_add_bos = bool(add_bos and sp.bos_id() >= 0)
    use_add_eos = bool(add_eos and sp.eos_id() >= 0)

    num_lines = 0
    num_tokens = 0
    out_tokens_file.parent.mkdir(parents=True, exist_ok=True)
    with input_text_file.open(encoding="utf-8") as fin, out_tokens_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if output_format == "piece":
                tokens = sp.encode(
                    line,
                    out_type=str,
                    add_bos=use_add_bos,
                    add_eos=use_add_eos,
                )
            else:
                tokens = [
                    str(i)
                    for i in sp.encode(
                        line,
                        out_type=int,
                        add_bos=use_add_bos,
                        add_eos=use_add_eos,
                    )
                ]
            fout.write(" ".join(tokens) + "\n")
            num_lines += 1
            num_tokens += len(tokens)

    return {
        "num_lines": num_lines,
        "num_tokens": num_tokens,
        "output_format": output_format,
        "add_bos": use_add_bos,
        "add_eos": use_add_eos,
    }


def train_and_tokenize(
    input_text_file: str | Path,
    out_model_file: str | Path,
    out_vocab_file: str | Path,
    out_tokens_file: str | Path,
    out_meta_file: str | Path,
    *,
    vocab_size: int = 8000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    user_defined_symbols: list[str] | None = None,
    bos_id: int = -1,
    eos_id: int = -1,
    output_format: str = "piece",
    add_bos: bool = False,
    add_eos: bool = False,
) -> dict:
    """学習とトークン化を実行し、メタデータを指定パスへ出力する。

    Returns:
        メタデータの辞書 (開始終了時間、処理時間、SentencePiece パラメータ、統計)
    """
    import json

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    params = train_model(
        input_text_file,
        out_model_file,
        out_vocab_file,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        user_defined_symbols=user_defined_symbols,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    stats = tokenize_file(
        out_model_file,
        input_text_file,
        out_tokens_file,
        output_format=output_format,
        add_bos=add_bos,
        add_eos=add_eos,
    )

    elapsed = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    metadata = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "input_text_file": str(input_text_file),
        },
        "outputs": {
            "model_file": str(out_model_file),
            "vocab_file": str(out_vocab_file),
            "tokens_file": str(out_tokens_file),
            "meta_file": str(out_meta_file),
        },
        "sentencepiece_params": params,
        "stats": stats,
    }

    out_meta_file = Path(out_meta_file)
    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata
