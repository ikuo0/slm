"""重み付き One-Hot 特徴量 (X) と正解トークン ID (Y) を生成するモジュール。

normalized 形式の JSON（単一ファイルまたはディレクトリ）と SentencePiece の
vocab ファイル（または model ファイル）を受け取り、以下の仕様で (X, Y) を作成する：

- 1文書ごとに先頭を context_length (デフォルト 32) 個の <s> でパディングし、
  文区切りには <n>、文書末尾には 1 個の </s> を配置する（末尾パディングは行わない）
- 文脈ウィンドウ（32単語）の直後に来る 33 単語目を正解ラベル Y (int32) とする
- 文脈内の各単語について、正解単語からの距離 d (1〜32) の逆数 (1/d) を
  その単語の語彙インデックスに加算した「重み付き One-Hot の総和」を 1 サンプルの X ベクトルとする
- X は疎行列 (scipy.sparse.csr_matrix) として .npz ファイルに保存し、
  Y は 1 次元配列 (numpy.ndarray, int32) として .npy ファイルに保存する
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.sparse as sp
import sentencepiece as spm

__all__ = [
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_NEWLINE_TOKEN",
    "DEFAULT_CONTEXT_LENGTH",
    "VocabTokenizer",
    "load_vocab_file",
    "document_to_body_ids",
    "body_ids_to_windows",
    "windows_to_csr",
    "build_xy_from_document",
    "build_xy_from_file",
    "build_xy_from_directory",
    "build_and_save_xy",
]

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_NEWLINE_TOKEN = "<n>"
DEFAULT_CONTEXT_LENGTH = 32


def load_vocab_file(vocab_file: str | Path) -> tuple[list[str], dict[str, int], dict[str, float]]:
    """SentencePiece の .vocab ファイルを読み込み、(ピース一覧, piece->id 辞書, piece->score 辞書) を返す。"""
    vocab_path = Path(vocab_file)
    pieces: list[str] = []
    piece_to_id: dict[str, int] = {}
    piece_to_score: dict[str, float] = {}

    with vocab_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            piece = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            pieces.append(piece)
            if piece not in piece_to_id:
                piece_to_id[piece] = idx
            piece_to_score[piece] = score

    return pieces, piece_to_id, piece_to_score


class VocabTokenizer:
    """SentencePiece の .vocab または .model を使ってテキストをトークン ID 列に変換するクラス。

    - .model が指定された場合、または .vocab と同じディレクトリに同名の .model が存在する場合は
      SentencePieceProcessor を用いてエンコードする。
    - .vocab のみ存在し .model が見つからない場合は、.vocab 内の対数確率スコアを用いた
      最尤（Viterbi）分割でトークン ID 列に変換する。
    """

    def __init__(
        self,
        vocab_file: str | Path,
        *,
        model_file: str | Path | None = None,
    ) -> None:
        self.vocab_file = Path(vocab_file)
        resolved_model: Path | None = None

        if model_file is not None:
            resolved_model = Path(model_file)
        elif self.vocab_file.suffix == ".model":
            resolved_model = self.vocab_file
        else:
            candidate = self.vocab_file.with_suffix(".model")
            if candidate.is_file():
                resolved_model = candidate

        self.model_file = resolved_model
        self._sp: spm.SentencePieceProcessor | None = None

        if self.model_file is not None and self.model_file.is_file():
            self._sp = spm.SentencePieceProcessor(model_file=str(self.model_file))

        if self.vocab_file.suffix == ".vocab" and self.vocab_file.is_file():
            self.pieces, self.piece_to_id, self.piece_to_score = load_vocab_file(self.vocab_file)
        elif self._sp is not None:
            vsize = self._sp.get_piece_size()
            self.pieces = [self._sp.id_to_piece(i) for i in range(vsize)]
            self.piece_to_id = {p: i for i, p in enumerate(self.pieces)}
            self.piece_to_score = {
                self.pieces[i]: float(self._sp.get_score(i)) for i in range(vsize)
            }
        else:
            raise FileNotFoundError(
                f"有効な vocab ファイルまたは model ファイルが見つかりません: {self.vocab_file}"
            )

        self.vocab_size = len(self.pieces)
        self.unk_id = self.piece_to_id.get("<unk>", 0)
        self._max_piece_len = max((len(p) for p in self.pieces), default=1)

    def token_to_id(self, token: str) -> int:
        """トークン文字列に対応する ID を返す（存在しない場合は unk_id）。"""
        if self._sp is not None:
            return int(self._sp.piece_to_id(token))
        return self.piece_to_id.get(token, self.unk_id)

    def encode_ids(self, text: str) -> list[int]:
        """テキストをトークン ID のリストに変換する。"""
        if not text:
            return []
        if self._sp is not None:
            return [int(i) for i in self._sp.encode(text, out_type=int)]
        return self._viterbi_encode_ids(text)

    def _viterbi_encode_ids(self, text: str) -> list[int]:
        """`.vocab` のみ与えられた場合の Viterbi 分割フォールバック。"""
        # SentencePiece のデフォルト空白置換（先頭および半角スペースを ▁ に変換）
        normalized = "▁" + text.replace(" ", "▁")
        n = len(normalized)
        best_score = [-math.inf] * (n + 1)
        best_edge: list[tuple[int, int] | None] = [None] * (n + 1)
        best_score[0] = 0.0

        for i in range(n):
            if best_score[i] == -math.inf:
                continue
            max_end = min(n, i + self._max_piece_len)
            matched = False
            for j in range(i + 1, max_end + 1):
                sub = normalized[i:j]
                if sub in self.piece_to_id:
                    matched = True
                    pid = self.piece_to_id[sub]
                    sc = self.piece_to_score.get(sub, -10.0)
                    cand = best_score[i] + sc
                    if cand > best_score[j]:
                        best_score[j] = cand
                        best_edge[j] = (i, pid)
            if not matched:
                # 未知文字は 1 文字進めて unk_id を割り当てる
                cand = best_score[i] - 100.0
                if cand > best_score[i + 1]:
                    best_score[i + 1] = cand
                    best_edge[i + 1] = (i, self.unk_id)

        ids: list[int] = []
        pos = n
        while pos > 0:
            edge = best_edge[pos]
            if edge is None:
                break
            prev_pos, pid = edge
            ids.append(pid)
            pos = prev_pos
        ids.reverse()
        return ids


def document_to_body_ids(
    doc: dict,
    tokenizer: VocabTokenizer,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[int]:
    """simplified 形式の正規化済み JSON 1文書から、先頭 <s> を除くトークン ID 列を生成する。

    生成されるトークン列:
        [文1のトークン列..., <n>, 文2のトークン列..., <n>, ..., 最終文のトークン列..., </s>]
    （先頭の 32 個の <s> パディングは body_ids_to_windows で付与する）
    """
    sentences = doc.get("sentences", [])
    cleaned_sentences: list[str] = []
    for s in sentences:
        text = s.get("sentence", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", newline_token)
        if text:
            cleaned_sentences.append(text)

    if not cleaned_sentences:
        return []

    body_ids: list[int] = []
    num_sents = len(cleaned_sentences)
    for idx, sent in enumerate(cleaned_sentences):
        # コーパス作成時と同じく、最終文以外は末尾に <n>、最終文は末尾に </s> を付けてエンコード
        suffix = newline_token if idx < num_sents - 1 else eos_token
        line_ids = tokenizer.encode_ids(f"{sent}{suffix}")
        body_ids.extend(line_ids)

    return body_ids


def body_ids_to_windows(
    body_ids: list[int],
    *,
    bos_id: int,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """1文書分のトークン ID 列に先頭 context_length 個の <s> をパディングし、(文脈ウィンドウ配列, Y配列) を返す。

    Returns:
        windows: shape=(N, context_length), dtype=int32
        y: shape=(N,), dtype=int32 (各ウィンドウの直後に来る 33 単語目のトークン ID)
    """
    if not body_ids:
        return (
            np.empty((0, context_length), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    padded = np.empty(context_length + len(body_ids), dtype=np.int32)
    padded[:context_length] = bos_id
    padded[context_length:] = body_ids

    # padded[:-1] から長さ context_length のスライディングウィンドウを作成
    windows = np.ascontiguousarray(
        sliding_window_view(padded[:-1], context_length),
        dtype=np.int32,
    )
    y = np.ascontiguousarray(padded[context_length:], dtype=np.int32)
    return windows, y


def windows_to_csr(
    windows: np.ndarray,
    *,
    vocab_size: int,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    dtype: type = np.float32,
) -> sp.csr_matrix:
    """文脈ウィンドウ配列 (N, context_length) から重み付き One-Hot 総和の CSR 疎行列 (N, vocab_size) を構築する。

    ウィンドウ内の列 j (0 <= j < context_length) の正解単語からの距離は d = context_length - j であり、
    重み 1 / d を該当単語 ID のインデックスに加算する。
    """
    num_samples = windows.shape[0]
    if num_samples == 0:
        return sp.csr_matrix((0, vocab_size), dtype=dtype)

    # 距離 d = context_length, context_length - 1, ..., 1 の逆数
    weights = np.array(
        [1.0 / (context_length - j) for j in range(context_length)],
        dtype=dtype,
    )

    row_indices = np.repeat(np.arange(num_samples, dtype=np.int32), context_length)
    col_indices = windows.ravel()
    data = np.tile(weights, num_samples)

    matrix = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(num_samples, vocab_size),
        dtype=dtype,
    )
    # 同一ウィンドウ内に同じ単語が複数回出現した場合の重みを合算して正規化する
    matrix.sum_duplicates()
    return matrix


def build_xy_from_document(
    doc: dict,
    tokenizer: VocabTokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """正規化済み JSON 辞書 1 件から (X: csr_matrix, Y: ndarray) を生成する。"""
    bos_id = tokenizer.token_to_id(bos_token)
    body_ids = document_to_body_ids(
        doc,
        tokenizer,
        newline_token=newline_token,
        eos_token=eos_token,
    )
    windows, y = body_ids_to_windows(
        body_ids,
        bos_id=bos_id,
        context_length=context_length,
    )
    x = windows_to_csr(
        windows,
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
    )
    return x, y


def build_xy_from_file(
    normalized_file: str | Path,
    tokenizer: VocabTokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """正規化済み JSON ファイル 1 件から (X, Y, stats) を生成する。"""
    in_path = Path(normalized_file)
    with in_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    x, y = build_xy_from_document(
        doc,
        tokenizer,
        context_length=context_length,
        bos_token=bos_token,
        eos_token=eos_token,
        newline_token=newline_token,
    )
    stats = {
        "num_documents": 1,
        "num_sentences": len(doc.get("sentences", [])),
        "num_samples": int(y.shape[0]),
        "vocab_size": tokenizer.vocab_size,
        "context_length": context_length,
        "x_shape": [int(x.shape[0]), int(x.shape[1])],
        "x_nnz": int(x.nnz),
        "y_shape": [int(y.shape[0])],
    }
    return x, y, stats


def build_xy_from_directory(
    normalized_dir: str | Path,
    tokenizer: VocabTokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """ディレクトリ内の全 *.json から一括で (X, Y, stats) を生成する。"""
    dir_path = Path(normalized_dir)
    bos_id = tokenizer.token_to_id(bos_token)

    all_windows: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    num_documents = 0
    num_sentences = 0

    for json_path in sorted(dir_path.glob("*.json")):
        with json_path.open(encoding="utf-8") as f:
            doc = json.load(f)
        body_ids = document_to_body_ids(
            doc,
            tokenizer,
            newline_token=newline_token,
            eos_token=eos_token,
        )
        if not body_ids:
            continue
        windows, y = body_ids_to_windows(
            body_ids,
            bos_id=bos_id,
            context_length=context_length,
        )
        all_windows.append(windows)
        all_y.append(y)
        num_documents += 1
        num_sentences += len(doc.get("sentences", []))

    if all_windows:
         stacked_windows = np.vstack(all_windows)
         stacked_y = np.concatenate(all_y)
    else:
        stacked_windows = np.empty((0, context_length), dtype=np.int32)
        stacked_y = np.empty((0,), dtype=np.int32)

    x = windows_to_csr(
        stacked_windows,
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
    )

    stats = {
        "num_documents": num_documents,
        "num_sentences": num_sentences,
        "num_samples": int(stacked_y.shape[0]),
        "vocab_size": tokenizer.vocab_size,
        "context_length": context_length,
        "x_shape": [int(x.shape[0]), int(x.shape[1])],
        "x_nnz": int(x.nnz),
        "y_shape": [int(stacked_y.shape[0])],
    }
    return x, stacked_y, stats


def build_and_save_xy(
    normalized_path: str | Path,
    vocab_file: str | Path,
    out_x_file: str | Path,
    out_y_file: str | Path,
    *,
    out_meta_file: str | Path | None = None,
    model_file: str | Path | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> dict:
    """normalized JSON（ファイルまたはディレクトリ）と vocab から X (.npz) と Y (.npy) を作成して保存する。

    Args:
        normalized_path: 正規化済み JSON ファイルまたはディレクトリのパス
        vocab_file: SentencePiece の .vocab ファイル（または .model ファイル）のパス
        out_x_file: X 疎行列の出力ファイルパス (.npz)
        out_y_file: Y 正解ラベル配列の出力ファイルパス (.npy)
        out_meta_file: メタデータ JSON の出力ファイルパス (任意)
        model_file: 明示的に SentencePiece の .model ファイルを指定する場合のパス (任意)
        context_length: 文脈ウィンドウの単語数 (default: 32)
        bos_token: 文書先頭パディング用の開始トークン (default: <s>)
        eos_token: 文書末尾の終了トークン (default: </s>)
        newline_token: 文区切りトークン (default: <n>)

    Returns:
        メタデータの辞書
    """
    if context_length < 1:
        raise ValueError(f"context_length は 1 以上を指定してください: {context_length}")

    normalized_path = Path(normalized_path)
    vocab_file = Path(vocab_file)
    out_x_file = Path(out_x_file)
    out_y_file = Path(out_y_file)

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    tokenizer = VocabTokenizer(vocab_file, model_file=model_file)

    if normalized_path.is_dir():
        x, y, stats = build_xy_from_directory(
            normalized_path,
            tokenizer,
            context_length=context_length,
            bos_token=bos_token,
            eos_token=eos_token,
            newline_token=newline_token,
        )
    elif normalized_path.is_file():
        x, y, stats = build_xy_from_file(
            normalized_path,
            tokenizer,
            context_length=context_length,
            bos_token=bos_token,
            eos_token=eos_token,
            newline_token=newline_token,
        )
    else:
        raise FileNotFoundError(f"入力パスが存在しません: {normalized_path}")

    out_x_file.parent.mkdir(parents=True, exist_ok=True)
    out_y_file.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(out_x_file), x)
    np.save(str(out_y_file), y)

    elapsed = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    outputs_dict = {
        "x_file": str(out_x_file),
        "y_file": str(out_y_file),
    }
    if out_meta_file is not None:
        outputs_dict["meta_file"] = str(Path(out_meta_file))

    metadata = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "normalized_path": str(normalized_path),
            "vocab_file": str(vocab_file),
            "model_file": str(tokenizer.model_file) if tokenizer.model_file else None,
        },
        "outputs": outputs_dict,
        "params": {
            "context_length": context_length,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "newline_token": newline_token,
        },
        "stats": stats,
    }

    if out_meta_file is not None:
        meta_path = Path(out_meta_file)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata

