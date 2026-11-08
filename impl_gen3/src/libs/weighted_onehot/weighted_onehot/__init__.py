"""重み付き One-Hot 特徴量 (X) と正解トークン ID (Y) を生成するモジュール (impl_gen3)。

Phase 1 で作成した共通フォーマットの正規化済み JSON（sentences: [{original, lines}]）と
SentencePiece の vocab ファイル（または model ファイル）を受け取り、以下の仕様で (X, Y) を作成する：

- 1文書（sentences 内の 1 original エントリ）ごとに先頭を context_length (デフォルト 32) 個の <s> でパディングし、
  文区切り（lines の要素間）には <n>、文書末尾には 1 個の </s> を配置する（末尾パディングは行わない）
- 文脈ウィンドウ（32単語）の直後に来る 33 単語目を正解ラベル Y (int32) とする
- 文脈内の各単語について、正解単語からの距離 d (1〜32) の逆数 (1/d) を
  その単語の語彙インデックスに加算した「重み付き One-Hot の総和」を 1 サンプルの X ベクトルとする
- コーパス間の偏りを防ぐため、文書順序を乱数シード固定でシャッフルした上で、
  shard_size（デフォルト 100万サンプル）単位で x_0001.npz / y_0001.npy, x_0002.npz / y_0002.npy, ... に分割保存する
"""

from __future__ import annotations

import gc
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
    "DEFAULT_SHARD_SIZE",
    "DEFAULT_SEED",
    "VocabTokenizer",
    "load_vocab_file",
    "lines_to_body_ids",
    "document_to_body_ids",
    "body_ids_to_windows",
    "windows_to_csr",
    "build_xy_from_document",
    "build_xy_from_file",
    "build_xy_from_directory",
    "build_and_save_xy_shards",
    "build_and_save_xy",
]

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_NEWLINE_TOKEN = "<n>"
DEFAULT_CONTEXT_LENGTH = 32
DEFAULT_SHARD_SIZE = 1_000_000
DEFAULT_SEED = 42
_CSR_WINDOW_CHUNK_SIZE = 200_000


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


def lines_to_body_ids(
    raw_lines: list[str],
    tokenizer: VocabTokenizer,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[int]:
    """1文書分の行リストから、先頭 <s> を除くトークン ID 列を生成する。

    生成されるトークン列:
        [行1のトークン列..., <n>, 行2のトークン列..., <n>, ..., 最終行のトークン列..., </s>]
    """
    cleaned_sentences: list[str] = []
    for raw in raw_lines:
        text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", newline_token).strip()
        if text:
            cleaned_sentences.append(text)

    if not cleaned_sentences:
        return []

    body_ids: list[int] = []
    num_sents = len(cleaned_sentences)
    for idx, sent in enumerate(cleaned_sentences):
        suffix = newline_token if idx < num_sents - 1 else eos_token
        line_ids = tokenizer.encode_ids(f"{sent}{suffix}")
        body_ids.extend(line_ids)

    return body_ids


def _iter_document_lines(doc: dict) -> list[list[str]]:
    """JSON 辞書から 1 文書ごとの行リスト (list[str]) の一覧を取り出す。"""
    if "lines" in doc and isinstance(doc["lines"], list):
        entry_lines = [str(line) for line in doc["lines"] if str(line).strip()]
        return [entry_lines] if entry_lines else []

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

    legacy_lines = [
        str(s.get("sentence", ""))
        for s in sentences
        if isinstance(s, dict) and str(s.get("sentence", "")).strip()
    ]
    return [legacy_lines] if legacy_lines else []


def document_to_body_ids(
    doc: dict,
    tokenizer: VocabTokenizer,
    *,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
) -> list[int]:
    """単一文書辞書から、先頭 <s> を除くトークン ID 列を生成する。"""
    doc_lines_list = _iter_document_lines(doc)
    if not doc_lines_list:
        return []
    return lines_to_body_ids(
        doc_lines_list[0],
        tokenizer,
        newline_token=newline_token,
        eos_token=eos_token,
    )


def body_ids_to_windows(
    body_ids: list[int],
    *,
    bos_id: int,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """1文書分のトークン ID 列に先頭 context_length 個の <s> をパディングし、(文脈ウィンドウ配列, Y配列) を返す。"""
    if not body_ids:
        return (
            np.empty((0, context_length), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    padded = np.empty(context_length + len(body_ids), dtype=np.int32)
    padded[:context_length] = bos_id
    padded[context_length:] = body_ids

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
    """文脈ウィンドウ配列 (N, context_length) から重み付き One-Hot 総和の CSR 疎行列 (N, vocab_size) を構築する。"""
    num_samples = int(windows.shape[0])
    if num_samples == 0:
        return sp.csr_matrix((0, vocab_size), dtype=dtype)

    weights = np.array(
        [1.0 / (context_length - j) for j in range(context_length)],
        dtype=dtype,
    )

    # 1シャード(100万件)変換時も一時配列メモリを抑えるため 20万行単位でCSR化して結合する
    if num_samples <= _CSR_WINDOW_CHUNK_SIZE:
        row_indices = np.repeat(np.arange(num_samples, dtype=np.int32), context_length)
        col_indices = windows.ravel()
        data = np.tile(weights, num_samples)
        matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_samples, vocab_size),
            dtype=dtype,
        )
        matrix.sum_duplicates()
        return matrix

    sub_chunks: list[sp.csr_matrix] = []
    for start in range(0, num_samples, _CSR_WINDOW_CHUNK_SIZE):
        sub_w = windows[start : start + _CSR_WINDOW_CHUNK_SIZE]
        n_sub = int(sub_w.shape[0])
        row_indices = np.repeat(np.arange(n_sub, dtype=np.int32), context_length)
        col_indices = sub_w.ravel()
        data = np.tile(weights, n_sub)
        sub_mat = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_sub, vocab_size),
            dtype=dtype,
        )
        sub_mat.sum_duplicates()
        sub_chunks.append(sub_mat)

    return sp.vstack(sub_chunks, format="csr") if len(sub_chunks) > 1 else sub_chunks[0]


def _build_xy_from_doc_lines_list(
    doc_lines_list: list[list[str]],
    tokenizer: VocabTokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """単一呼び出し用: 文書ごとの行リスト一覧から (X: csr_matrix, Y: ndarray, stats) を構築する。"""
    bos_id = tokenizer.token_to_id(bos_token)
    csr_chunks: list[sp.csr_matrix] = []
    window_buffer: list[np.ndarray] = []
    buffered_rows = 0
    all_y: list[np.ndarray] = []
    num_documents = 0
    num_sentences = 0

    def _flush_buffer() -> None:
        nonlocal window_buffer, buffered_rows
        if not window_buffer:
            return
        stacked = np.vstack(window_buffer)
        csr_chunks.append(
            windows_to_csr(
                stacked,
                vocab_size=tokenizer.vocab_size,
                context_length=context_length,
            )
        )
        window_buffer = []
        buffered_rows = 0

    for doc_lines in doc_lines_list:
        body_ids = lines_to_body_ids(
            doc_lines,
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
        window_buffer.append(windows)
        buffered_rows += int(windows.shape[0])
        all_y.append(y)
        num_documents += 1
        num_sentences += len(doc_lines)

        if buffered_rows >= _CSR_WINDOW_CHUNK_SIZE:
            _flush_buffer()

    _flush_buffer()

    if csr_chunks:
        x = sp.vstack(csr_chunks, format="csr") if len(csr_chunks) > 1 else csr_chunks[0]
        stacked_y = np.concatenate(all_y)
    else:
        x = sp.csr_matrix((0, tokenizer.vocab_size), dtype=np.float32)
        stacked_y = np.empty((0,), dtype=np.int32)

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


def build_xy_from_document(
    doc: dict,
    tokenizer: VocabTokenizer,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """正規化済み JSON 辞書から (X: csr_matrix, Y: ndarray) を生成する。"""
    doc_lines_list = _iter_document_lines(doc)
    x, y, _ = _build_xy_from_doc_lines_list(
        doc_lines_list,
        tokenizer,
        context_length=context_length,
        bos_token=bos_token,
        eos_token=eos_token,
        newline_token=newline_token,
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
    """正規化済み共通フォーマット JSON ファイルから (X, Y, stats) を生成する。"""
    in_path = Path(normalized_file)
    with in_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    doc_lines_list = _iter_document_lines(doc)
    return _build_xy_from_doc_lines_list(
        doc_lines_list,
        tokenizer,
        context_length=context_length,
        bos_token=bos_token,
        eos_token=eos_token,
        newline_token=newline_token,
    )


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
    all_doc_lines: list[list[str]] = []
    for json_path in sorted(dir_path.glob("*.json")):
        with json_path.open(encoding="utf-8") as f:
            doc = json.load(f)
        all_doc_lines.extend(_iter_document_lines(doc))

    return _build_xy_from_doc_lines_list(
        all_doc_lines,
        tokenizer,
        context_length=context_length,
        bos_token=bos_token,
        eos_token=eos_token,
        newline_token=newline_token,
    )


def _load_all_doc_lines(normalized_path: Path) -> list[list[str]]:
    """normalized_path（ファイルまたはディレクトリ）から全文書の行リスト一覧を読み込む。"""
    if normalized_path.is_dir():
        all_doc_lines: list[list[str]] = []
        for json_path in sorted(normalized_path.glob("*.json")):
            with json_path.open(encoding="utf-8") as f:
                doc = json.load(f)
            all_doc_lines.extend(_iter_document_lines(doc))
        return all_doc_lines
    if normalized_path.is_file():
        with normalized_path.open(encoding="utf-8") as f:
            doc = json.load(f)
        res = _iter_document_lines(doc)
        del doc
        return res
    raise FileNotFoundError(f"入力パスが存在しません: {normalized_path}")


def _clean_old_shard_files(out_dir: Path) -> None:
    """出力ディレクトリ内の既存の分割ファイル (x_NNNN.npz, y_NNNN.npy) を削除する。"""
    if not out_dir.is_dir():
        return
    for p in out_dir.glob("x_[0-9][0-9][0-9][0-9]*.npz"):
        try:
            p.unlink()
        except OSError:
            pass
    for p in out_dir.glob("y_[0-9][0-9][0-9][0-9]*.npy"):
        try:
            p.unlink()
        except OSError:
            pass


def build_and_save_xy_shards(
    normalized_path: str | Path,
    vocab_file: str | Path,
    out_directory: str | Path,
    *,
    out_meta_file: str | Path | None = None,
    model_file: str | Path | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    shard_size: int = DEFAULT_SHARD_SIZE,
    shuffle_documents: bool = True,
    seed: int = DEFAULT_SEED,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> dict:
    """normalized JSON と vocab から shard_size（デフォルト100万件）単位で x_0001.npz / y_0001.npy, ... を分割保存する。"""
    if context_length < 1:
        raise ValueError(f"context_length は 1 以上を指定してください: {context_length}")
    if shard_size < 1:
        raise ValueError(f"shard_size は 1 以上を指定してください: {shard_size}")

    normalized_path = Path(normalized_path)
    vocab_file = Path(vocab_file)
    out_dir = Path(out_directory).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_old_shard_files(out_dir)

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    tokenizer = VocabTokenizer(vocab_file, model_file=model_file)
    bos_id = tokenizer.token_to_id(bos_token)
    vocab_size = tokenizer.vocab_size

    doc_lines_list = _load_all_doc_lines(normalized_path)
    total_raw_docs = len(doc_lines_list)

    # コーパス種類（chABSA / livedoor / aozora）や作家の偏りをなくすため文書順序をシード固定でシャッフル
    rng = np.random.default_rng(seed)
    if shuffle_documents and total_raw_docs > 1:
        perm = rng.permutation(total_raw_docs)
        doc_lines_list = [doc_lines_list[int(i)] for i in perm]
        del perm

    window_buffer: list[np.ndarray] = []
    y_buffer: list[np.ndarray] = []
    buffered_rows = 0

    shards_info: list[dict] = []
    x_files: list[str] = []
    y_files: list[str] = []

    num_documents = 0
    num_sentences = 0
    total_samples = 0
    total_x_nnz = 0
    shard_index = 0

    def _write_one_shard(emit_w: np.ndarray, emit_y: np.ndarray) -> None:
        nonlocal shard_index, total_samples, total_x_nnz
        shard_index += 1
        shard_t0 = time.perf_counter()

        # チャンク内のサンプル順序もシャッフルしてから CSR 化する
        n_rows = int(emit_w.shape[0])
        if shuffle_documents and n_rows > 1:
            row_perm = rng.permutation(n_rows)
            emit_w = emit_w[row_perm]
            emit_y = emit_y[row_perm]
            del row_perm

        x_shard = windows_to_csr(
            emit_w,
            vocab_size=vocab_size,
            context_length=context_length,
        )
        x_shard_path = (out_dir / f"x_{shard_index:04d}.npz").resolve()
        y_shard_path = (out_dir / f"y_{shard_index:04d}.npy").resolve()

        sp.save_npz(str(x_shard_path), x_shard)
        np.save(str(y_shard_path), emit_y)

        shard_nnz = int(x_shard.nnz)
        shard_elapsed = round(time.perf_counter() - shard_t0, 3)

        x_path_str = str(x_shard_path)
        y_path_str = str(y_shard_path)
        x_files.append(x_path_str)
        y_files.append(y_path_str)
        shards_info.append(
            {
                "shard_index": shard_index,
                "x_file": x_path_str,
                "y_file": y_path_str,
                "num_samples": n_rows,
                "x_shape": [n_rows, vocab_size],
                "x_nnz": shard_nnz,
                "y_shape": [n_rows],
                "elapsed_seconds": shard_elapsed,
            }
        )
        total_samples += n_rows
        total_x_nnz += shard_nnz
        print(
            f"  [Shard {shard_index:04d}] saved: {x_shard_path.name} / {y_shard_path.name} "
            f"(samples={n_rows:,}, nnz={shard_nnz:,}, {shard_elapsed:.2f}s)",
            flush=True,
        )
        del x_shard

    def _drain_ready_shards(*, flush_all: bool = False) -> None:
        nonlocal window_buffer, y_buffer, buffered_rows
        if buffered_rows == 0:
            return

        while buffered_rows >= shard_size:
            stacked_w = np.vstack(window_buffer) if len(window_buffer) > 1 else window_buffer[0]
            stacked_y = np.concatenate(y_buffer) if len(y_buffer) > 1 else y_buffer[0]

            emit_w = np.ascontiguousarray(stacked_w[:shard_size], dtype=np.int32)
            emit_y = np.ascontiguousarray(stacked_y[:shard_size], dtype=np.int32)
            rem_w = np.ascontiguousarray(stacked_w[shard_size:], dtype=np.int32)
            rem_y = np.ascontiguousarray(stacked_y[shard_size:], dtype=np.int32)
            del stacked_w, stacked_y

            if rem_w.shape[0] > 0:
                window_buffer = [rem_w]
                y_buffer = [rem_y]
                buffered_rows = int(rem_w.shape[0])
            else:
                window_buffer = []
                y_buffer = []
                buffered_rows = 0

            _write_one_shard(emit_w, emit_y)
            del emit_w, emit_y
            gc.collect()

        if flush_all and buffered_rows > 0:
            stacked_w = np.vstack(window_buffer) if len(window_buffer) > 1 else window_buffer[0]
            stacked_y = np.concatenate(y_buffer) if len(y_buffer) > 1 else y_buffer[0]
            window_buffer = []
            y_buffer = []
            buffered_rows = 0
            _write_one_shard(
                np.ascontiguousarray(stacked_w, dtype=np.int32),
                np.ascontiguousarray(stacked_y, dtype=np.int32),
            )
            del stacked_w, stacked_y
            gc.collect()

    for doc_idx in range(total_raw_docs):
        doc_lines = doc_lines_list[doc_idx]
        # 処理済み文書の文字列リストを即座に解放してメモリ消費を抑える
        doc_lines_list[doc_idx] = []

        body_ids = lines_to_body_ids(
            doc_lines,
            tokenizer,
            newline_token=newline_token,
            eos_token=eos_token,
        )
        if not body_ids:
            continue

        # 長い1文書でも全ウィンドウを密配列化せず、チャンクの空き行数だけコピーする。
        padded = np.empty(context_length + len(body_ids), dtype=np.int32)
        padded[:context_length] = bos_id
        padded[context_length:] = body_ids
        windows = sliding_window_view(padded[:-1], context_length)
        y = padded[context_length:]
        num_documents += 1
        num_sentences += len(doc_lines)
        del doc_lines, body_ids
        offset = 0
        while offset < len(y):
            end = min(len(y), offset + shard_size - buffered_rows)
            window_buffer.append(windows[offset:end].copy())
            y_buffer.append(y[offset:end].copy())
            buffered_rows += end - offset
            offset = end
            if buffered_rows == shard_size:
                _drain_ready_shards(flush_all=False)
        del windows, y, padded

    del doc_lines_list
    _drain_ready_shards(flush_all=True)

    # 万が一サンプル数が 0 件だった場合も空の x_0001.npz / y_0001.npy を 1 個出力する
    if shard_index == 0:
        _write_one_shard(
            np.empty((0, context_length), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    elapsed = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    outputs_dict: dict = {
        "out_directory": str(out_dir),
        "x_files": x_files,
        "y_files": y_files,
        "x_file": x_files[0] if x_files else None,
        "y_file": y_files[0] if y_files else None,
        "shards": shards_info,
    }
    if out_meta_file is not None:
        outputs_dict["meta_file"] = str(Path(out_meta_file).resolve())

    stats = {
        "num_documents": num_documents,
        "num_sentences": num_sentences,
        "num_samples": total_samples,
        "num_shards": len(shards_info),
        "shard_size": shard_size,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "x_shape": [total_samples, vocab_size],
        "x_nnz": total_x_nnz,
        "y_shape": [total_samples],
    }

    metadata = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "normalized_path": str(normalized_path.resolve()),
            "vocab_file": str(vocab_file.resolve()),
            "model_file": str(tokenizer.model_file.resolve()) if tokenizer.model_file else None,
        },
        "outputs": outputs_dict,
        "params": {
            "context_length": context_length,
            "shard_size": shard_size,
            "shuffle_documents": shuffle_documents,
            "seed": seed,
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


def build_and_save_xy(
    normalized_path: str | Path,
    vocab_file: str | Path,
    out_x_file: str | Path | None = None,
    out_y_file: str | Path | None = None,
    *,
    out_directory: str | Path | None = None,
    out_meta_file: str | Path | None = None,
    model_file: str | Path | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    shard_size: int = DEFAULT_SHARD_SIZE,
    shuffle_documents: bool = True,
    seed: int = DEFAULT_SEED,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
) -> dict:
    """normalized JSON と vocab から 100万サンプル単位で x_0001.npz / y_0001.npy, ... を作成して保存する。

    out_directory が指定された場合はそのディレクトリ配下に、
    out_x_file が指定された場合は Path(out_x_file).parent（またはディレクトリ指定時はそのパス）配下に
    x_0001.npz, x_0002.npz, ... および y_0001.npy, y_0002.npy, ... を出力する。
    """
    if out_directory is not None:
        resolved_out_dir = Path(out_directory)
    elif out_x_file is not None:
        p_x = Path(out_x_file)
        resolved_out_dir = p_x.parent if p_x.suffix.lower() == ".npz" else p_x
    elif out_y_file is not None:
        p_y = Path(out_y_file)
        resolved_out_dir = p_y.parent if p_y.suffix.lower() == ".npy" else p_y
    else:
        raise ValueError("out_directory または out_x_file のいずれかを指定してください")

    return build_and_save_xy_shards(
        normalized_path,
        vocab_file,
        resolved_out_dir,
        out_meta_file=out_meta_file,
        model_file=model_file,
        context_length=context_length,
        shard_size=shard_size,
        shuffle_documents=shuffle_documents,
        seed=seed,
        bos_token=bos_token,
        eos_token=eos_token,
        newline_token=newline_token,
    )
