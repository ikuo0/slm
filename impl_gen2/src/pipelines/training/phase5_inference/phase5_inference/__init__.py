"""Phase 5 推論・文章生成パイプライン (impl_gen2: 3ハッシュインデックス圧縮ベクトル対応)。

Phase 4 が出力した context.json から、
- 学習済み NN モデル (model.keras, 入力次元=vector_dim, 出力次元=vocab_size)
- SentencePiece 語彙ファイル (sp.vocab) およびモデルファイル (sp.model)
- 圧縮次元数 (vector_dim, デフォルト 256)、ハッシュ数 (num_hashes, デフォルト 3)、文脈ウィンドウ長 (context_length=32) 等
を取得し、先頭 32 個の <s> から各トークンを 3 つのハッシュインデックス
(`h1idx = hash1(word) % vector_dim`, `h2idx = hash2(word) % vector_dim`, `h3idx = hash3(word) % vector_dim`)
に写像して重み付き加算を行い、自己回帰で 1 トークンずつ予測して文章を生成する。

終了条件:
- 終了トークン (</s>) が予測された時点 ("eos")
- または生成された本文文字数が max_chars に達した時点 ("max_chars")

サンプリング方式:
- "greedy": 常に最大確率の単語を選択
- "random": 各単語の確率 p_i を p_i ** prob_power してから正規化した分布でランダム抽選
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

# libs 配下のモジュールを単体実行時でも import できるようにパスを補完する
_LIBS_DIR = Path(__file__).resolve().parents[4] / "libs"
for _lib_name in ("weighted_onehot", "runtime_stats"):
    _lib_path = _LIBS_DIR / _lib_name
    if _lib_path.is_dir() and str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

from runtime_stats import get_memory_stats  # noqa: E402
from weighted_onehot import (  # noqa: E402
    DEFAULT_BOS_TOKEN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    DEFAULT_NUM_HASHES,
    DEFAULT_VECTOR_DIM,
    VocabTokenizer,
)

__all__ = [
    "DEFAULT_NUM_DOCUMENTS",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_SAMPLING_MODE",
    "DEFAULT_PROB_POWER",
    "DEFAULT_VECTOR_DIM",
    "DEFAULT_NUM_HASHES",
    "load_context",
    "save_context",
    "resolve_inference_resources",
    "sample_next_token_id",
    "generate_single_document",
    "format_readable_text",
    "run_phase5_inference",
]

DEFAULT_NUM_DOCUMENTS = 10
DEFAULT_MAX_CHARS = 1000
DEFAULT_SAMPLING_MODE = "random"
DEFAULT_PROB_POWER = 2.0


def load_context(context_file: str | Path) -> dict:
    """context.json を読み込んで辞書として返す。"""
    path = Path(context_file)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, dict) else {}


def save_context(context: dict, out_context_file: str | Path) -> None:
    """ディレクトリ・ファイルのパス情報を格納した辞書を context.json に書き出す。"""
    path = Path(out_context_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)


def resolve_inference_resources(context: dict) -> dict:
    """context.json から推論に必要な model_file, vocab_file, sp_model_file, vector_dim, num_hashes, context_length 等を解決する。"""
    # 1. NN モデルのパス (phase4_training)
    nn_model_file: Path | None = None
    vector_dim: int | None = None
    num_hashes: int = DEFAULT_NUM_HASHES
    phase4 = context.get("phase4_training", {})
    if isinstance(phase4, dict):
        if phase4.get("vector_dim"):
            vector_dim = int(phase4["vector_dim"])
        if phase4.get("model_file"):
            nn_model_file = Path(phase4["model_file"])
        elif phase4.get("meta_file"):
            meta_path = Path(phase4["meta_file"])
            if meta_path.is_file():
                with meta_path.open(encoding="utf-8") as f:
                    meta4 = json.load(f)
                m_str = meta4.get("outputs", {}).get("model_file")
                if m_str:
                    nn_model_file = Path(m_str)
                if vector_dim is None and meta4.get("params", {}).get("vector_dim"):
                    vector_dim = int(meta4["params"]["vector_dim"])

    if nn_model_file is None:
        raise ValueError("context.json 内に phase4_training の model_file パスが見つかりません")

    # 2. SentencePiece の vocab_file / model_file および vector_dim, num_hashes (phase3_featurize または phase2_tokenize)
    vocab_file: Path | None = None
    sp_model_file: Path | None = None
    context_length = DEFAULT_CONTEXT_LENGTH
    bos_token = DEFAULT_BOS_TOKEN
    eos_token = DEFAULT_EOS_TOKEN
    newline_token = DEFAULT_NEWLINE_TOKEN

    phase3 = context.get("phase3_featurize", {})
    if isinstance(phase3, dict):
        if phase3.get("vocab_file"):
            vocab_file = Path(phase3["vocab_file"])
        if phase3.get("model_file"):
            sp_model_file = Path(phase3["model_file"])
        if vector_dim is None and phase3.get("vector_dim"):
            vector_dim = int(phase3["vector_dim"])
        if phase3.get("num_hashes"):
            num_hashes = int(phase3["num_hashes"])
        if phase3.get("meta_file"):
            meta3_path = Path(phase3["meta_file"])
            if meta3_path.is_file():
                with meta3_path.open(encoding="utf-8") as f:
                    meta3 = json.load(f)
                params3 = meta3.get("params", {})
                if vector_dim is None and params3.get("vector_dim"):
                    vector_dim = int(params3["vector_dim"])
                if params3.get("num_hashes"):
                    num_hashes = int(params3["num_hashes"])
                context_length = int(params3.get("context_length", context_length))
                bos_token = str(params3.get("bos_token", bos_token))
                eos_token = str(params3.get("eos_token", eos_token))
                newline_token = str(params3.get("newline_token", newline_token))

    phase2 = context.get("phase2_tokenize", {})
    if isinstance(phase2, dict):
        if vocab_file is None and phase2.get("vocab_file"):
            vocab_file = Path(phase2["vocab_file"])
        if sp_model_file is None and phase2.get("model_file"):
            sp_model_file = Path(phase2["model_file"])

    if vocab_file is None:
        raise ValueError("context.json 内に vocab_file パスが見つかりません")

    return {
        "nn_model_file": nn_model_file,
        "vocab_file": vocab_file,
        "sp_model_file": sp_model_file,
        "vector_dim": vector_dim,
        "num_hashes": num_hashes,
        "context_length": context_length,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "newline_token": newline_token,
    }


def _piece_to_surface(piece: str) -> str:
    """SentencePiece のピース文字列から先頭ダミースペース (▁) を除去した表層文字列を返す。"""
    return piece.replace("▁", "")


def sample_next_token_id(
    probs: np.ndarray,
    *,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    prob_power: float = DEFAULT_PROB_POWER,
    rng: np.random.Generator | None = None,
    banned_ids: tuple[int, ...] = (),
) -> tuple[int, float]:
    """出力確率ベクトルから次のトークン ID とその元の予測確率を返す。"""
    p = np.asarray(probs, dtype=np.float64).copy()
    p = np.maximum(p, 0.0)
    for bid in banned_ids:
        if 0 <= bid < p.shape[0]:
            p[bid] = 0.0

    if sampling_mode == "greedy":
        chosen_id = int(np.argmax(p))
        return chosen_id, float(probs[chosen_id])

    if sampling_mode != "random":
        raise ValueError(f"未対応の sampling_mode です: {sampling_mode}")

    if rng is None:
        rng = np.random.default_rng()

    p_max = float(np.max(p))
    if p_max <= 0.0:
        chosen_id = int(rng.integers(0, p.shape[0]))
        return chosen_id, float(probs[chosen_id])

    if prob_power != 1.0:
        # 数値アンダーフロー・オーバーフローを防ぐため最大値で正規化してからべき乗する
        scaled = (p / p_max) ** float(prob_power)
    else:
        scaled = p

    total = float(np.sum(scaled))
    if total <= 0.0 or not np.isfinite(total):
        chosen_id = int(np.argmax(p))
        return chosen_id, float(probs[chosen_id])

    norm_p = scaled / total
    chosen_id = int(rng.choice(p.shape[0], p=norm_p))
    return chosen_id, float(probs[chosen_id])


def generate_single_document(
    model: tf.keras.Model,
    tokenizer: VocabTokenizer,
    *,
    vector_dim: int | None = None,
    num_hashes: int = DEFAULT_NUM_HASHES,
    id_to_hash_index: np.ndarray | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    prob_power: float = DEFAULT_PROB_POWER,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    rng: np.random.Generator | None = None,
) -> dict:
    """先頭 context_length 個の <s> から開始し、</s> が出現するか max_chars に達するまで1文書を生成する。"""
    effective_vector_dim = (
        int(vector_dim)
        if vector_dim is not None
        else (int(model.input_shape[-1]) if model.input_shape and model.input_shape[-1] else DEFAULT_VECTOR_DIM)
    )
    if id_to_hash_index is None:
        hash_table = tokenizer.build_hash_map(
            vector_dim=effective_vector_dim,
            num_hashes=num_hashes,
        )
    elif id_to_hash_index.ndim == 1:
        hash_table = id_to_hash_index[:, None]
    else:
        hash_table = id_to_hash_index

    bos_id = tokenizer.token_to_id(bos_token)
    eos_id = tokenizer.token_to_id(eos_token)
    newline_id = tokenizer.token_to_id(newline_token)
    unk_id = tokenizer.unk_id

    # 学習時と同じく、文脈内の位置 j (0 <= j < context_length) の重みは 1 / (context_length - j)
    weights = np.array(
        [1.0 / (context_length - j) for j in range(context_length)],
        dtype=np.float32,
    )
    window = np.full(context_length, bos_id, dtype=np.int32)
    x_buf = np.zeros((1, effective_vector_dim), dtype=np.float32)

    # <s> と <unk> は生成対象から除外する
    banned_ids = (bos_id, unk_id)

    generated_ids: list[int] = []
    generated_pieces: list[str] = []
    num_chars = 0
    stopped_reason = "max_chars"

    # <n> ばかり連続生成された場合の無限ループ防止上限
    max_step_limit = max(max_chars * 4, 200)

    for _ in range(max_step_limit):
        x_buf.fill(0.0)
        # ウィンドウ内の各単語について 3 つのハッシュインデックス (h1idx, h2idx, h3idx) それぞれに weight を加算
        window_hashes = hash_table[window]  # shape: (context_length, num_hashes)
        for h_col in range(window_hashes.shape[1]):
            np.add.at(x_buf[0], window_hashes[:, h_col], weights)

        # 1サンプル推論では model(x_buf, training=False) が高速
        preds = model(x_buf, training=False).numpy()[0]

        next_id, _ = sample_next_token_id(
            preds,
            sampling_mode=sampling_mode,
            prob_power=prob_power,
            rng=rng,
            banned_ids=banned_ids,
        )

        if next_id == eos_id:
            stopped_reason = "eos"
            break

        piece = tokenizer.pieces[next_id] if 0 <= next_id < len(tokenizer.pieces) else ""
        generated_ids.append(next_id)
        generated_pieces.append(piece)

        if next_id != newline_id and piece != newline_token:
            surface = _piece_to_surface(piece)
            num_chars += len(surface)

        # ウィンドウを1つ右へスライド
        window[:-1] = window[1:]
        window[-1] = next_id

        if num_chars >= max_chars:
            stopped_reason = "max_chars"
            break
    else:
        stopped_reason = "max_steps"

    # <n> を実際の改行 \n に変換したテキストと、そのままの raw_text を組み立てる
    raw_parts: list[str] = []
    text_parts: list[str] = []
    for tid, piece in zip(generated_ids, generated_pieces):
        if tid == newline_id or piece == newline_token:
            raw_parts.append(newline_token)
            text_parts.append("\n")
        else:
            surf = _piece_to_surface(piece)
            raw_parts.append(surf)
            text_parts.append(surf)

    raw_text = "".join(raw_parts)
    text = "".join(text_parts)
    lines = text.split("\n")

    return {
        "stopped_reason": stopped_reason,
        "num_chars": num_chars,
        "num_tokens": len(generated_ids),
        "num_sentences": len(lines),
        "lines": lines,
        "text": text,
        "raw_text": raw_text,
    }


def format_readable_text(documents: list[dict], params: dict | None = None) -> str:
    """目視確認用のプレーンテキスト（# {生成番号} {簡単な推定情報}\\n{生成文章}\\n\\n）を組み立てる。"""
    blocks: list[str] = []
    mode_info = ""
    if params:
        mode = params.get("sampling_mode", "")
        if mode == "random":
            mode_info = f", mode=random(power={params.get('prob_power')})"
        elif mode:
            mode_info = f", mode={mode}"
        if params.get("vector_dim"):
            mode_info += f", dim={params['vector_dim']}"
        if params.get("num_hashes"):
            mode_info += f", hashes={params['num_hashes']}"

    for doc in documents:
        header = (
            f"# {doc['document_id']} "
            f"[stopped={doc['stopped_reason']}, "
            f"chars={doc['num_chars']}, "
            f"sentences={doc['num_sentences']}, "
            f"tokens={doc['num_tokens']}, "
            f"elapsed={doc['elapsed_seconds']}s"
            f"{mode_info}]"
        )
        blocks.append(f"{header}\n{doc['text']}\n\n")
    return "".join(blocks)


def run_phase5_inference(
    input_context_file: str | Path,
    out_texts_file: str | Path,
    out_meta_file: str | Path,
    out_context_file: str | Path,
    *,
    out_readable_file: str | Path | None = None,
    num_documents: int = DEFAULT_NUM_DOCUMENTS,
    max_chars: int = DEFAULT_MAX_CHARS,
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    prob_power: float = DEFAULT_PROB_POWER,
    seed: int = 42,
    vector_dim: int | None = None,
    num_hashes: int | None = None,
    nn_model_file: str | Path | None = None,
    vocab_file: str | Path | None = None,
    sp_model_file: str | Path | None = None,
    context_length: int | None = None,
    bos_token: str | None = None,
    eos_token: str | None = None,
    newline_token: str | None = None,
) -> dict:
    """Phase 4 の context.json を元に文章生成を実行し、生成結果・目視用テキスト・metadata.json・context.json を出力する。

    Args:
        input_context_file: Phase 4 が出力した context.json のファイルパス
        out_texts_file: 生成した文章一覧の出力ファイルパス (JSON)
        out_meta_file: メタデータの出力ファイルパス (JSON)
        out_context_file: context.json の出力ファイルパス (JSON)
        out_readable_file: 目視確認用のプレーンテキスト出力ファイルパス (任意)
        num_documents: 生成する文章（文書）の本数 (default: 10)
        max_chars: 1文書あたりの最大文字数 (</s> が現れなくても打ち切る上限、default: 1000)
        sampling_mode: "random" または "greedy" (default: "random")
        prob_power: random 時に確率を何乗してから正規化するか (default: 2.0)
        seed: 乱数シード (default: 42)
        vector_dim: 圧縮ベクトル次元数の上書き指定 (省略時は context.json またはモデルの入力次元から自動取得)
        num_hashes: 1単語あたりに使用するハッシュインデックス数の上書き指定 (省略時は context.json から自動取得、default: 3)
        nn_model_file: 学習済み NN モデル (.keras) を明示的に上書き指定する場合のパス (任意)
        vocab_file: SentencePiece の .vocab ファイルを明示的に上書き指定する場合のパス (任意)
        sp_model_file: SentencePiece の .model ファイルを明示的に上書き指定する場合のパス (任意)
        context_length: 文脈ウィンドウ長の上書き指定 (任意)
        bos_token: 開始記号の上書き指定 (任意)
        eos_token: 終了記号の上書き指定 (任意)
        newline_token: 文区切り記号の上書き指定 (任意)

    Returns:
        出力したメタデータの辞書
    """
    if num_documents < 1:
        raise ValueError(f"num_documents は 1 以上を指定してください: {num_documents}")
    if max_chars < 1:
        raise ValueError(f"max_chars は 1 以上を指定してください: {max_chars}")
    if sampling_mode not in ("random", "greedy"):
        raise ValueError(f"sampling_mode は 'random' または 'greedy' を指定してください: {sampling_mode}")
    if prob_power <= 0.0:
        raise ValueError(f"prob_power は 0 より大きい値を指定してください: {prob_power}")

    input_context_file = Path(input_context_file)
    context = load_context(input_context_file)
    resolved = resolve_inference_resources(context)

    resolved_nn_model = Path(nn_model_file) if nn_model_file is not None else resolved["nn_model_file"]
    resolved_vocab = Path(vocab_file) if vocab_file is not None else resolved["vocab_file"]
    resolved_sp_model = (
        Path(sp_model_file) if sp_model_file is not None else resolved["sp_model_file"]
    )
    resolved_num_hashes = (
        int(num_hashes) if num_hashes is not None else int(resolved["num_hashes"])
    )
    resolved_context_length = (
        int(context_length) if context_length is not None else int(resolved["context_length"])
    )
    resolved_bos = str(bos_token) if bos_token is not None else str(resolved["bos_token"])
    resolved_eos = str(eos_token) if eos_token is not None else str(resolved["eos_token"])
    resolved_newline = (
        str(newline_token) if newline_token is not None else str(resolved["newline_token"])
    )

    out_texts_file = Path(out_texts_file)
    out_meta_file = Path(out_meta_file)
    out_context_file = Path(out_context_file)
    resolved_readable_file = Path(out_readable_file) if out_readable_file is not None else None

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()

    tokenizer = VocabTokenizer(resolved_vocab, model_file=resolved_sp_model)
    model = tf.keras.models.load_model(str(resolved_nn_model))

    if vector_dim is not None:
        resolved_vector_dim = int(vector_dim)
    elif resolved["vector_dim"] is not None:
        resolved_vector_dim = int(resolved["vector_dim"])
    elif model.input_shape and model.input_shape[-1]:
        resolved_vector_dim = int(model.input_shape[-1])
    else:
        resolved_vector_dim = DEFAULT_VECTOR_DIM

    id_to_hash_index = tokenizer.build_hash_map(
        vector_dim=resolved_vector_dim,
        num_hashes=resolved_num_hashes,
    )
    rng = np.random.default_rng(seed)

    documents: list[dict] = []
    for doc_idx in range(1, num_documents + 1):
        doc_start = time.perf_counter()
        doc_res = generate_single_document(
            model,
            tokenizer,
            vector_dim=resolved_vector_dim,
            num_hashes=resolved_num_hashes,
            id_to_hash_index=id_to_hash_index,
            max_chars=max_chars,
            sampling_mode=sampling_mode,
            prob_power=prob_power,
            context_length=resolved_context_length,
            bos_token=resolved_bos,
            eos_token=resolved_eos,
            newline_token=resolved_newline,
            rng=rng,
        )
        doc_elapsed = time.perf_counter() - doc_start
        documents.append(
            {
                "document_id": doc_idx,
                "stopped_reason": doc_res["stopped_reason"],
                "num_chars": doc_res["num_chars"],
                "num_tokens": doc_res["num_tokens"],
                "num_sentences": doc_res["num_sentences"],
                "elapsed_seconds": round(doc_elapsed, 3),
                "lines": doc_res["lines"],
                "text": doc_res["text"],
                "raw_text": doc_res["raw_text"],
            }
        )

    elapsed = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc)

    eos_count = sum(1 for d in documents if d["stopped_reason"] == "eos")
    max_chars_count = sum(1 for d in documents if d["stopped_reason"] == "max_chars")
    avg_chars = sum(d["num_chars"] for d in documents) / len(documents)
    avg_tokens = sum(d["num_tokens"] for d in documents) / len(documents)
    avg_sentences = sum(d["num_sentences"] for d in documents) / len(documents)
    mem_stats = get_memory_stats()

    stats = {
        "num_documents": len(documents),
        "vector_dim": resolved_vector_dim,
        "num_hashes": resolved_num_hashes,
        "vocab_size": tokenizer.vocab_size,
        "eos_stopped_count": eos_count,
        "max_chars_stopped_count": max_chars_count,
        "avg_chars": round(avg_chars, 2),
        "avg_tokens": round(avg_tokens, 2),
        "avg_sentences": round(avg_sentences, 2),
        "peak_rss_kb": mem_stats["peak_rss_kb"],
        "peak_rss_mb": mem_stats["peak_rss_mb"],
        "current_rss_kb": mem_stats["current_rss_kb"],
        "current_rss_mb": mem_stats["current_rss_mb"],
        "peak_vm_kb": mem_stats["peak_vm_kb"],
        "peak_vm_mb": mem_stats["peak_vm_mb"],
        "memory": mem_stats,
    }

    params_dict = {
        "num_documents": num_documents,
        "max_chars": max_chars,
        "sampling_mode": sampling_mode,
        "prob_power": prob_power,
        "seed": seed,
        "vector_dim": resolved_vector_dim,
        "num_hashes": resolved_num_hashes,
        "context_length": resolved_context_length,
        "bos_token": resolved_bos,
        "eos_token": resolved_eos,
        "newline_token": resolved_newline,
    }

    texts_payload = {
        "generated_at": finished_at.isoformat(),
        "params": params_dict,
        "stats": stats,
        "documents": documents,
    }
    out_texts_file.parent.mkdir(parents=True, exist_ok=True)
    with out_texts_file.open("w", encoding="utf-8") as f:
        json.dump(texts_payload, f, ensure_ascii=False, indent=2)

    if resolved_readable_file is not None:
        readable_str = format_readable_text(documents, params=params_dict)
        resolved_readable_file.parent.mkdir(parents=True, exist_ok=True)
        resolved_readable_file.write_text(readable_str, encoding="utf-8")

    outputs_dict = {
        "texts_file": str(out_texts_file),
        "readable_file": str(resolved_readable_file) if resolved_readable_file else None,
        "meta_file": str(out_meta_file),
        "context_file": str(out_context_file),
    }

    metadata = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "input_context_file": str(input_context_file),
            "nn_model_file": str(resolved_nn_model),
            "vocab_file": str(resolved_vocab),
            "sp_model_file": str(resolved_sp_model) if resolved_sp_model else None,
        },
        "outputs": outputs_dict,
        "params": params_dict,
        "stats": stats,
    }

    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    context["phase5_inference"] = {
        "input_context_file": str(input_context_file),
        "nn_model_file": str(resolved_nn_model),
        "vocab_file": str(resolved_vocab),
        "sp_model_file": str(resolved_sp_model) if resolved_sp_model else None,
        "vector_dim": resolved_vector_dim,
        "num_hashes": resolved_num_hashes,
        "texts_file": str(out_texts_file),
        "readable_file": str(resolved_readable_file) if resolved_readable_file else None,
        "meta_file": str(out_meta_file),
    }
    save_context(context, out_context_file)

    return metadata
