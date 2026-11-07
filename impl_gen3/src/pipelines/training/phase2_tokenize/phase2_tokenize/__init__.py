"""Phase 2 トークン化パイプライン（コーパス作成・SentencePiece 語彙サイズ探索＆トークン化）。

Phase 1 が出力した context.json を元に正規化済み共通フォーマット JSON ファイルを取得し、
1. コーパス作成（バルク化: 1文書を <s>...</s> で囲み、<n>\\n 区切りで 1文 = 1行 として出力）(chabsa_corpus)
2. 初期値 (vocab_size_start) から指数 (vocab_size_exponent) のべき乗で vocab_size を増やしながら
   SentencePiece 学習・トークン化・レポート作成 (report_{vocab_size}.json) を繰り返す
   - 下限割れ (Vocabulary size is smaller than required_chars) はスキップして次へ進む
   - 終わり値 (vocab_size_end) を超えた処理を実行完了するか、上限割れ等のエラーが出た時点で終了
3. 全レポートの「単語数適切指数 (appropriateness_score)」を集計し、最も高い vocab_size を採用して
   context.json に全試行ディレクトリと採用ディレクトリ・各ファイルパスを記録する。
"""

from __future__ import annotations

from collections import Counter
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm

# libs 配下のモジュールを単体実行時でも import できるようにパスを補完する
_LIBS_DIR = Path(__file__).resolve().parents[4] / "libs"
for _lib_name in ("chabsa_corpus", "sp_tokenizer"):
    _lib_path = _LIBS_DIR / _lib_name
    if _lib_path.is_dir() and str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

from chabsa_corpus import (  # noqa: E402
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    build_corpus,
    get_special_tokens,
)
from sp_tokenizer import tokenize_file, train_model  # noqa: E402

__all__ = [
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_NEWLINE_TOKEN",
    "load_context",
    "save_context",
    "resolve_phase1_normalized_path",
    "resolve_phase1_normalized_directory",
    "compute_appropriateness_score",
    "evaluate_vocab_trial",
    "run_phase2_tokenize",
]


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


def resolve_phase1_normalized_path(context: dict) -> Path:
    """Phase 1 の context 情報から正規化済み JSON ファイル（またはディレクトリ）のパスを解決する。"""
    phase1 = context.get("phase1_preprocess", {})
    if isinstance(phase1, dict):
        if phase1.get("normalized_file"):
            return Path(phase1["normalized_file"])
        if phase1.get("normalized_directory"):
            return Path(phase1["normalized_directory"])
        meta_path_str = phase1.get("meta_file") or phase1.get("metadata")
        if meta_path_str:
            meta_path = Path(meta_path_str)
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            outputs = meta.get("outputs", {})
            norm_path = outputs.get("normalized_file") or outputs.get("normalized_directory")
            if norm_path:
                return Path(norm_path)
    raise ValueError(
        "context.json 内に phase1_preprocess の normalized_file (または normalized_directory) パスが見つかりません"
    )


# 後方互換エイリアス
resolve_phase1_normalized_directory = resolve_phase1_normalized_path


def _merge_user_defined_symbols(
    special_tokens: list[str],
    user_defined_symbols: list[str] | None,
) -> list[str]:
    """コーパスの特殊記号 (<s>, </s>, <n> 等) と外部指定の user_defined_symbols を重複なく結合する。"""
    merged: list[str] = []
    for token in [*(special_tokens or []), *(user_defined_symbols or [])]:
        if token and token not in merged:
            merged.append(token)
    return merged


def _is_vocab_too_small_error(exc: Exception) -> bool:
    """SentencePiece の下限割れエラー (smaller than required_chars) かどうかを判定する。"""
    msg = str(exc)
    return "smaller than required_chars" in msg


def _count_corpus_chars(
    corpus_file: Path,
    special_tokens: list[str],
) -> dict:
    """コーパスファイルの文字数を集計する（特殊記号を含む文字数・除いた純粋な本文文字数）。"""
    total_chars_with_special = 0
    total_content_chars = 0
    with corpus_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            total_chars_with_special += len(line)
            content = line
            for token in special_tokens:
                if token:
                    content = content.replace(token, "")
            total_content_chars += len(content)
    return {
        "num_chars": total_content_chars,
        "num_chars_with_special": total_chars_with_special,
    }


def compute_appropriateness_score(
    *,
    chars_per_token: float,
    frequent_vocab_ratio: float,
    normalized_entropy: float,
    unk_ratio: float,
    target_chars_per_token: float = 2.2,
    length_sigma: float = 0.8,
) -> dict:
    """単語数適切指数 (appropriateness_score) とその内訳スコアを計算する。

    - length_score: 1単語あたり文字数が target_chars_per_token (2.2文字) に近いほど1.0に近づくガウススコア
    - active_score: 設定語彙のうちコーパス内で2回以上出現した有効語彙の割合
    - entropy_score: シャノンエントロピーの正規化値 (語彙が偏りなく使われているか)
    - unk_penalty: 未知語 (<unk>) 発生率による減点係数
    """
    length_score = math.exp(
        -((chars_per_token - target_chars_per_token) ** 2) / (2.0 * (length_sigma**2))
    )
    active_score = max(0.0, min(1.0, frequent_vocab_ratio))
    entropy_score = max(0.0, min(1.0, normalized_entropy))
    unk_penalty = max(0.0, 1.0 - 5.0 * unk_ratio)

    appropriateness_score = length_score * active_score * entropy_score * unk_penalty
    return {
        "appropriateness_score": round(appropriateness_score, 6),
        "length_score": round(length_score, 6),
        "active_score": round(active_score, 6),
        "entropy_score": round(entropy_score, 6),
        "unk_penalty": round(unk_penalty, 6),
    }


def evaluate_vocab_trial(
    *,
    vocab_size: int,
    model_file: Path,
    vocab_file: Path,
    tokens_file: Path,
    report_file: Path,
    corpus_file: Path,
    num_documents: int,
    num_sentences: int,
    num_chars: int,
    num_chars_with_special: int,
    special_tokens: list[str],
    elapsed_seconds: float,
) -> dict:
    """学習済み SentencePiece モデルとコーパスから各種統計と単語数適切指数を計算し、report_{vocab_size}.json に保存する。"""
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    actual_vocab_size = sp.get_piece_size()
    unk_id = sp.unk_id()
    special_ids = {sp.piece_to_id(t) for t in special_tokens if t and sp.piece_to_id(t) != unk_id}

    token_counts: Counter[int] = Counter()
    total_tokens = 0
    content_tokens = 0
    unk_count = 0
    num_lines = 0

    with corpus_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            ids = sp.encode(line, out_type=int)
            num_lines += 1
            total_tokens += len(ids)
            for tid in ids:
                token_counts[tid] += 1
                if tid == unk_id:
                    unk_count += 1
                if tid not in special_ids:
                    content_tokens += 1

    active_vocab_count = len(token_counts)
    frequent_vocab_count = sum(1 for _, cnt in token_counts.items() if cnt >= 2)
    singleton_vocab_count = sum(1 for _, cnt in token_counts.items() if cnt == 1)

    active_vocab_ratio = active_vocab_count / actual_vocab_size if actual_vocab_size > 0 else 0.0
    frequent_vocab_ratio = (
        frequent_vocab_count / actual_vocab_size if actual_vocab_size > 0 else 0.0
    )
    unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0.0

    # シャノンエントロピー H = -sum(p * ln(p)) と正規化エントロピー H / ln(V)
    if total_tokens > 0 and actual_vocab_size > 1:
        entropy = -sum(
            (cnt / total_tokens) * math.log(cnt / total_tokens)
            for cnt in token_counts.values()
            if cnt > 0
        )
        normalized_entropy = entropy / math.log(actual_vocab_size)
    else:
        entropy = 0.0
        normalized_entropy = 0.0

    chars_per_document = num_chars / num_documents if num_documents > 0 else 0.0
    sentences_per_document = num_sentences / num_documents if num_documents > 0 else 0.0
    tokens_per_document = total_tokens / num_documents if num_documents > 0 else 0.0
    content_tokens_per_document = content_tokens / num_documents if num_documents > 0 else 0.0

    chars_per_sentence = num_chars / num_sentences if num_sentences > 0 else 0.0
    tokens_per_sentence = total_tokens / num_sentences if num_sentences > 0 else 0.0
    content_tokens_per_sentence = content_tokens / num_sentences if num_sentences > 0 else 0.0

    chars_per_token = num_chars / content_tokens if content_tokens > 0 else 0.0

    score_details = compute_appropriateness_score(
        chars_per_token=chars_per_token,
        frequent_vocab_ratio=frequent_vocab_ratio,
        normalized_entropy=normalized_entropy,
        unk_ratio=unk_ratio,
    )

    report = {
        "vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "appropriateness_score": score_details["appropriateness_score"],
        "score_breakdown": score_details,
        "corpus_stats": {
            "num_documents": num_documents,
            "num_sentences": num_sentences,
            "num_lines": num_lines,
            "num_chars": num_chars,
            "num_chars_with_special": num_chars_with_special,
            "chars_per_document": round(chars_per_document, 3),
            "sentences_per_document": round(sentences_per_document, 3),
            "chars_per_sentence": round(chars_per_sentence, 3),
        },
        "token_stats": {
            "num_tokens": total_tokens,
            "num_content_tokens": content_tokens,
            "tokens_per_document": round(tokens_per_document, 3),
            "content_tokens_per_document": round(content_tokens_per_document, 3),
            "tokens_per_sentence": round(tokens_per_sentence, 3),
            "content_tokens_per_sentence": round(content_tokens_per_sentence, 3),
            "chars_per_token": round(chars_per_token, 4),
            "active_vocab_count": active_vocab_count,
            "active_vocab_ratio": round(active_vocab_ratio, 4),
            "frequent_vocab_count": frequent_vocab_count,
            "frequent_vocab_ratio": round(frequent_vocab_ratio, 4),
            "singleton_vocab_count": singleton_vocab_count,
            "unk_count": unk_count,
            "unk_ratio": round(unk_ratio, 6),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
        },
        "files": {
            "directory": str(model_file.parent),
            "model_file": str(model_file),
            "vocab_file": str(vocab_file),
            "tokens_file": str(tokens_file),
            "report_file": str(report_file),
        },
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def run_phase2_tokenize(
    input_context_file: str | Path,
    out_directory: str | Path,
    out_meta_file: str | Path,
    out_context_file: str | Path,
    *,
    out_corpus_file: str | Path | None = None,
    vocab_size_start: int = 1500,
    vocab_size_end: int = 8000,
    vocab_size_exponent: float = 1.05,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    user_defined_symbols: list[str] | None = None,
    bos_id: int = -1,
    eos_id: int = -1,
    output_format: str = "piece",
    add_bos: bool = False,
    add_eos: bool = False,
) -> dict:
    """Phase 1 の context.json を元にコーパス作成と vocab_size べき乗探索を実行し、最良モデルを選定して context.json を出力する。"""
    if vocab_size_start < 2:
        raise ValueError(f"vocab_size_start は 2 以上を指定してください: {vocab_size_start}")
    if vocab_size_exponent <= 1.0:
        raise ValueError(
            f"vocab_size_exponent は 1.0 より大きい値を指定してください: {vocab_size_exponent}"
        )

    input_context_file = Path(input_context_file)
    context = load_context(input_context_file)
    input_normalized_path = resolve_phase1_normalized_path(context)

    out_directory = Path(out_directory)
    out_corpus_path = (
        Path(out_corpus_file) if out_corpus_file is not None else out_directory / "corpus.txt"
    )
    out_meta_file = Path(out_meta_file)
    out_context_file = Path(out_context_file)

    total_started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()

    # 1. コーパス作成（1文書を <s>...</s> で囲み、<n>\n 区切りの 1文 = 1行 形式）
    corpus_started_at = datetime.now(timezone.utc)
    corpus_start = time.perf_counter()
    corpus_stats = build_corpus(
        input_normalized_path,
        out_corpus_path,
        newline_token=newline_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    special_tokens = get_special_tokens(
        newline_token=newline_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    char_stats = _count_corpus_chars(out_corpus_path, special_tokens)
    corpus_stats.update(char_stats)

    corpus_elapsed = time.perf_counter() - corpus_start
    corpus_finished_at = datetime.now(timezone.utc)

    corpus_meta = {
        "started_at": corpus_started_at.isoformat(),
        "finished_at": corpus_finished_at.isoformat(),
        "elapsed_seconds": round(corpus_elapsed, 3),
        "input": {
            "normalized_file": str(input_normalized_path),
        },
        "outputs": {
            "corpus_file": str(out_corpus_path),
        },
        "params": {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "newline_token": newline_token,
        },
        "stats": corpus_stats,
    }

    # 2. vocab_size をべき乗 ((start ^ exp) ^ exp ...) で増やしながら学習・トークン化・レポート作成
    effective_symbols = _merge_user_defined_symbols(special_tokens, user_defined_symbols)

    trials: list[dict] = []
    skipped_trials: list[dict] = []
    vocab_directories: dict[str, str] = {}
    stopped_reason = ""
    error_detail: dict | None = None

    current_val = float(vocab_size_start)
    prev_vocab_size = 0

    while True:
        vocab_size = int(round(current_val))
        if vocab_size <= prev_vocab_size:
            vocab_size = prev_vocab_size + 1
            current_val = float(vocab_size)

        trial_dir = out_directory / f"vocab{vocab_size}"
        trial_model_file = trial_dir / "sp.model"
        trial_vocab_file = trial_dir / "sp.vocab"
        trial_tokens_file = trial_dir / "tokens.txt"
        trial_report_file = trial_dir / f"report_{vocab_size}.json"

        trial_start = time.perf_counter()
        try:
            train_model(
                out_corpus_path,
                trial_model_file,
                trial_vocab_file,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                user_defined_symbols=effective_symbols,
                bos_id=bos_id,
                eos_id=eos_id,
            )
            tokenize_file(
                trial_model_file,
                out_corpus_path,
                trial_tokens_file,
                output_format=output_format,
                add_bos=add_bos,
                add_eos=add_eos,
            )
            trial_elapsed = time.perf_counter() - trial_start

            report = evaluate_vocab_trial(
                vocab_size=vocab_size,
                model_file=trial_model_file,
                vocab_file=trial_vocab_file,
                tokens_file=trial_tokens_file,
                report_file=trial_report_file,
                corpus_file=out_corpus_path,
                num_documents=corpus_stats["num_documents"],
                num_sentences=corpus_stats["num_sentences"],
                num_chars=corpus_stats["num_chars"],
                num_chars_with_special=corpus_stats["num_chars_with_special"],
                special_tokens=effective_symbols,
                elapsed_seconds=trial_elapsed,
            )
            trials.append(report)
            vocab_directories[f"vocab{vocab_size}"] = str(trial_dir)
        except Exception as exc:
            if trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

            # 下限割れエラー (smaller than required_chars) はスキップして次へ進む
            if _is_vocab_too_small_error(exc):
                skipped_trials.append(
                    {
                        "vocab_size": vocab_size,
                        "reason": "vocab_size_too_small",
                        "error_message": str(exc),
                    }
                )
                if vocab_size > vocab_size_end:
                    stopped_reason = f"exceeded_vocab_size_end ({vocab_size} > {vocab_size_end})"
                    break
                prev_vocab_size = vocab_size
                current_val = current_val**vocab_size_exponent
                continue

            # 上限割れ (Vocabulary size too high) やその他のエラーは探索終了
            stopped_reason = f"error_at_vocab_{vocab_size}"
            error_detail = {
                "failed_vocab_size": vocab_size,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            break

        # 終わり値を超えた処理を実行した場合はここで終了
        if vocab_size > vocab_size_end:
            stopped_reason = f"exceeded_vocab_size_end ({vocab_size} > {vocab_size_end})"
            break

        prev_vocab_size = vocab_size
        current_val = current_val**vocab_size_exponent

    if not trials:
        msg = "有効な vocab_size の学習結果が 1 件も得られませんでした。"
        if error_detail:
            msg += f" (vocab_size={error_detail['failed_vocab_size']}: {error_detail['error_message']})"
        elif skipped_trials:
            last_skip = skipped_trials[-1]
            msg += f" (最終スキップ vocab_size={last_skip['vocab_size']}: {last_skip['error_message']})"
        raise RuntimeError(msg)

    # 3. 全レポートを集計して単語数適切指数 (appropriateness_score) が最大のものを採用
    best_trial = max(trials, key=lambda r: (r["appropriateness_score"], -r["vocab_size"]))
    best_files = best_trial["files"]

    total_elapsed = time.perf_counter() - total_start
    total_finished_at = datetime.now(timezone.utc)

    trials_summary = [
        {
            "vocab_size": r["vocab_size"],
            "actual_vocab_size": r["actual_vocab_size"],
            "appropriateness_score": r["appropriateness_score"],
            "chars_per_token": r["token_stats"]["chars_per_token"],
            "frequent_vocab_ratio": r["token_stats"]["frequent_vocab_ratio"],
            "normalized_entropy": r["token_stats"]["normalized_entropy"],
            "num_tokens": r["token_stats"]["num_tokens"],
            "directory": r["files"]["directory"],
            "report_file": r["files"]["report_file"],
        }
        for r in trials
    ]

    outputs_dict = {
        "out_directory": str(out_directory),
        "corpus_file": str(out_corpus_path),
        "vocab_directories": vocab_directories,
        "selected_vocab_size": best_trial["vocab_size"],
        "selected_directory": best_files["directory"],
        "model_file": best_files["model_file"],
        "vocab_file": best_files["vocab_file"],
        "tokens_file": best_files["tokens_file"],
        "report_file": best_files["report_file"],
        "meta_file": str(out_meta_file),
        "context_file": str(out_context_file),
    }

    metadata = {
        "started_at": total_started_at.isoformat(),
        "finished_at": total_finished_at.isoformat(),
        "elapsed_seconds": round(total_elapsed, 3),
        "input": {
            "input_context_file": str(input_context_file),
            "normalized_file": str(input_normalized_path),
        },
        "outputs": outputs_dict,
        "params": {
            "vocab_size_start": vocab_size_start,
            "vocab_size_end": vocab_size_end,
            "vocab_size_exponent": vocab_size_exponent,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "newline_token": newline_token,
            "model_type": model_type,
            "character_coverage": character_coverage,
            "user_defined_symbols": effective_symbols,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "output_format": output_format,
            "add_bos": add_bos,
            "add_eos": add_eos,
        },
        "search_result": {
            "num_trials": len(trials),
            "num_skipped": len(skipped_trials),
            "skipped_trials": skipped_trials,
            "stopped_reason": stopped_reason,
            "error_detail": error_detail,
            "selected_vocab_size": best_trial["vocab_size"],
            "selected_appropriateness_score": best_trial["appropriateness_score"],
            "trials": trials_summary,
        },
        "steps": {
            "corpus": corpus_meta,
            "selected_trial": best_trial,
        },
        "stats": {
            "num_documents": corpus_stats["num_documents"],
            "num_sentences": corpus_stats["num_sentences"],
            "num_lines": corpus_stats["num_lines"],
            "num_chars": corpus_stats["num_chars"],
            "selected_vocab_size": best_trial["vocab_size"],
            "selected_appropriateness_score": best_trial["appropriateness_score"],
            "num_tokens": best_trial["token_stats"]["num_tokens"],
        },
    }

    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Phase 3 用の context.json を出力（各 vocab{size} ディレクトリと、採用された代表パスを保存）
    phase2_context = {
        "input_context_file": str(input_context_file),
        "normalized_file": str(input_normalized_path),
        "out_directory": str(out_directory),
        "corpus_file": str(out_corpus_path),
        "vocab_directories": vocab_directories,
        "selected_vocab_size": best_trial["vocab_size"],
        "selected_directory": best_files["directory"],
        "model_file": best_files["model_file"],
        "vocab_file": best_files["vocab_file"],
        "tokens_file": best_files["tokens_file"],
        "report_file": best_files["report_file"],
        "meta_file": str(out_meta_file),
    }

    context["phase2_tokenize"] = phase2_context
    save_context(context, out_context_file)

    return metadata
