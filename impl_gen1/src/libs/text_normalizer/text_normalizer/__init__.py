"""日本語テキストの正規化モジュール。

全角統一・括弧/記号類の統一・数字の縮約などを行う。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "unicode_normalize",
    "to_full_width",
    "pre_normalize",
    "unify_brackets",
    "unify_horizontal_bars",
    "unify_alphabet_case",
    "unify_commas",
    "unify_periods",
    "unify_slashes",
    "unify_waves",
    "unify_quotes",
    "unify_digits_012",
    "normalize_text",
]


def unicode_normalize(s: str) -> str:
    # Unicode正規化 (NFKC) を行う
    return unicodedata.normalize("NFKC", s)


def to_full_width(s: str) -> str:
    # 半角 ASCII 印字文字を全角に変換する
    return s.translate(str.maketrans({chr(i): chr(i + 0xFEE0) for i in range(0x21, 0x7F)}))


def pre_normalize(s: str) -> str:
    s = unicode_normalize(s)
    s = to_full_width(s)
    return s


##############################
# 以下の処理は、全て全角統一後に行うことを想定している
##############################


def unify_brackets(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # 始まり括弧として扱う文字
    open_brackets = (
        "（"
        
        "［"
        "｛"
        "〈"
        "《"
        "「"
        "『"
        "【"
        "〔"
        "〖"
        "〘"
        "〚"
        "｟"
        "〿"
        "⦅"
        "⦃"
        "⟨"
        "⟪"
        "⦇"
        "⟦"
        "⟬"
        "⟮"
        "⦉"
        "⦋"
        "⦍"
        "⦏"
        "⦑"
        "⦓"
        "⦕"
        "⦗"
    )

    # 終わり括弧として扱う文字
    close_brackets = (
        "）"
        "］"
        "｝"
        "〉"
        "》"
        "」"
        "』"
        "】"
        "〕"
        "〗"
        "〙"
        "〛"
        "｠"
        "⦆"
        "⦄"
        "⟩"
        "⟫"
        "⦈"
        "⟧"
        "⟭"
        "⟯"
        "⦊"
        "⦌"
        "⦎"
        "⦐"
        "⦒"
        "⦔"
        "⦖"
        "⦘"
    )

    open_pattern = "[" + re.escape(open_brackets) + "]"
    close_pattern = "[" + re.escape(close_brackets) + "]"

    s = re.sub(open_pattern, "（", s)
    s = re.sub(close_pattern, "）", s)

    return s


def unify_horizontal_bars(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提

    horizontal_bars = (
        "－"  # fullwidth hyphen-minus
        "−"  # minus sign
        "‐"  # hyphen
        "-"  # non-breaking hyphen
        "‒"  # figure dash
        "–"  # en dash
        "—"  # em dash
        "―"  # horizontal bar
        "⁃"  # hyphen bullet
        "﹘"  # small em dash
        "﹣"  # small hyphen-minus
        "ー"  # Japanese prolonged sound mark
        "ｰ"  # halfwidth katakana-hiragana prolonged sound mark
        "─"  # box drawings light horizontal
        "━"  # box drawings heavy horizontal
        "╌"  # box drawings light double dash horizontal
        "╍"  # box drawings heavy double dash horizontal
        "┄"  # box drawings light triple dash horizontal
        "┅"  # box drawings heavy triple dash horizontal
        "┈"  # box drawings light quadruple dash horizontal
        "┉"  # box drawings heavy quadruple dash horizontal
        "╴"  # box drawings light left
        "╶"  # box drawings light right
        "╸"  # box drawings heavy left
        "╺"  # box drawings heavy right
    )

    pattern = "[" + re.escape(horizontal_bars) + "]"
    return re.sub(pattern, "－", s)


def unify_alphabet_case(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # 全角英小文字を全角英大文字に統一する

    lower_alphabets = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    upper_alphabets = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"

    translation_table = str.maketrans(lower_alphabets, upper_alphabets)

    return s.translate(translation_table)


def unify_commas(s: str) -> str:
    # 読点・カンマっぽいものを「、」に統一する

    comma_chars = ("、" "，" "," "､" "﹐" "﹑")

    pattern = "[" + re.escape(comma_chars) + "]"
    return re.sub(pattern, "、", s)


def unify_periods(s: str) -> str:
    # 句点・ピリオドっぽいものを「．」に統一する

    period_chars = ("。" "．" "." "｡" "﹒")

    pattern = "[" + re.escape(period_chars) + "]"
    return re.sub(pattern, "．", s)


def unify_slashes(s: str) -> str:
    slash_chars = ("／" "/" "＼" "\\" "⁄" "∕")

    pattern = "[" + re.escape(slash_chars) + "]"
    return re.sub(pattern, "／", s)


def unify_waves(s: str) -> str:
    wave_chars = ("〜" "～" "~" "∼" "∽" "∿")

    pattern = "[" + re.escape(wave_chars) + "]"
    return re.sub(pattern, "～", s)


def unify_quotes(s: str) -> str:
    # 引用符系（同じ記号で囲む系）を「”」に統一する
    # open/close の対応は考慮しない
    quote_chars = ("“" "”" "‘" "’" "〝" "〟" "＂" "＇" '"' "'")

    pattern = "[" + re.escape(quote_chars) + "]"
    return re.sub(pattern, "”", s)


def unify_digits_012(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # [0]     -> ０
    # [1, 2]  -> １
    # [!012]  -> ３

    digit_table = str.maketrans(
        {
            "０": "０",
            "１": "１",
            "２": "１",
            "３": "３",
            "４": "３",
            "５": "３",
            "６": "３",
            "７": "３",
            "８": "３",
            "９": "３",
        }
    )

    return s.translate(digit_table)


def normalize_text(s: str) -> str:
    """テキストに全正規化処理を順に適用する。"""
    s = pre_normalize(s)
    s = unify_brackets(s)
    s = unify_horizontal_bars(s)
    s = unify_alphabet_case(s)
    s = unify_commas(s)
    s = unify_periods(s)
    s = unify_slashes(s)
    s = unify_waves(s)
    s = unify_quotes(s)
    s = unify_digits_012(s)
    return s
