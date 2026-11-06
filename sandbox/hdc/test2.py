import numpy as np

DIM = 10000

# 1. 各単語に「高密度ランダムベクトル (+1/-1)」を割り当てる（One-Hotではない）
def create_hvector(dim):
    return np.random.choice([-1, 1], size=dim)

VOCAB = {
    "Cat":  create_hvector(DIM),
    "Fish": create_hvector(DIM),
    "Eat":  create_hvector(DIM),
}

def permute(v, shift=1):
    """巡回シフト（行列演算ではなく、単なるメモリのローテーション O(D)）"""
    return np.roll(v, shift)

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / DIM

# 2. 文脈ベクトルの逐次更新: C_t = roll(C_{t-1}) + Word_t
context = np.zeros(DIM)

# 文: "Cat" -> "Eat" -> "Fish"
context = permute(context) + VOCAB["Cat"]   # 1単語目
context = permute(context) + VOCAB["Eat"]   # 2単語目
context = permute(context) + VOCAB["Fish"]  # 3単語目

# 展開すると: context = permute(Cat, 2) + permute(Eat, 1) + Fish

# --------------------------------------------------
# 検証: 文脈ベクトルから過去の履歴をクエリする
# --------------------------------------------------

# Q1. 「直前（0ステップ前）に来た単語は Fish か？」
print(f"直前が Fish である類似度: {cosine_sim(context, VOCAB['Fish']):.3f}")  # ≒ 1.0 (高い)
print(f"直前が Cat  である類似度: {cosine_sim(context, VOCAB['Cat']):.3f}")   # ≒ 0.0 (直交)

# Q2. 「2つ前（2ステップ前）に来た単語は Cat か？」
# 2回シフトした Cat と比較する
cat_2_steps_ago = permute(VOCAB["Cat"], 2)
fish_2_steps_ago = permute(VOCAB["Fish"], 2)

print(f"\n2つ前が Cat  である類似度: {cosine_sim(context, cat_2_steps_ago):.3f}")   # ≒ 1.0 (高い)
print(f"2つ前が Fish である類似度: {cosine_sim(context, fish_2_steps_ago):.3f}")  # ≒ 0.0 (直交)


"""
python sandbox/hdc/test2.py
"""
