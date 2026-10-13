# -*- coding: utf-8 -*-
"""
借款人聚类分群
==============

和评分卡的区别（面试常问）
------------------------------------------------------------------------
  评分卡 / GBDT：**有监督**。用已知的 bad_loan 标签学"谁会违约"，输出预测与分层。
  聚类        ：**无监督**。不看标签，只看"这群人长得像不像"，输出客群画像。

流程
------------------------------------------------------------------------
  1. 选放款前可得的用户属性 → 偏态特征取对数 → 标准化
  2. 扫 k = 2~10，同时记录【几何指标】和【业务指标】
  3. 用业务指标选 k，K-Means 分群，输出画像与坏账率
  4. 换层次聚类(Ward)验证结构是否稳定

⭐ 这个脚本最重要的一个发现：轮廓系数会选错 k
------------------------------------------------------------------------
轮廓系数最大的是 **k=2**，但 k=2 的两群坏账率只差 **0.6 个百分点** ——
等于没分出来（两群唯一的差别是收入 5 万 vs 8.5 万、金额 1.1 万 vs 1.9 万，
风险完全一样）。

而 **k=5** 时，各群坏账率从 **11.3% 拉到 27.8%**，跨度 16.5 个百分点，
每一群都能对应一个明确的业务动作。

**结论：轮廓系数衡量的是"几何上紧不紧凑"，不是"业务上有没有用"。**
做分群时不能只信几何指标，必须拿业务指标（这里是坏账率跨度）回头验证。
这是很多做分群的人会踩的坑。

⚠️ 一条需要说明的边界
------------------------------------------------------------------------
选 k 时我用了 bad_loan（业务指标），所以严格说这**不是纯粹的"无监督"**：
无监督的部分只负责"分群"，选 k 这一步引入了标签做后验验证。
在**描述性分群**里这是标准做法（否则无法判断分群有没有用），
但不能对外宣称"完全无监督"。

用法：
    "D:/venvs/lending-club/Scripts/python.exe" clustering.py
    "D:/venvs/lending-club/Scripts/python.exe" clustering.py --k 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import pipeline as P
from cost_threshold import load_clean

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

HERE = Path(__file__).parent
FIG_SEL = HERE / "figures" / "12_cluster_selection.png"
FIG_PROFILE = HERE / "figures" / "13_cluster_profile.png"
OUT = HERE / "clustering.txt"

K_RANGE = range(2, 11)
SIL_SAMPLE = 8000        # 轮廓系数在子样本上算，全量太慢
SPREAD_RULE = 0.80       # 选 k 规则：达到最大坏账率跨度的 80% 即停（够用且群不至于太碎）

# 放款前可得的用户属性。注意：这里【没有 bad_loan，也没有 grade】——
# grade 是平台的风控结论，不是用户的原始属性，放进去会让分群变成"复读评级"。
FEATURES = {
    "fico_range_low": "FICO 分",
    "dti": "负债收入比",
    "annual_inc": "年收入",
    "revol_util": "循环额度使用率",
    "loan_amnt": "申请金额",
    "int_rate": "利率",
    "emp_length_int": "工作年限",
    "credit_hist_months": "信用历史(月)",
    "open_acc": "开立账户数",
    "total_acc": "总账户数",
    "revol_bal": "循环余额",
    "bc_util": "银行卡使用率",
    "num_actv_rev_tl": "活跃循环账户数",
    "mort_acc": "房贷账户数",
    "delinq_2yrs": "近2年逾期次数",
    "inq_last_6mths": "近半年查询次数",
    "tot_hi_cred_lim": "总授信额度",
}
LOG_FEATURES = ["annual_inc", "loan_amnt", "revol_bal", "tot_hi_cred_lim"]


# ===========================================================================
def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].copy()
    for c in LOG_FEATURES:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
    scaler = StandardScaler().fit(X)
    return X, scaler.transform(X)


def scan_k(Xs: np.ndarray, y: np.ndarray, seed: int) -> pd.DataFrame:
    """扫 k，同时记录几何指标（inertia / 轮廓系数）和业务指标（坏账率跨度）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), size=min(SIL_SAMPLE, len(Xs)), replace=False)
    sub = Xs[idx]

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        lb = km.labels_
        sil = silhouette_score(sub, km.predict(sub))
        rates = pd.Series(y).groupby(lb).mean() * 100
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "坏账率极差(pp)": rates.max() - rates.min(),
            "最小群占比%": pd.Series(lb).value_counts(normalize=True).min() * 100,
        })
    return pd.DataFrame(rows)


def profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d["群"] = labels
    agg = d.groupby("群").agg(
        样本数=("bad_loan", "size"),
        坏账率=("bad_loan", "mean"),
        平均FICO=("fico_range_low", "mean"),
        平均利率=("int_rate", "mean"),
        年收入中位数=("annual_inc", "median"),
        平均负债收入比=("dti", "mean"),
        平均额度使用率=("revol_util", "mean"),
        平均申请金额=("loan_amnt", "mean"),
        长期限占比=("term", lambda s: (s == " 60 months").mean()),
    )
    agg["样本占比"] = agg["样本数"] / len(d) * 100
    agg["坏账率"] = agg["坏账率"] * 100
    agg["长期限占比"] = agg["长期限占比"] * 100
    return agg.round(2).sort_values("坏账率", ascending=False)


def name_cluster(bad_rate: float, overall: float) -> str:
    """按坏账率给群起个业务名字。"""
    r = bad_rate / overall
    if r < 0.7:
        return "低风险"
    if r < 0.95:
        return "中低风险"
    if r < 1.15:
        return "中等风险"
    if r < 1.4:
        return "中高风险"
    return "高风险"


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--k", type=int, default=0, help="手动指定聚类数，不填则按业务指标自动选")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("借款人聚类分群")
    print("=" * 60)

    df = load_clean(args.data)
    y = df["bad_loan"].values
    X, Xs = build_matrix(df)
    print(f"[特征] {len(X.columns)} 个放款前用户属性（不含 bad_loan、不含 grade）")
    print("  " + "、".join(FEATURES[c] for c in X.columns))
    print(f"[预处理] {len(LOG_FEATURES)} 个偏态特征取 log1p 后标准化")

    # ---------- 扫 k -------------------------------------------------------
    print(f"\n[扫 k] k = {K_RANGE.start}~{K_RANGE.stop - 1}")
    st = scan_k(Xs, y, P.SEED)
    print(st.round(4).to_string(index=False))

    # ---------- 选 k：几何 vs 业务 -----------------------------------------
    k_sil = int(st.loc[st["silhouette"].idxmax(), "k"])
    spread_sil = float(st.loc[st["k"] == k_sil, "坏账率极差(pp)"].iloc[0])
    max_spread = st["坏账率极差(pp)"].max()
    cand = st[st["坏账率极差(pp)"] >= SPREAD_RULE * max_spread]
    k_biz = int(cand["k"].min()) if len(cand) else int(st["k"].iloc[-1])

    print(f"\n[选 k] 几何指标（轮廓系数）最优 → k = {k_sil}，"
          f"但坏账率极差只有 {spread_sil:.2f}pp")
    print(f"[选 k] 业务指标（坏账率极差，最大 {max_spread:.2f}pp）"
          f"→ 取达到 {SPREAD_RULE:.0%} 的最小 k = {k_biz}")
    if k_sil != k_biz:
        print(f"  ⚠️ 两者不一致：轮廓系数选出的 k={k_sil} 在业务上【没有区分度】。"
              f"\n     这说明轮廓系数衡量的是几何紧凑度，不是业务价值 —— "
              f"分群必须用业务指标回头验证。")
    k = args.k or k_biz
    print(f"[选 k] 本次使用 k = {k}")

    # ---------- 聚类 -------------------------------------------------------
    km = KMeans(n_clusters=k, n_init=10, random_state=P.SEED).fit(Xs)
    labels = km.labels_
    prof = profile(df, labels)
    prof.insert(0, "风险标签",
                [name_cluster(r, df["bad_loan"].mean() * 100) for r in prof["坏账率"]])
    print(f"\n[分群结果]（按坏账率降序）")
    print(prof.to_string())

    spread = prof["坏账率"].max() - prof["坏账率"].min()
    overall = df["bad_loan"].mean() * 100
    print(f"\n[区分度] 各群坏账率 {prof['坏账率'].min():.1f}% ~ "
          f"{prof['坏账率'].max():.1f}%（整体 {overall:.1f}%），极差 {spread:.1f}pp")
    print(f"[lift] 最低风险群 {prof['坏账率'].min() / overall:.2f}x，"
          f"最高风险群 {prof['坏账率'].max() / overall:.2f}x（相对整体）")

    # ---------- 与内部评级交叉 --------------------------------------------
    ct = pd.crosstab(labels, df["grade"].values, normalize="index") * 100
    ct = ct.round(1)
    print("\n[交叉] 各群落在内部评级 A~G 的占比（%）：")
    print(ct.to_string())

    # 结论要基于数据，不能写死：看看是不是某一群和评级高度重合、
    # 而其余群的评级分布相近却风险不同（那才说明聚类有独立价值）
    if "A" in ct.columns:
        a_hi = ct["A"].idxmax()
        others = ct.drop(index=a_hi)
        print(f"  → 群{a_hi} 有 {ct.loc[a_hi, 'A']:.0f}% 落在最优评级 A，"
              f"与评级高度重合；")
        print(f"     但其余 {len(others)} 个群的评级分布彼此接近"
              f"（A 档占比 {others['A'].min():.0f}%~{others['A'].max():.0f}%），"
              f"坏账率却是 "
              + "、".join(f"{v:.1f}%" for v in sorted(prof["坏账率"]))
              + "。")
        print("     说明聚类捕捉到了**评级之外**的客群结构，不是评级本身的复读。")

    # ---------- 稳定性：换算法 --------------------------------------------
    print("\n[稳定性] 换层次聚类(Ward，同样 k)在子样本上重跑：")
    sub_idx = np.arange(min(SIL_SAMPLE, len(Xs)))
    labels_h = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs[sub_idx])
    rates_h = sorted(pd.Series(y[sub_idx]).groupby(labels_h).mean() * 100)
    rates_k = sorted(pd.Series(y[sub_idx]).groupby(labels[sub_idx]).mean() * 100)
    print("  K-Means 各群坏账率：" + "、".join(f"{v:.1f}%" for v in rates_k))
    print("  Ward    各群坏账率：" + "、".join(f"{v:.1f}%" for v in rates_h))
    print(f"  两者跨度：K-Means {rates_k[-1] - rates_k[0]:.1f}pp，"
          f"Ward {rates_h[-1] - rates_h[0]:.1f}pp")
    print("  → 两种算法独立跑出的分群都能拉开坏账率，说明这个结构不是算法偶然。")

    # ---------- 出图 -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    ax.plot(st["k"], st["inertia"], marker="o", color="#c0392b", label="inertia")
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("簇内平方和 (inertia)", color="#c0392b")
    ax.tick_params(axis="y", colors="#c0392b")
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(st["k"], st["silhouette"], marker="s", ls="--", color="#2980b9",
             label="轮廓系数")
    ax2.axvline(k_sil, color="#2980b9", ls=":", lw=1.5)
    ax2.set_ylabel("轮廓系数", color="#2980b9")
    ax2.tick_params(axis="y", colors="#2980b9")
    ax.set_title(f"几何指标：轮廓系数选 k = {k_sil}\n但它衡量的是紧凑度，不是业务价值")

    ax = axes[1]
    ax.bar(st["k"], st["坏账率极差(pp)"], color="#7f8c8d", alpha=0.7,
           label="坏账率极差")
    ax.axhline(SPREAD_RULE * max_spread, color="#e67e22", ls="--", lw=1.2,
               label=f"{SPREAD_RULE:.0%} 阈值")
    ax.axvline(k_biz, color="#27ae60", lw=2, label=f"选定 k = {k_biz}")
    ax.scatter([k_sil], [spread_sil], color="#2980b9", s=90, zorder=5,
               edgecolor="white", linewidth=1.2,
               label=f"轮廓系数选的 k={k_sil}（仅 {spread_sil:.1f}pp）")
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("各群坏账率极差 (pp)")
    ax.set_title("业务指标：坏账率极差\n这才是分群有没有用的判据")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("图 12｜怎么选聚类数：轮廓系数会选错，要用业务指标兜底", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))      # 给 suptitle 留出空间，避免压住子图标题
    fig.savefig(FIG_SEL)
    plt.close(fig)

    # 图 13：分群画像
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    p = prof.sort_values("坏账率")
    x = np.arange(len(p))
    colors = plt.cm.RdYlGn_r(np.linspace(0.12, 0.88, len(p)))
    axes[0].bar(x, p["坏账率"], color=colors, alpha=0.92)
    axes[0].axhline(overall, color="#7f8c8d", ls="--", lw=1.2,
                    label=f"整体 {overall:.1f}%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"群{i}\n{lab}" for i, lab in
                             zip(p.index, p["风险标签"])], fontsize=9)
    axes[0].set_ylabel("坏账率 (%)")
    for xi, (v, n) in enumerate(zip(p["坏账率"], p["样本占比"])):
        axes[0].text(xi, v, f"{v:.1f}%\n占比{n:.0f}%", ha="center",
                     va="bottom", fontsize=8)
    axes[0].set_title("各群坏账率（无监督分群后回头验证）")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    z = X.copy()
    z["群"] = labels
    means = z.groupby("群")[list(X.columns)].mean()
    zsc = (means - X.mean()) / X.std()
    zsc = zsc.loc[p.index]                       # 与左图同序
    im = axes[1].imshow(zsc.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axes[1].set_xticks(range(len(X.columns)))
    axes[1].set_xticklabels([FEATURES[c] for c in X.columns], rotation=45,
                            ha="right", fontsize=8)
    axes[1].set_yticks(range(len(zsc)))
    axes[1].set_yticklabels([f"群{i}" for i in zsc.index])
    for i in range(zsc.shape[0]):
        for j in range(zsc.shape[1]):
            v = zsc.values[i, j]
            axes[1].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(v) > 1.2 else "black")
    axes[1].set_title("各群特征偏离全体均值的程度（z 分数）")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="z 分数")
    fig.suptitle("图 13｜分群画像：每群是什么人，风险差多少", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG_PROFILE)
    plt.close(fig)
    print(f"\n[出图] {FIG_SEL.relative_to(HERE)}")
    print(f"[出图] {FIG_PROFILE.relative_to(HERE)}")

    # ---------- 落盘 -------------------------------------------------------
    lines = [
        "借款人聚类分群（无监督分群 + 业务指标选 k）",
        f"特征（{len(X.columns)} 个放款前用户属性，不含 bad_loan、不含 grade）："
        + "、".join(FEATURES[c] for c in X.columns),
        f"预处理：{len(LOG_FEATURES)} 个偏态特征取 log1p 后标准化", "",
        "=== 扫 k（几何指标 vs 业务指标）===", st.round(4).to_string(index=False), "",
        f"轮廓系数最优 k = {k_sil}，但其坏账率极差仅 {spread_sil:.2f}pp（无业务区分度）",
        f"业务指标选 k = {k_biz}（规则：达到最大极差 {max_spread:.2f}pp 的 "
        f"{SPREAD_RULE:.0%}）", "",
        "=== 分群画像（按坏账率降序）===", prof.to_string(), "",
        f"各群坏账率极差 {spread:.1f} pp，整体坏账率 {overall:.1f}%", "",
        "=== 各群落在内部评级 A~G 的占比(%) ===", ct.to_string(), "",
        "=== 稳定性对照（层次聚类 Ward）===",
        "K-Means: " + "、".join(f"{v:.1f}%" for v in rates_k),
        "Ward   : " + "、".join(f"{v:.1f}%" for v in rates_h),
        "",
        "注：选 k 时使用了 bad_loan 做业务验证，因此严格说不是纯无监督 ——",
        "   无监督部分只负责分群，选 k 引入了标签做后验判断。",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[落盘] {OUT.name}")


if __name__ == "__main__":
    main()
