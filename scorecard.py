# -*- coding: utf-8 -*-
"""
WOE + IV 评分卡
===============

银行风控的标准交付物。逻辑回归 + 分箱 + WOE 变换，输出一张可以直接交给
业务方和监管的评分卡 —— 每一分怎么来的，都能逐项拆开。

为什么还要做这个（GBDT 的 AUC 明明更高）
------------------------------------------------------------------------
这是面试里一定会被问的「可解释性 vs 性能」权衡：

  GBDT  ：AUC 更高，但 300 棵树你没法跟监管解释"为什么拒了这个人"
  评分卡：AUC 略低，但每一分的来源清清楚楚，且天然单调、可人工复核

而且评分卡有个 GBDT 没有的性质：**分数单调对应坏账率**，
业务方可以直接说"650 分以上批、600 分以下拒"，还能拿去定价。

关键指标
------------------------------------------------------------------------
  WOE (Weight of Evidence) = ln( 该箱好客户占比 / 该箱坏客户占比 )
      > 0 说明这一箱里好客户相对多 → 低风险
  IV  (Information Value) = Σ(好客户占比 − 坏客户占比) × WOE
      经验判断：<0.02 无用，0.02~0.1 弱，0.1~0.3 中，0.3~0.5 强，>0.5 可疑

⚠️ 两条纪律（做错就全废）
  1. **分箱边界和 WOE 只在训练集上算**，再套用到测试集 —— 否则就是标签泄漏
  2. WOE 里的好/坏占比要加平滑（本脚本用 +0.5），否则某个箱子坏账为 0 时 WOE = ∞

用法：
    "D:/venvs/lending-club/Scripts/python.exe" scorecard.py
    "D:/venvs/lending-club/Scripts/python.exe" scorecard.py --split temporal
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
from sklearn.linear_model import LogisticRegression

import pipeline as P
from cost_threshold import fit_and_score, load_clean, split_data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

HERE = Path(__file__).parent
FIG_IV = HERE / "figures" / "10_scorecard_iv.png"
FIG_SCORE = HERE / "figures" / "11_scorecard_score.png"
OUT = HERE / "scorecard.txt"

N_BINS = 5              # 数值特征分几箱（箱越多越容易过拟合，5 箱是风控常规）
IV_MIN = 0.02           # IV 低于此值的特征直接丢
MAX_FEATURES = 12       # 评分卡最多放几个变量（太多业务方维护不动）

# 评分刻度（业界通行参数）：基准分 600、基准好坏比 20:1、每 50 分好坏比翻倍
BASE_SCORE = 600
BASE_ODDS_BAD = 1 / 20.0    # P(bad)/P(good) = 1:20，即基准分对应 4.8% 坏账率
PDO = 50


# ===========================================================================
# 分箱：边界只在训练集上确定
# ===========================================================================
MAX_CAT_BINS = 12       # 类别列超过这个数量就做粗分类


def make_binner(x_tr: pd.Series, is_num: bool, y_tr: pd.Series | None = None):
    """在【训练集】上确定分箱规则。返回 (类型, 规格)。"""
    if is_num:
        try:
            binned = pd.qcut(x_tr, q=N_BINS, duplicates="drop")
        except (ValueError, IndexError):
            binned = pd.cut(x_tr, bins=max(2, min(N_BINS, x_tr.nunique())))
        return ("num", binned.categories)

    vc = x_tr.value_counts()
    thr = max(50, int(0.01 * len(x_tr)))
    candidates = list(vc[vc >= thr].index)

    # 高基数类别（如 sub_grade 有 35 档）不能简单地"留前 12 个、其余塞 OTHER"——
    # 那样 OTHER 会吃掉三分之一样本，等于把信息扔了。
    # 风控标准做法是 coarse classing：按【坏账率】把相邻档合并成若干组。
    # 注意这里只用了训练集的 y，是安全的。
    if y_tr is not None and len(candidates) > MAX_CAT_BINS:
        g = pd.DataFrame({"x": x_tr.values, "y": np.asarray(y_tr)})
        st = g[g["x"].isin(candidates)].groupby("x")["y"].agg(["mean", "count"])
        st = st.sort_values("mean")                    # 按坏账率从低到高排
        # 底部不足 min_per_bin 的档位直接归入 OTHER，避免 WOE 抖动
        min_per_bin = max(30, int(0.005 * len(x_tr)))
        keep = st[st["count"] >= min_per_bin]
        n_groups = min(MAX_CAT_BINS // 2, len(keep))
        if n_groups >= 2:
            mapping = {}
            for gi, idxs in enumerate(np.array_split(np.arange(len(keep)), n_groups)):
                for cat in keep.index[idxs]:
                    mapping[cat] = f"第{gi + 1}组"
            return ("catmap", mapping)

    if not candidates:                # 极端情况：全是稀有类别
        candidates = list(vc.index[:1])
    return ("cat", candidates[:MAX_CAT_BINS])


def apply_binner(x: pd.Series, binner) -> pd.Series:
    kind, spec = binner
    if kind == "num":
        return pd.cut(x, bins=spec)
    if kind == "catmap":
        return x.map(spec).fillna("OTHER")
    return x.where(x.isin(spec), other="OTHER")


def fit_woe(x_tr: pd.Series, y_tr: pd.Series, binner) -> tuple[dict, float, pd.DataFrame]:
    """在训练集上算每个箱的 WOE。返回 (woe映射, IV, 明细表)。"""
    b = apply_binner(x_tr, binner)
    g = pd.DataFrame({"bin": b.astype(object), "y": np.asarray(y_tr)})
    st = g.groupby("bin", observed=False)["y"].agg(["count", "sum"])
    st.columns = ["n", "bad"]
    st["good"] = st["n"] - st["bad"]

    total_bad, total_good = st["bad"].sum(), st["good"].sum()
    k = len(st)
    # +0.5 平滑：某个箱子坏账为 0 时，log(0) 会炸
    st["dist_bad"] = (st["bad"] + 0.5) / (total_bad + 0.5 * k)
    st["dist_good"] = (st["good"] + 0.5) / (total_good + 0.5 * k)
    st["woe"] = np.log(st["dist_good"] / st["dist_bad"])
    st["iv"] = (st["dist_good"] - st["dist_bad"]) * st["woe"]
    st["bad_rate"] = st["bad"] / st["n"]

    return st["woe"].to_dict(), float(st["iv"].sum()), st


def to_woe(x: pd.Series, binner, woe_map: dict) -> pd.Series:
    b = apply_binner(x, binner).astype(object)
    # 测试集里出现训练集没见过的新类别 → 落在 OTHER，映射不到就给 0（中性）
    return b.map(woe_map).astype(float).fillna(0.0)


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample", type=int, default=P.SAMPLE_N)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--cutoff", type=int, default=2015)
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("WOE + IV 评分卡")
    print("=" * 60)

    df = load_clean(args.data)
    tr_idx, te_idx = split_data(df, args.split, args.cutoff)
    print(f"[划分] {args.split}：训练 {len(tr_idx):,} / 测试 {len(te_idx):,}")

    X = df.drop(columns=["bad_loan"]).reset_index(drop=True)
    y = df["bad_loan"].reset_index(drop=True)
    if args.split == "temporal" and "issue_year" in X.columns:
        X = X.drop(columns=["issue_year"])

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    y_te = y.iloc[te_idx].reset_index(drop=True)

    # ---------- 1) 逐特征算 IV，只用训练集 --------------------------------
    print(f"\n[分箱] 每个特征分 {N_BINS} 箱，边界与 WOE 只在训练集上确定")
    audit = []
    for col in X.columns:
        is_num = pd.api.types.is_numeric_dtype(X_tr[col])
        if is_num and X_tr[col].nunique() < 3:
            continue                                   # 近似常量列，没有区分度
        try:
            binner = make_binner(X_tr[col], is_num, y_tr)
            _, iv, _ = fit_woe(X_tr[col], y_tr, binner)
            audit.append({"特征": col, "类型": "数值" if is_num else "类别",
                          "箱数": len(binner[1]), "IV": iv})
        except Exception as e:
            print(f"  [跳过] {col}: {type(e).__name__}: {e}")

    audit_df = pd.DataFrame(audit).sort_values("IV", ascending=False)
    print("\n[IV 排序] 前 15 个特征：")
    print(audit_df.head(15).round(4).to_string(index=False))

    def strength(iv):
        if iv < 0.02:
            return "无用"
        if iv < 0.1:
            return "弱"
        if iv < 0.3:
            return "中"
        if iv < 0.5:
            return "强"
        return "过强(可疑)"

    audit_df["强度"] = audit_df["IV"].map(strength)

    # ---------- 2) 选特征 + WOE 变换 --------------------------------------
    chosen = audit_df[audit_df["IV"] >= IV_MIN]["特征"].tolist()[:MAX_FEATURES]
    print(f"\n[选特征] IV >= {IV_MIN} 且取前 {MAX_FEATURES} 个，共 {len(chosen)} 个：")
    print("  " + ", ".join(f"{c}({audit_df.set_index('特征').loc[c, 'IV']:.3f})"
                           for c in chosen))

    binners, woe_maps, woe_tables = {}, {}, {}
    for col in chosen:
        is_num = pd.api.types.is_numeric_dtype(X_tr[col])
        binners[col] = make_binner(X_tr[col], is_num, y_tr)
        woe_maps[col], _, woe_tables[col] = fit_woe(X_tr[col], y_tr, binners[col])

    W_tr = pd.DataFrame({c: to_woe(X_tr[c], binners[c], woe_maps[c]) for c in chosen})
    W_te = pd.DataFrame({c: to_woe(X_te[c], binners[c], woe_maps[c]) for c in chosen})

    # ---------- 2.5) 去冗余 -------------------------------------------------
    # grade 和 sub_grade 是父子关系（sub_grade 是 grade 的细分），WOE 几乎线性相关。
    # 两个都留在评分卡里 = 同一个信息算两遍，系数会互相打架、不稳定。
    # 标准做法：相关性超过阈值的，只留 IV 更高的那个。
    iv_of = audit_df.set_index("特征")["IV"]
    dropped = []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            if a in dropped or b in dropped:
                continue
            r = abs(np.corrcoef(W_tr[a].values, W_tr[b].values)[0, 1])
            if r > 0.85:
                weaker = a if iv_of[a] < iv_of[b] else b
                dropped.append(weaker)
    if dropped:
        print(f"\n[去冗余] 以下变量与保留变量相关性 > 0.85，已剔除：{dropped}")
        chosen = [c for c in chosen if c not in dropped]
        W_tr, W_te = W_tr[chosen], W_te[chosen]
        print(f"[去冗余] 最终进入评分卡：{chosen}")

    # ---------- 3) 逻辑回归 -------------------------------------------------
    lr = LogisticRegression(max_iter=2000, random_state=P.SEED).fit(W_tr, y_tr)
    p_tr = lr.predict_proba(W_tr)[:, 1]
    p_te = lr.predict_proba(W_te)[:, 1]

    print("\n[逻辑回归系数]（应为负：WOE 越大=风险越低）")
    coef = pd.DataFrame({"特征": chosen, "系数": lr.coef_[0].round(4),
                         "IV": [audit_df.set_index('特征').loc[c, 'IV'] for c in chosen]})
    print(coef.round(4).to_string(index=False))
    bad_sign = coef[coef["系数"] > 0]
    if len(bad_sign):
        print(f"  ⚠️ 有 {len(bad_sign)} 个特征系数为正：{list(bad_sign['特征'])} —— "
              "说明该特征的分箱不是单调的，上评分卡前要人工合并相邻箱")

    # ---------- 4) 评分卡落表 ----------------------------------------------
    factor = PDO / np.log(2)
    offset = BASE_SCORE + factor * np.log(BASE_ODDS_BAD)
    base_points = offset - factor * lr.intercept_[0]
    rows = [{"特征": "__基准分__", "分箱": "-", "样本数": len(y_tr),
             "坏账率%": round(100 * y_tr.mean(), 2), "WOE": np.nan,
             "得分": round(base_points, 1)}]
    for c, b in zip(chosen, lr.coef_[0]):
        t = woe_tables[c]
        for bin_label, r in t.iterrows():
            rows.append({
                "特征": c,
                "分箱": str(bin_label),
                "样本数": int(r["n"]),
                "坏账率%": round(100 * r["bad_rate"], 2),
                "WOE": round(r["woe"], 4),
                "得分": round(-factor * b * r["woe"], 1),
            })
    card = pd.DataFrame(rows)
    print(f"\n[评分卡] 基准分 {BASE_SCORE}，基准好坏比 1:{int(1/BASE_ODDS_BAD)}，"
          f"PDO={PDO}")
    print(card.head(25).to_string(index=False))
    print(f"  ……（共 {len(card)} 行，完整表见 {OUT.name}）")

    # ---------- 5) 分数 → 坏账率（单调性验证）-----------------------------
    sel = df.iloc[te_idx][["loan_amnt", "int_rate", "term"]].reset_index(drop=True)
    _ = sel  # 保留：如需按金额加权可在这里扩展
    score = base_points + sum(-factor * lr.coef_[0][i] * W_te[chosen[i]]
                              for i in range(len(chosen)))
    buckets = pd.qcut(score, q=10, duplicates="drop")
    band = pd.DataFrame({"score": score, "bad": y_te.values, "band": buckets})
    band_stat = band.groupby("band", observed=False)["bad"].agg(["count", "mean"])
    band_stat.columns = ["样本数", "坏账率"]
    band_stat["坏账率%"] = (band_stat["坏账率"] * 100).round(2)
    band_stat = band_stat[["样本数", "坏账率%"]].reset_index()

    # ---------- 6) 与 GBDT 对比 --------------------------------------------
    print("\n[对比] 同一划分下，评分卡 vs 梯度提升：")
    _, gb_probs, _, _ = fit_and_score(df, tr_idx, te_idx, args.split, "none")
    cmp_df = pd.DataFrame({
        "评分卡(WOE+LR)": P.evaluate(y_te, p_te),
        "梯度提升(GBDT)": P.evaluate(y_te, gb_probs),
    }).T[["AUC", "PR-AUC", "KS", "Acc", "R@P0.5"]]
    print(cmp_df.round(4).to_string())
    gap = cmp_df.loc["梯度提升(GBDT)", "AUC"] - cmp_df.loc["评分卡(WOE+LR)", "AUC"]
    print(f"  → GBDT 的 AUC 高出 {gap:.4f}，这是用【可解释性】换来的性能差；")
    print("    评分卡每一项得分都能拆开给业务方看，GBDT 做不到。")

    # ---------- 7) 出图 -----------------------------------------------------
    top = audit_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#c0392b" if v >= 0.3 else "#2980b9" if v >= 0.1
              else "#7f8c8d" if v >= 0.02 else "#bdc3c7" for v in top["IV"]]
    ax.barh(range(len(top)), top["IV"], color=colors, alpha=0.9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["特征"], fontsize=9)
    for i, v in enumerate(top["IV"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.axvline(0.02, color="#7f8c8d", ls="--", lw=1, alpha=0.7)
    ax.axvline(0.1, color="#2980b9", ls="--", lw=1, alpha=0.7)
    ax.axvline(0.3, color="#c0392b", ls="--", lw=1, alpha=0.7)
    ax.text(0.021, len(top) - 0.5, " 0.02 弱", fontsize=8, color="#7f8c8d")
    ax.text(0.101, len(top) - 0.5, " 0.1 中", fontsize=8, color="#2980b9")
    ax.text(0.301, len(top) - 0.5, " 0.3 强", fontsize=8, color="#c0392b")
    ax.set_xlabel("IV（信息值）")
    ax.set_title("图 10｜特征 IV 排序\nIV 越高区分能力越强；>0.5 通常意味着泄漏，要警惕",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    FIG_IV.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_IV)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(band_stat))
    ax.bar(x, band_stat["坏账率%"], color="#c0392b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in x])
    ax.set_xlabel("评分十分位（1 = 分数最低 = 风险最高）")
    ax.set_ylabel("坏账率 (%)")
    for i, (r, n) in enumerate(zip(band_stat["坏账率%"], band_stat["样本数"])):
        ax.text(i, r, f"{r:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.axhline(100 * y_te.mean(), color="#7f8c8d", ls="--", lw=1.2,
               label=f"整体坏账率 {100*y_te.mean():.1f}%")
    ax.set_title("图 11｜评分卡的分数必须单调对应坏账率\n"
                 "不单调就说明分箱有问题，不能上生产", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_SCORE)
    plt.close(fig)
    print(f"\n[出图] {FIG_IV.relative_to(HERE)}")
    print(f"[出图] {FIG_SCORE.relative_to(HERE)}")

    # ---------- 8) 落盘 -----------------------------------------------------
    lines = [f"WOE + IV 评分卡（{args.split} 划分）",
             f"训练 {len(tr_idx):,} / 测试 {len(te_idx):,}，"
             f"测试集坏账率 {y_te.mean():.2%}", "",
             "=== 特征 IV 排序 ===", audit_df.round(4).to_string(index=False), "",
             "=== 逻辑回归系数 ===", coef.to_string(index=False), "",
             f"=== 评分卡（基准分 {BASE_SCORE}，PDO={PDO}）===",
             card.to_string(index=False), "",
             "=== 测试集评分十分位坏账率 ===",
             band_stat.to_string(index=False), "",
             "=== 评分卡 vs GBDT ===", cmp_df.round(4).to_string()]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[落盘] {OUT.name}")


if __name__ == "__main__":
    main()
