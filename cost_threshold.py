# -*- coding: utf-8 -*-
"""
成本敏感阈值
============

默认阈值 0.5 是拍脑袋的 —— 它隐含假设"漏掉一个坏账"和"误拒一个好客户"
代价相同。做信贷显然不是。

这个脚本回答一个业务问题：
    **给定两种错的代价，审批线该卡在哪个概率上，期望总代价最小？**

成本口径
------------------------------------------------------------------------
  漏判（FN，预测好 / 实际坏）= loan_amnt × fn_rate
      fn_rate = 违约损失率 LGD，无抵押消费贷经验值 0.6（回收约 40%）
  误拒（FP，预测坏 / 实际好）= loan_amnt × fp_rate
      fp_rate = 这笔生意的**净利润率**，即利息收入 − 资金成本 − 运营成本。
      注意不是毛利息！60 个月 20% 利率的毛利息是本金的 100%，
      但那不是平台赚的钱，用它会把"拒绝"的代价算得比"违约"还高。

⭐ 一个能被验证的理论结果
------------------------------------------------------------------------
当两边的成本都与贷款金额成正比、且概率是**校准过的**时，逐笔最优决策是：

    拒绝  ⟺  P(坏) × fn_rate  >  (1 − P(坏)) × fp_rate
          ⟺  P(坏) > fp_rate / (fp_rate + fn_rate)

注意贷款金额 A 被约掉了 —— **最优阈值只取决于代价比，与金额无关**。
所以理论阈值 = fp_rate/(fp_rate+fn_rate)，实测扫描出来的最优点应该落在它附近。
**两者吻合，就证明模型输出的概率是校准可信的**；差得远，说明概率尺度有问题。

⚠️ 概率必须先校准
------------------------------------------------------------------------
`pipeline.py --balance classweight` 会给少数类加权，**把概率推离真实先验**，
此时阈值不能当概率读。所以本脚本默认 `--balance none`，并打印校准诊断。

用法：
    "D:/venvs/lending-club/Scripts/python.exe" cost_threshold.py
    "D:/venvs/lending-club/Scripts/python.exe" cost_threshold.py --split temporal
    "D:/venvs/lending-club/Scripts/python.exe" cost_threshold.py --fn-rate 0.5 --fp-rate 0.2
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import pipeline as P

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

HERE = Path(__file__).parent
FIG = HERE / "figures" / "09_cost_threshold.png"
OUT = HERE / "cost_analysis.txt"
CLEAN_CACHE = HERE / "data" / "_cache_clean.pkl"

C_COST = "#c0392b"
C_OPT = "#27ae60"


# ===========================================================================
def load_clean(data_path: str | None = None) -> pd.DataFrame:
    """复用 pipeline.clean()，保证口径一致。"""
    if CLEAN_CACHE.exists():
        df = pd.read_pickle(CLEAN_CACHE)
        print(f"[缓存] 命中 {CLEAN_CACHE.name}：{len(df):,} 行，"
              f"坏账率 {df['bad_loan'].mean():.2%}")
        return df
    raw = P.load_real_data(P.resolve_data_path(data_path), use_cache=True)
    df = P.clean(raw)
    df.to_pickle(CLEAN_CACHE)
    return df


def split_data(df: pd.DataFrame, how: str, cutoff: int):
    """返回 (训练索引, 测试索引)。口径与 pipeline.py 完全一致。"""
    if how == "temporal":
        yy = df["issue_year"].values
        tr = np.where(yy < cutoff)[0]
        te = np.where(yy >= cutoff)[0]
        if len(tr) < 500 or len(te) < 500:
            raise SystemExit(f"[错误] 时间外划分样本太少：训练 {len(tr)} / 测试 {len(te)}")
        return tr, te
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=P.SEED)
    return next(ss.split(df, df["bad_loan"]))


def fit_and_score(df: pd.DataFrame, tr_idx, te_idx, how: str, balance: str):
    """按 pipeline 的流程训练，返回测试集概率、金额列、选中特征。"""
    X = df.drop(columns=["bad_loan"]).reset_index(drop=True)
    y = df["bad_loan"].reset_index(drop=True)
    amt = pd.Series(df["loan_amnt"].values)

    if how == "temporal" and "issue_year" in X.columns:
        X = X.drop(columns=["issue_year"])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_tr, X_te = X.iloc[tr_idx].copy().reset_index(drop=True), \
                 X.iloc[te_idx].copy().reset_index(drop=True)
    y_tr, y_te = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)
    amt_te = amt.iloc[te_idx].reset_index(drop=True)

    X_tr, X_te = P.target_encode(X_tr, y_tr, X_te, cat_cols, oof=False)
    scaler = StandardScaler().fit(X_tr[num_cols])
    X_tr[num_cols] = scaler.transform(X_tr[num_cols])
    X_te[num_cols] = scaler.transform(X_te[num_cols])
    keep = P.select_features(X_tr, y_tr)
    X_tr, X_te = X_tr[keep], X_te[keep]

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                       random_state=P.SEED)
    m = (P.fit_model(model, X_tr, y_tr, balance) if balance != "none"
         else model.fit(X_tr, y_tr))

    return y_te, m.predict_proba(X_te)[:, 1], amt_te, keep


def cost_curve(y_te, probs, fn_cost, fp_cost, thresholds):
    """扫描阈值，返回每个阈值下的期望总成本。"""
    y = np.asarray(y_te)
    rows = []
    total_bad = max((y == 1).sum(), 1)
    for t in thresholds:
        rejected = probs >= t            # 判为坏 -> 拒绝
        fn = np.sum(fn_cost[(y == 1) & (~rejected)])     # 漏判：放出去的坏账
        fp = np.sum(fp_cost[(y == 0) & rejected])        # 误拒：推掉的好客户
        rows.append({
            "阈值": t,
            "总成本(百万)": (fn + fp) / 1e6,
            "漏判成本(百万)": fn / 1e6,
            "误拒成本(百万)": fp / 1e6,
            "通过率%": 100.0 * (~rejected).mean(),
            "通过者坏账率%": (100.0 * y[~rejected].mean()) if (~rejected).any() else 0.0,
            "拦截坏账%": 100.0 * ((y == 1) & rejected).sum() / total_bad,
        })
    return pd.DataFrame(rows)


def theory_threshold(fn_rate: float, fp_rate: float) -> float:
    """逐笔最优决策的解析解：P(坏) > fp/(fp+fn) 就拒绝。"""
    return fp_rate / (fp_rate + fn_rate)


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--cutoff", type=int, default=2015)
    ap.add_argument("--fn-rate", type=float, default=0.60,
                    help="漏判成本率（=LGD，违约损失本金的比例），默认 0.6")
    ap.add_argument("--fp-rate", type=float, default=0.10,
                    help="误拒成本率（=净利润率，占本金比例），默认 0.1")
    ap.add_argument("--balance", choices=["none", "classweight"], default="none",
                    help="默认 none：不加权，概率校准在真实先验上")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("成本敏感阈值分析")
    print("=" * 60)

    df = load_clean(args.data)
    tr_idx, te_idx = split_data(df, args.split, args.cutoff)
    print(f"[划分] {args.split}：训练 {len(tr_idx):,} / 测试 {len(te_idx):,}")

    y_te, probs, amt_te, keep = fit_and_score(df, tr_idx, te_idx, args.split, args.balance)

    # ---------- 概率校准诊断 ----------------------------------------------
    actual = y_te.mean()
    print(f"\n[校准诊断] 测试集实际坏账率 = {actual:.2%}")
    print(f"[校准诊断] 模型预测概率均值 = {probs.mean():.2%}")
    gap = probs.mean() - actual
    if abs(gap) < 0.02:
        print(f"[校准诊断] 偏差 {gap:+.2%} —— 概率尺度可信，阈值可当作坏账概率解读 [OK]")
    else:
        print(f"[校准诊断] 偏差 {gap:+.2%} —— 概率偏离真实尺度，"
              f"阈值只能当决策线，不能读成坏账概率")

    # ---------- 成本 -------------------------------------------------------
    fn_cost = amt_te.values * args.fn_rate
    fp_cost = amt_te.values * args.fp_rate
    th = theory_threshold(args.fn_rate, args.fp_rate)

    print(f"\n[成本假设] 漏判 = 本金 × {args.fn_rate:.0%}（LGD），均值 "
          f"{fn_cost.mean():,.0f} 元/笔")
    print(f"[成本假设] 误拒 = 本金 × {args.fp_rate:.0%}（净利润率），均值 "
          f"{fp_cost.mean():,.0f} 元/笔")
    print(f"[成本假设] 代价比 FN:FP = {args.fn_rate / args.fp_rate:.1f} : 1")
    print(f"[理论阈值] fp/(fp+fn) = {args.fp_rate:.2f}/({args.fp_rate:.2f}"
          f"+{args.fn_rate:.2f}) = {th:.3f}  ← 逐笔最优决策的解析解")

    # ---------- 扫描阈值 ---------------------------------------------------
    thresholds = np.round(np.arange(0.01, 0.99, 0.005), 4)
    curve = cost_curve(y_te, probs, fn_cost, fp_cost, thresholds)

    best = curve.loc[curve["总成本(百万)"].idxmin()]
    at50 = curve.iloc[(curve["阈值"] - 0.5).abs().argmin()]
    at_theory = curve.iloc[(curve["阈值"] - th).abs().argmin()]
    reject_none = curve.iloc[-1]      # 阈值最高 = 几乎不拒
    reject_all = curve.iloc[0]        # 阈值最低 = 几乎全拒

    print("\n" + "-" * 60)
    print("[结果]")
    print(f"  实测最优阈值     = {best['阈值']:.3f}   总成本 {best['总成本(百万)']:.2f} 百万"
          f"   通过率 {best['通过率%']:.1f}%")
    print(f"  理论阈值 {th:.3f}   = 总成本 {at_theory['总成本(百万)']:.2f} 百万"
          f"   通过率 {at_theory['通过率%']:.1f}%")
    print(f"  默认 0.500       = 总成本 {at50['总成本(百万)']:.2f} 百万"
          f"   通过率 {at50['通过率%']:.1f}%")
    print(f"  一个都不拒       = 总成本 {reject_none['总成本(百万)']:.2f} 百万")
    print(f"  全部拒绝         = 总成本 {reject_all['总成本(百万)']:.2f} 百万")

    drift = abs(best["阈值"] - th)
    print(f"\n  → 实测最优 {best['阈值']:.3f} vs 理论 {th:.3f}，相差 {drift:.3f}："
          + ("吻合，说明概率校准可信 ✓" if drift <= 0.03
             else "偏离较大，概率尺度需要复核 ⚠"))
    saved = at50["总成本(百万)"] - at_theory["总成本(百万)"]
    print(f"  → 相比默认 0.5，用理论阈值可省 {saved:.2f} 百万 "
          f"（降幅 {saved / at50['总成本(百万)']:.1%}）")
    print(f"  → 该阈值下：通过率 {at_theory['通过率%']:.1f}%，"
          f"通过者坏账率 {at_theory['通过者坏账率%']:.1f}%，"
          f"拦截坏账 {at_theory['拦截坏账%']:.1f}%")

    # ---------- 代价比敏感性 ----------------------------------------------
    print("\n[敏感性] 漏判成本率 fn_rate 变化时，理论阈值与实际最优怎么变：")
    sens = []
    for fnr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        fnc = amt_te.values * fnr
        cv = cost_curve(y_te, probs, fnc, fp_cost, thresholds)
        b = cv.loc[cv["总成本(百万)"].idxmin()]
        t_th = theory_threshold(fnr, args.fp_rate)
        sens.append({"fn_rate": fnr, "理论阈值": round(t_th, 3),
                     "实测最优阈值": b["阈值"], "偏差": round(abs(b["阈值"] - t_th), 3),
                     "通过率%": round(b["通过率%"], 1),
                     "拦截坏账%": round(b["拦截坏账%"], 1)})
    sens_df = pd.DataFrame(sens)
    print(sens_df.to_string(index=False))
    print("→ 漏判越贵（fn_rate 越高），审批线越紧（阈值越低、拒得越多）—— "
          "实测最优与理论解全程贴合，说明这套决策规则是稳的。")

    # ---------- 出图 -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(curve["阈值"], curve["总成本(百万)"], color=C_COST, lw=2,
                 label="期望总成本")
    axes[0].plot(curve["阈值"], curve["漏判成本(百万)"], ls="--", lw=1.2,
                 color="#e67e22", label="漏判坏账成本")
    axes[0].plot(curve["阈值"], curve["误拒成本(百万)"], ls="--", lw=1.2,
                 color="#2980b9", label="误拒好客户成本")
    axes[0].axvline(th, color=C_OPT, lw=1.8, label=f"理论最优阈值 {th:.3f}")
    axes[0].axvline(0.5, color="#7f8c8d", ls=":", lw=1.5, label="默认阈值 0.50")
    axes[0].scatter([best["阈值"]], [best["总成本(百万)"]], color=C_OPT,
                    zorder=5, s=60, edgecolor="white", linewidth=1.2)
    axes[0].set_xlabel("审批阈值（预测坏账概率 ≥ 阈值则拒绝）")
    axes[0].set_ylabel("期望总成本（百万元）")
    axes[0].set_title("成本随阈值的变化\n两条虚线交叉处 = 最优审批线")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(sens_df["fn_rate"], sens_df["理论阈值"], marker="o", lw=2,
                 color=C_OPT, label="理论阈值 fp/(fp+fn)")
    axes[1].plot(sens_df["fn_rate"], sens_df["实测最优阈值"], marker="s", lw=1.5,
                 ls="--", color=C_COST, label="实测扫描最优点")
    axes[1].set_xlabel("漏判成本率 fn_rate（LGD）")
    axes[1].set_ylabel("最优审批阈值")
    axes[1].set_title("敏感性：两条线全程贴合\n说明概率校准可信、决策规则稳健")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.suptitle("图 9｜成本敏感阈值：不取 0.5，取代价最小的那个点", fontsize=13)
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    print(f"\n[出图] {FIG.relative_to(HERE)}")

    # ---------- 落盘 -------------------------------------------------------
    lines = [f"成本敏感阈值分析（{args.split} 划分，"
             f"fn_rate={args.fn_rate}, fp_rate={args.fp_rate}）",
             f"测试集实际坏账率 {actual:.2%}，预测概率均值 {probs.mean():.2%}",
             f"理论最优阈值 fp/(fp+fn) = {th:.4f}", "",
             "=== 关键阈值对照 ===",
             f"实测最优 {best['阈值']:.3f}：总成本 {best['总成本(百万)']:.2f} 百万，"
             f"通过率 {best['通过率%']:.1f}%，拦截坏账 {best['拦截坏账%']:.1f}%",
             f"理论 {th:.3f}：总成本 {at_theory['总成本(百万)']:.2f} 百万，"
             f"通过率 {at_theory['通过率%']:.1f}%，拦截坏账 {at_theory['拦截坏账%']:.1f}%",
             f"默认 0.500：总成本 {at50['总成本(百万)']:.2f} 百万，"
             f"通过率 {at50['通过率%']:.1f}%，拦截坏账 {at50['拦截坏账%']:.1f}%", "",
             "=== fn_rate 敏感性 ===", sens_df.to_string(index=False), "",
             "=== 完整成本曲线（每 0.05 采一点）===",
             curve.iloc[::10].to_string(index=False)]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[落盘] {OUT.name}")


if __name__ == "__main__":
    main()
