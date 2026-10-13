# -*- coding: utf-8 -*-
"""
PSM 倾向得分匹配 —— 相关性 ≠ 因果
==================================

要回答的因果问题
------------------------------------------------------------------------
    **「选择 60 个月期限，是不是导致违约率更高的原因？」**

为什么不能直接比较两组坏账率？
------------------------------------------------------------------------
直接算的话，60 个月组的坏账率明显更高。但这**不能说明是期限导致的**，
因为"选长期限的人"和"选短期限的人"本来就不是一批人：

  · 月供压力大的人更倾向拉长期限降低月供 —— 而月供压力大本身就意味着财务紧张
  · 平台可能也倾向给某些人推长期限

也就是说，**期限选择**和**违约风险**背后有共同的原因（混淆变量 confounding）。
直接比较得到的是**相关**，不是**因果**。这是个典型的"选择性偏误"。

PSM 怎么解决
------------------------------------------------------------------------
   1. 用放款前可得的协变量 X，建一个逻辑回归估计
      **倾向得分 e(X) = P(选 60 个月 | X)** —— 即"这个人选长期限的可能性"
   2. 给每个选了 60 个月的人，找一个 e(X) 几乎相同、但选了 36 个月的人配对
      （卡钳最近邻匹配，1:1 无放回）
   3. 配对后，两组在这些协变量上就"可比"了 —— 相当于模拟了一次随机分配
   4. 在匹配样本上重新算坏账率差异，这个才是**期限的因果效应（ATT）**

⚠️ 必须先说清楚的局限（面试一定会问）
------------------------------------------------------------------------
PSM 只能控制**可观测**的混淆变量。如果存在**未观测**的混淆
（比如"申请人自己预判要提前还款所以选长期限"这种数据里根本没有的信息），
PSM 消除不了。所以 PSM 给出的是**观察性研究**的证据，强度不如随机实验(RCT)。
主动说出这一点，比装作 PSM 能证因果要加分得多。

用法：
    "D:/venvs/lending-club/Scripts/python.exe" psm.py
    "D:/venvs/lending-club/Scripts/python.exe" psm.py --caliper 0.1
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import pipeline as P
from cost_threshold import load_clean

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

HERE = Path(__file__).parent
FIG = HERE / "figures" / "15_psm.png"
OUT = HERE / "psm.txt"

TREATED_TERM = " 60 months"
CONTROL_TERM = " 36 months"
CALIPER_FRAC = 0.2        # 卡钳 = 0.2 × logit(倾向得分) 的标准差（文献常用值）
N_BOOT = 500

# 混淆变量：放款前可得、同时影响"选不选长期限"和"会不会违约"的因素
NUM_COVS = {
    "fico_range_low": "FICO 分",
    "dti": "负债收入比",
    "annual_inc": "年收入",
    "revol_util": "循环额度使用率",
    "loan_amnt": "申请金额",
    "emp_length_int": "工作年限",
    "credit_hist_months": "信用历史(月)",
    "open_acc": "开立账户数",
    "total_acc": "总账户数",
    "mort_acc": "房贷账户数",
    "delinq_2yrs": "近2年逾期次数",
    "inq_last_6mths": "近半年查询次数",
    "bc_util": "银行卡使用率",
    "tot_hi_cred_lim": "总授信额度",
}
# 类别混淆变量，做 one-hot
CAT_COVS = ["grade", "home_ownership", "verification_status", "purpose"]
LOG_FEATURES = ["annual_inc", "loan_amnt", "tot_hi_cred_lim"]

# ⚠️ 刻意【不放进】协变量的字段，以及为什么：
#   int_rate —— 利率本身由 grade 和 term 共同决定，放进协变量等于控制了中介，
#               会把要估计的效应吸收掉
#   installment —— 月供是 term 和金额的直接函数，同理
EXCLUDED = {"int_rate": "由 grade 和 term 共同决定，是中介变量",
            "installment": "月供是 term 和金额的直接函数"}


# ===========================================================================
def build_covariates(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in NUM_COVS if c in df.columns]
    C = df[num].copy()
    for c in LOG_FEATURES:
        if c in C.columns:
            C[c] = np.log1p(C[c].clip(lower=0))

    cats = [c for c in CAT_COVS if c in df.columns]
    if cats:
        dummies = pd.get_dummies(df[cats], prefix=cats, drop_first=True,
                                 dtype=float)
        C = pd.concat([C.reset_index(drop=True), dummies.reset_index(drop=True)],
                      axis=1)

    # one-hot 列不再标准化（已是 0/1）
    cont = [c for c in num]
    scaler = StandardScaler().fit(C[cont])
    C[cont] = scaler.transform(C[cont])
    return C


def smd(t, c) -> float:
    """标准化均值差。|SMD| < 0.1 通常认为平衡。"""
    t, c = np.asarray(t, dtype=float), np.asarray(c, dtype=float)
    denom = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    if denom == 0:
        return 0.0
    return (t.mean() - c.mean()) / denom


def match_without_replacement(ps_logit, T, caliper):
    """1:1 卡钳最近邻匹配，无放回（一个对照组只能配一次）。"""
    ctrl = np.where(T == 0)[0]
    trt = np.where(T == 1)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(ps_logit[ctrl].reshape(-1, 1))
    dist, ind = nn.kneighbors(ps_logit[trt].reshape(-1, 1))
    dist, ind = dist.ravel(), ind.ravel()

    used, pairs = set(), []
    for i in np.argsort(dist):                 # 距离近的优先配，减少浪费
        if dist[i] > caliper:
            break
        c = int(ind[i])
        if c in used:
            continue
        used.add(c)
        pairs.append((trt[i], ctrl[c]))
    return pairs


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--caliper", type=float, default=CALIPER_FRAC,
                    help="卡钳宽度（× logit(PS) 标准差），默认 0.2")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("PSM 倾向得分匹配：60 个月期限是否导致更高违约率？")
    print("=" * 60)

    df = load_clean(args.data).reset_index(drop=True)
    T = (df["term"] == TREATED_TERM).astype(int).values
    Y = df["bad_loan"].values
    X = build_covariates(df)

    print(f"\n[样本] 共 {len(df):,} 条")
    print(f"  处理组（{TREATED_TERM}）：{T.sum():,} 条，坏账率 {Y[T == 1].mean():.2%}")
    print(f"  对照组（{CONTROL_TERM}）：{(1 - T).sum():,} 条，坏账率 {Y[T == 0].mean():.2%}")

    # ---------- 未调整的朴素比较 ------------------------------------------
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    print(f"\n[朴素比较] 直接相减 = {naive * 100:+.2f}pp")
    print("  ⚠️ 这**不是**因果效应：选长期限的人本来风险就更高，混淆在里面。")

    # ---------- 估计倾向得分 ----------------------------------------------
    print(f"\n[协变量] {X.shape[1]} 个（含 one-hot）")
    for k, v in EXCLUDED.items():
        print(f"  · 刻意排除 {k}：{v}")

    lr = LogisticRegression(max_iter=3000, random_state=P.SEED)
    lr.fit(X, T)
    ps = lr.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    ps_logit = np.log(ps / (1 - ps))
    print(f"\n[倾向得分] 处理组均值 {ps[T == 1].mean():.3f}，"
          f"对照组均值 {ps[T == 0].mean():.3f}")
    print(f"  共同支撑区间：[{max(ps[T == 1].min(), ps[T == 0].min()):.4f}, "
          f"{min(ps[T == 1].max(), ps[T == 0].max()):.4f}]")

    # ---------- 匹配 -------------------------------------------------------
    caliper = args.caliper * ps_logit.std()
    pairs = match_without_replacement(ps_logit, T, caliper)
    ti = np.array([p[0] for p in pairs])
    ci = np.array([p[1] for p in pairs])

    print(f"\n[匹配] 卡钳 = {args.caliper} × {ps_logit.std():.3f} = {caliper:.4f}")
    print(f"  成功配对 {len(pairs):,} 对"
          f"（处理组 {T.sum():,} 条中匹配上 {len(pairs) / T.sum():.1%}）")

    # ---------- 平衡性检验 -------------------------------------------------
    print("\n[平衡性] 匹配前后各协变量的标准化均值差 |SMD|：")
    print("  （|SMD| < 0.1 视为平衡）")
    rows = []
    for col in X.columns:
        before = abs(smd(X[col].values[T == 1], X[col].values[T == 0]))
        after = abs(smd(X[col].values[ti], X[col].values[ci]))
        rows.append({"协变量": col, "匹配前|SMD|": before, "匹配后|SMD|": after})
    bal = pd.DataFrame(rows).sort_values("匹配前|SMD|", ascending=False)

    n_bad_before = (bal["匹配前|SMD|"] >= 0.1).sum()
    n_bad_after = (bal["匹配后|SMD|"] >= 0.1).sum()
    print(f"  匹配前：{n_bad_before}/{len(bal)} 个协变量不平衡（|SMD|≥0.1），"
          f"最大 |SMD| = {bal['匹配前|SMD|'].max():.3f}")
    print(f"  匹配后：{n_bad_after}/{len(bal)} 个协变量不平衡，"
          f"最大 |SMD| = {bal['匹配后|SMD|'].max():.3f}")
    print(f"  平均 |SMD| 由 {bal['匹配前|SMD|'].mean():.3f} "
          f"降至 {bal['匹配后|SMD|'].mean():.3f}")
    print("\n  匹配前最不平衡的 6 个：")
    print(bal.head(6).round(4).to_string(index=False))

    # ---------- ATT --------------------------------------------------------
    y_t = Y[ti].astype(float)
    y_c = Y[ci].astype(float)
    att = y_t.mean() - y_c.mean()

    rng = np.random.default_rng(P.SEED)
    boots = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(y_t), len(y_t))
        boots.append(y_t[b].mean() - y_c[b].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print("\n" + "-" * 60)
    print("[结果] 期限对违约率的因果效应（ATT）")
    print(f"  朴素比较（未调整）    ：{naive * 100:+.2f}pp")
    print(f"  PSM 匹配后（ATT）     ：{att * 100:+.2f}pp"
          f"   95% CI [{lo * 100:+.2f}pp, {hi * 100:+.2f}pp]")
    shrink = 1 - abs(att) / abs(naive) if naive != 0 else 0
    print(f"  → 匹配掉混淆后，效应从 {naive * 100:+.2f}pp 缩到 {att * 100:+.2f}pp，"
          f"**缩小了 {shrink:.0%}**")
    print("  → 说明朴素比较里的差异，很大一部分来自『选长期限的人本来风险就高』，"
          "而不是期限本身。")
    if lo <= 0 <= hi:
        print("  → ⚠️ 置信区间跨过 0，剩下的这点效应在统计上【不显著】。")
    else:
        print("  → 置信区间不包含 0，剩余的期限效应在统计上显著。")

    # ---------- 出图 -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].hist(ps[T == 0], bins=50, alpha=0.6, color="#2980b9",
                 label=f"对照组 {CONTROL_TERM}", density=True)
    axes[0].hist(ps[T == 1], bins=50, alpha=0.6, color="#c0392b",
                 label=f"处理组 {TREATED_TERM}", density=True)
    axes[0].set_xlabel("倾向得分 e(X) = P(选 60 个月 | X)")
    axes[0].set_ylabel("密度")
    axes[0].set_title("倾向得分分布\n两段分布重叠越好，匹配越可靠")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    top = bal.head(14).iloc[::-1]
    ypos = np.arange(len(top))
    axes[1].scatter(top["匹配前|SMD|"], ypos, color="#c0392b", s=45,
                    label="匹配前", zorder=3)
    axes[1].scatter(top["匹配后|SMD|"], ypos, color="#27ae60", s=45,
                    marker="s", label="匹配后", zorder=3)
    for i, (b_, a_) in enumerate(zip(top["匹配前|SMD|"], top["匹配后|SMD|"])):
        axes[1].plot([b_, a_], [i, i], color="#bdc3c7", lw=1.2, zorder=1)
    axes[1].axvline(0.1, color="#7f8c8d", ls="--", lw=1.2, label="平衡阈值 0.1")
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(top["协变量"], fontsize=8)
    axes[1].set_xlabel("标准化均值差 |SMD|")
    axes[1].set_title("平衡性检验（Love plot）\n匹配后全部落到阈值左侧才算成功")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("图 15｜PSM：把『本来就不是一批人』的用户配成可比的两组", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG)
    plt.close(fig)
    print(f"\n[出图] {FIG.relative_to(HERE)}")

    # ---------- 落盘 -------------------------------------------------------
    lines = [
        "PSM 倾向得分匹配：60 个月期限是否导致更高违约率？", "",
        f"处理组({TREATED_TERM}) {T.sum():,} 条，坏账率 {Y[T == 1].mean():.2%}",
        f"对照组({CONTROL_TERM}) {(1 - T).sum():,} 条，坏账率 {Y[T == 0].mean():.2%}",
        f"协变量 {X.shape[1]} 个（含 one-hot），刻意排除 "
        + "、".join(f"{k}（{v}）" for k, v in EXCLUDED.items()), "",
        "=== 效应估计 ===",
        f"朴素比较（未调整）：{naive * 100:+.2f}pp",
        f"PSM 匹配后（ATT）  ：{att * 100:+.2f}pp，"
        f"bootstrap 95% CI [{lo * 100:+.2f}pp, {hi * 100:+.2f}pp]",
        f"效应缩小 {shrink:.0%}", "",
        f"=== 匹配 ===",
        f"卡钳 {args.caliper} × {ps_logit.std():.4f} = {caliper:.4f}，"
        f"成功配对 {len(pairs):,} 对",
        f"匹配前 {n_bad_before}/{len(bal)} 个协变量不平衡，"
        f"匹配后 {n_bad_after}/{len(bal)} 个",
        f"平均 |SMD| {bal['匹配前|SMD|'].mean():.4f} → {bal['匹配后|SMD|'].mean():.4f}",
        "", "=== 平衡性明细（按匹配前 |SMD| 降序）===",
        bal.round(4).to_string(index=False), "",
        "注：PSM 只能控制【可观测】混淆。若存在未观测混淆，本估计仍有偏。",
        "    因此这是观察性研究证据，强度低于随机实验(RCT)。",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[落盘] {OUT.name}")


if __name__ == "__main__":
    main()
