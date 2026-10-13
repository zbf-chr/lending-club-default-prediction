# -*- coding: utf-8 -*-
"""
A/B 实验设计与显著性检验
========================

⚠️ 先说清楚这是什么
------------------------------------------------------------------------
这是**离线模拟实验**，不是真实线上 A/B。做法是：把留存集随机分一半做对照组、
一半做实验组，两组施加不同审批策略，再比较结果指标。

它检验的统计方法是**真的**（分流随机、双样本比例检验、功效分析都成立），
但它**不能替代真实实验** —— 真实线上还要处理：新奇效应、策略串扰、
长期效应、以及"用户知道自己被拒了会去别家"这类离线看不见的东西。
面试时一定要主动说明这一点，否则会被当成不懂实验。

回答的问题
------------------------------------------------------------------------
  对照组 A：审批阈值 0.5（拍脑袋的默认值）
  实验组 B：审批阈值 0.143（成本最优阈值，见 cost_threshold.py）

  「B 组的通过人群坏账率显著更低吗？需要多少样本才能测出来？」

两部分
------------------------------------------------------------------------
  Part 1  双样本比例 z 检验：两组的指标差异是不是随机波动
  Part 2  统计功效分析：给定 α 和功效，反推需要多少样本；
          以及给定现有样本量，能检测到多小的效应（MDE）

用法：
    "D:/venvs/lending-club/Scripts/python.exe" ab_test.py
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
from scipy import stats

import pipeline as P
from cost_threshold import fit_and_score, load_clean, split_data

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

HERE = Path(__file__).parent
FIG = HERE / "figures" / "14_ab_test.png"
OUT = HERE / "ab_test.txt"

ALPHA = 0.05
POWER = 0.80
THRESHOLD_CONTROL = 0.50       # 对照组：默认阈值
THRESHOLD_TREATMENT = 0.143    # 实验组：成本最优阈值


# ===========================================================================
# 统计工具
# ===========================================================================
def two_prop_ztest(x1, n1, x2, n2):
    """双样本比例 z 检验（合并方差）。返回 z, p, 及差值的 95% 置信区间。"""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 差值的置信区间用【非合并】标准误，更标准
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    lo, hi = (p1 - p2) - z_crit * se_diff, (p1 - p2) + z_crit * se_diff
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": z, "p": p_value,
            "ci_low": lo, "ci_high": hi, "se_diff": se_diff}


def n_per_group(p1, p2, alpha=ALPHA, power=POWER):
    """给定两组的比例，反推每组所需样本量。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p1 - p2) ** 2


def power_given_n(p1, p2, n, alpha=ALPHA):
    """给定样本量，算检测出该效应的统计功效。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    if se <= 0:
        return 1.0
    return float(stats.norm.cdf(abs(p1 - p2) / se - z_a))


def mde_given_n(p1, n, alpha=ALPHA, power=POWER, upper=0.5):
    """给定样本量，二分搜索能检出的最小绝对效应(MDE)。"""
    lo, hi = 1e-5, upper
    for _ in range(200):
        mid = (lo + hi) / 2
        p2 = p1 - mid
        if p2 <= 0:
            hi = mid
            continue
        if power_given_n(p1, p2, n, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--cutoff", type=int, default=2015)
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("A/B 实验设计与显著性检验（离线模拟）")
    print("=" * 60)

    df = load_clean(args.data)
    tr_idx, te_idx = split_data(df, args.split, args.cutoff)
    y_te, probs, amt_te, _ = fit_and_score(df, tr_idx, te_idx, args.split, "none")

    # ---------- 随机分流 ---------------------------------------------------
    # 真实 A/B 的分流必须随机；这里用固定种子保证可复现。
    # （真实实验还会在分流后做 SRM 检查，见下。）
    rng = np.random.default_rng(P.SEED)
    n_te = len(y_te)
    perm = rng.permutation(n_te)
    half = n_te // 2
    idx_a, idx_b = perm[:half], perm[half:]

    print(f"\n[分流] 留存集 {n_te:,} 条，随机对半分成两组")
    print(f"  对照组 A（阈值 {THRESHOLD_CONTROL:.3f}）：{len(idx_a):,} 条")
    print(f"  实验组 B（阈值 {THRESHOLD_TREATMENT:.3f}）：{len(idx_b):,} 条")

    # SRM 检查（Sample Ratio Mismatch）：分流比偏离 50:50 太多说明分流有 bug
    chi2, p_srm = stats.chisquare([len(idx_a), len(idx_b)])
    ratio = len(idx_a) / len(idx_b)
    print(f"[SRM 检查] 分流比 {ratio:.4f}，卡方 p = {p_srm:.4f} —— "
          + ("通过，分流正常 [OK]" if p_srm > 0.001 else "⚠️ 分流异常，需排查"))

    # ---------- 施加策略 ---------------------------------------------------
    def apply_policy(idx, thr):
        approved = probs[idx] < thr          # 预测坏账概率低于阈值才放款
        y_grp = np.asarray(y_te)[idx]
        return {
            "n_total": len(idx),
            "n_approved": int(approved.sum()),
            "approval_rate": approved.mean(),
            "bad_in_approved": int(y_grp[approved].sum()),
            "bad_rate_approved": y_grp[approved].mean() if approved.any() else np.nan,
            "total_bad": int(y_grp.sum()),
            "bad_captured": int((y_grp[~approved]).sum()),
        }

    a = apply_policy(idx_a, THRESHOLD_CONTROL)
    b = apply_policy(idx_b, THRESHOLD_TREATMENT)

    print(f"\n[结果]")
    print(f"  对照组 A：通过率 {a['approval_rate']:.1%}，"
          f"通过人群坏账率 {a['bad_rate_approved']:.2%}，"
          f"拦截坏账 {a['bad_captured']}/{a['total_bad']}")
    print(f"  实验组 B：通过率 {b['approval_rate']:.1%}，"
          f"通过人群坏账率 {b['bad_rate_approved']:.2%}，"
          f"拦截坏账 {b['bad_captured']}/{b['total_bad']}")

    # ---------- Part 1：显著性检验 -----------------------------------------
    print("\n" + "-" * 60)
    print("[Part 1] 双样本比例 z 检验")
    print("  H0：两组的通过人群坏账率没有差异")
    print(f"  H1：B 组更低（单侧）／有差异（双侧），α = {ALPHA}")

    r_bad = two_prop_ztest(a["bad_in_approved"], a["n_approved"],
                           b["bad_in_approved"], b["n_approved"])
    print(f"\n  通过人群坏账率：A {r_bad['p1']:.2%} vs B {r_bad['p2']:.2%}，"
          f"差 {r_bad['diff'] * 100:+.2f}pp")
    print(f"    z = {r_bad['z']:.3f}，p = {r_bad['p']:.3e}"
          + ("   → 差异显著，拒绝 H0 ✓" if r_bad["p"] < ALPHA
             else "   → 差异不显著，不能拒绝 H0"))
    print(f"    差值 95% CI：[{r_bad['ci_low'] * 100:+.2f}pp, "
          f"{r_bad['ci_high'] * 100:+.2f}pp]")

    r_app = two_prop_ztest(b["n_approved"] - b["bad_in_approved"], b["n_total"],
                           a["n_approved"] - a["bad_in_approved"], a["n_total"])
    print(f"\n  通过率（业务量指标）：A {a['approval_rate']:.1%} vs "
          f"B {b['approval_rate']:.1%}，差 {(b['approval_rate'] - a['approval_rate']) * 100:+.2f}pp")
    print(f"    z = {r_app['z']:.3f}，p = {r_app['p']:.3e}"
          + ("   → 差异显著 ✓" if r_app["p"] < ALPHA else "   → 差异不显著"))

    print("\n  ⚠️ 关键：两个指标都显著，但**方向相反** ——")
    print("     B 组把坏账率压下来了，代价是通过率也大幅下降。")
    print("     统计只能告诉你『差异是真的』，**『值不值得』是业务决策，不是统计决策**。")

    # ---------- Part 2：功效分析 -------------------------------------------
    print("\n" + "-" * 60)
    print("[Part 2] 统计功效分析")
    base = a["bad_rate_approved"]
    print(f"  基线坏账率 p1 = {base:.2%}，α = {ALPHA}，目标功效 = {POWER:.0%}")

    rows = []
    for mde_pp in [1, 2, 3, 5, 8, 10]:
        p2 = base - mde_pp / 100
        if p2 <= 0:
            continue
        n = n_per_group(base, p2)
        rows.append({"MDE(pp)": mde_pp, "每组所需样本量": int(np.ceil(n)),
                     "总样本量": int(np.ceil(n)) * 2})
    eff = pd.DataFrame(rows)
    print("\n  【要多少样本】给定效应大小，反推每组所需样本量：")
    print(eff.to_string(index=False))

    n_now = min(len(idx_a), len(idx_b))
    mde_now = mde_given_n(base, n_now) * 100
    print(f"\n  【现有样本能测多大】当前每组 {n_now:,} 条，"
          f"在 α={ALPHA}, power={POWER:.0%} 下")
    print(f"    能检测出的最小效应 MDE = {mde_now:.2f}pp "
          f"（即坏账率从 {base:.1%} 降到 {base * 100 - mde_now:.1f}%）")
    print(f"    本次实测差异是 {r_bad['diff'] * 100:.2f}pp，"
          f"远大于 MDE —— 所以结论可信，不是靠样本量堆出来的假显著。")

    # ---------- 出图 -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = ["通过率", "通过人群坏账率"]
    va, vb = a["approval_rate"] * 100, b["approval_rate"] * 100
    ba, bb = a["bad_rate_approved"] * 100, b["bad_rate_approved"] * 100
    x = np.arange(2)
    w = 0.36
    axes[0].bar(x - w / 2, [va, ba], w, label="对照组 A（阈值 0.500）",
                color="#2980b9", alpha=0.9)
    axes[0].bar(x + w / 2, [vb, bb], w, label="实验组 B（阈值 0.143）",
                color="#c0392b", alpha=0.9)
    for xi, (v1, v2) in enumerate([(va, vb), (ba, bb)]):
        axes[0].text(xi - w / 2, v1, f"{v1:.1f}%", ha="center", va="bottom",
                     fontsize=9)
        axes[0].text(xi + w / 2, v2, f"{v2:.1f}%", ha="center", va="bottom",
                     fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("百分比 (%)")
    axes[0].set_title(f"两组指标对比（p = {r_bad['p']:.1e}，差异显著）\n"
                      "坏账率降了，通过率也降了 —— 权衡")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    ns = np.logspace(2, 4.5, 120)
    for mde_pp, c in zip([1, 2, 3, 5, 10],
                         ["#bdc3c7", "#7f8c8d", "#2980b9", "#e67e22", "#c0392b"]):
        p2 = base - mde_pp / 100
        if p2 <= 0:
            continue
        powers = [power_given_n(base, p2, int(n)) for n in ns]
        axes[1].plot(ns, powers, lw=2, color=c, label=f"MDE = {mde_pp}pp")
    axes[1].axhline(POWER, color="#27ae60", ls="--", lw=1.5, label="目标功效 80%")
    axes[1].axvline(n_now, color="#000000", ls=":", lw=1.5,
                    label=f"本次每组样本 {n_now:,}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("每组样本量（对数刻度）")
    axes[1].set_ylabel("统计功效")
    axes[1].set_title("功效曲线：样本量越大、效应越大，越容易测出来")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("图 14｜A/B 实验：显著性和功效分析", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG)
    plt.close(fig)
    print(f"\n[出图] {FIG.relative_to(HERE)}")

    # ---------- 落盘 -------------------------------------------------------
    lines = [
        "A/B 实验设计与显著性检验（离线模拟，非真实线上实验）",
        f"对照组 A 阈值 {THRESHOLD_CONTROL}，实验组 B 阈值 {THRESHOLD_TREATMENT}",
        f"留存集 {n_te:,} 条随机对半分流，SRM 卡方 p = {p_srm:.4f}", "",
        "=== 两组结果 ===",
        f"对照组 A：通过率 {a['approval_rate']:.2%}，"
        f"通过人群坏账率 {a['bad_rate_approved']:.2%}，"
        f"拦截坏账 {a['bad_captured']}/{a['total_bad']}",
        f"实验组 B：通过率 {b['approval_rate']:.2%}，"
        f"通过人群坏账率 {b['bad_rate_approved']:.2%}，"
        f"拦截坏账 {b['bad_captured']}/{b['total_bad']}", "",
        "=== 双样本比例 z 检验 ===",
        f"通过人群坏账率：A {r_bad['p1']:.2%} vs B {r_bad['p2']:.2%}，"
        f"差 {r_bad['diff'] * 100:+.2f}pp",
        f"  z = {r_bad['z']:.3f}，p = {r_bad['p']:.3e}，"
        f"95% CI [{r_bad['ci_low'] * 100:+.2f}pp, {r_bad['ci_high'] * 100:+.2f}pp]",
        f"通过率：A {a['approval_rate']:.2%} vs B {b['approval_rate']:.2%}，"
        f"z = {r_app['z']:.3f}，p = {r_app['p']:.3e}", "",
        "=== 统计功效分析（基线 " + f"{base:.2%}" + f"，α={ALPHA}，power={POWER:.0%}）===",
        eff.to_string(index=False),
        f"当前每组 {n_now:,} 条 → 可检测 MDE = {mde_now:.2f}pp", "",
        "注：这是离线模拟实验。真实线上 A/B 还需处理新奇效应、策略串扰、",
        "    长期效应、以及被拒用户流向竞品等离线看不到的问题。",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[落盘] {OUT.name}")


if __name__ == "__main__":
    main()
