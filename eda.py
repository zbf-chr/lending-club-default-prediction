# -*- coding: utf-8 -*-
"""
Lending Club —— EDA 可视化
==========================

复用 `pipeline.py` 的加载与清洗函数，保证【图和模型是同一套口径】。
这一点面试会被追问：如果 EDA 说坏账率 21.5%、模型说 18%，那就是两套口径，站不住。

产出（figures/ 目录）：

  01_target_definition.png   标签怎么定义的：三档分布（确定好 / 确定坏 / 剔除）
  02_bad_rate_by_grade.png   坏账率随评级单调上升 —— 验证「评级」这个风控核心变量有效
  03_bad_rate_by_fico_intrate.png  坏账率随 FICO / 利率分箱
  04_bad_rate_by_term_purpose.png  期限 / 借款用途
  05_bad_rate_by_region_year.png   地区 / 放款年份（时间趋势）
  06_bad_rate_by_dti_income.png    DTI / 年收入
  07_corr_with_target.png    ★ 与标签相关性 Top20 —— 直接暴露时间泄漏
  08_missing_top.png         缺失率 Top20 —— 解释为什么要按语义填充

用法：
    "D:/venvs/lending-club/Scripts/python.exe" eda.py
    "D:/venvs/lending-club/Scripts/python.exe" eda.py --sample 200000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，中文会乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")            # 无窗口后端：只存文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline as P              # 复用数据加载 / 清洗 / 业务定义

# 中文字体：Windows 11 自带微软雅黑；没有就退化到黑体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

FIG_DIR = Path(__file__).parent / "figures"
# 缓存用 pickle：零额外依赖，且能精确保留 dtype（parquet 需要 pyarrow）
CACHE = Path(__file__).parent / "data" / "_cache_raw_sample.pkl"

# 统一的配色：坏账相关的图一律用暖色系，避免每张图颜色随机
C_BAD = "#c0392b"
C_GOOD = "#27ae60"
C_NEUTRAL = "#7f8c8d"


# ===========================================================================
# 数据：带缓存的读取
# ===========================================================================
def get_raw_sample(data_path: str | None, sample_n: int, use_cache: bool) -> pd.DataFrame:
    """
    读原始抽样数据。全量 226 万行读一次要 2 分多钟，所以缓存成 parquet，
    改图的时候不用反复等。`--no-cache` 可强制重读。
    """
    if use_cache and CACHE.exists():
        df = pd.read_pickle(CACHE)
        print(f"[缓存] 命中 {CACHE.name}：{len(df):,} 行 × {df.shape[1]} 列")
        return df

    path = P.resolve_data_path(data_path)
    keep = set(P.RAW_COLS)
    df = pd.read_csv(path, usecols=lambda c: c in keep, low_memory=False)
    print(f"[加载] {path.name}：全量 {df.shape[0]:,} 行 × {df.shape[1]} 列")
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=P.SEED).reset_index(drop=True)
        print(f"[加载] 抽样至 {len(df):,} 行（random_state={P.SEED}）")

    if use_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(CACHE)
        print(f"[缓存] 已写入 {CACHE.name}")
    return df


# ===========================================================================
# 画图小工具
# ===========================================================================
def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.relative_to(FIG_DIR.parent)}")


def bad_rate_by(df: pd.DataFrame, col: str, bins=None, labels=None,
                top: int | None = None) -> pd.DataFrame:
    """
    按某个维度分组算坏账率 + 样本量。
    返回列：n / bad / bad_rate，按坏账率降序。
    """
    s = df[col]
    if bins is not None:
        s = pd.cut(s, bins=bins, labels=labels, include_lowest=True)
    g = df.assign(_k=s).groupby("_k", observed=True)["bad_loan"].agg(["size", "mean"])
    g = g.rename(columns={"size": "n", "mean": "bad_rate"})
    g["bad"] = (g["n"] * g["bad_rate"]).round().astype(int)
    g = g.sort_values("bad_rate", ascending=False)
    if top:
        g = g.head(top)
    return g.reset_index().rename(columns={"_k": col})


def bar_bad_rate(g: pd.DataFrame, key: str, ax, title: str,
                 overall: float | None = None, rotate: int = 0) -> None:
    """坏账率柱状图 + 样本量次坐标轴，并用虚线标出全局基准坏账率。"""
    x = np.arange(len(g))
    ax.bar(x, g["bad_rate"] * 100, color=C_BAD, alpha=0.85, label="坏账率")
    if overall is not None:
        ax.axhline(overall * 100, color=C_NEUTRAL, ls="--", lw=1.2,
                   label=f"全局基准 {overall:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(g[key].astype(str), rotation=rotate,
                       ha="right" if rotate else "center")
    ax.set_ylabel("坏账率 (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, g["n"], color="#2c3e50", marker="o", ms=3, lw=1, alpha=0.6)
    ax2.set_ylabel("样本量", color="#2c3e50")
    ax2.tick_params(axis="y", colors="#2c3e50")
    ax2.grid(False)


# ===========================================================================
# 各张图
# ===========================================================================
def fig01_target(df: pd.DataFrame) -> None:
    """标签定义：三档分布。这张图回答「什么算坏账、为什么剔掉 39% 的样本」。"""
    status = df["loan_status"].value_counts()

    def bucket(s: str) -> str:
        if s in P.BAD_STATUS:
            return "确定坏账"
        if s in P.GOOD_STATUS:
            return "确定好账"
        return "结局未定(剔除)"

    grp = status.rename_axis("loan_status").reset_index(name="n")
    grp["归类"] = grp["loan_status"].map(bucket)
    agg = grp.groupby("归类")["n"].sum().reindex(
        ["确定好账", "确定坏账", "结局未定(剔除)"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：三档占比
    colors = [C_GOOD, C_BAD, C_NEUTRAL]
    axes[0].bar(agg.index, agg.values, color=colors, alpha=0.9)
    total = agg.sum()
    for i, v in enumerate(agg.values):
        axes[0].text(i, v, f"{v:,}\n{v / total:.1%}", ha="center",
                     va="bottom", fontsize=9)
    axes[0].set_ylabel("记录数")
    axes[0].set_title(f"标签三档分布（抽样 {total:,} 条）\n"
                      f"剔除「结局未定」后坏账率 = "
                      f"{agg['确定坏账'] / (agg['确定坏账'] + agg['确定好账']):.1%}")
    axes[0].grid(axis="y", alpha=0.25)

    # 右：原始 loan_status 明细
    st = status.sort_values()
    axes[1].barh([s if len(s) < 42 else s[:40] + "…" for s in st.index],
                 st.values, color=C_NEUTRAL, alpha=0.9)
    for i, v in enumerate(st.values):
        axes[1].text(v, i, f" {v:,}", va="center", fontsize=8)
    axes[1].set_xlabel("记录数")
    axes[1].set_title("原始 loan_status 明细（长尾字段容易定义错）")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("图 1｜标签怎么定义 —— 这是全项目第一个业务判断", fontsize=13)
    _save(fig, "01_target_definition.png")


def fig02_grade(df: pd.DataFrame) -> None:
    """评级 grade：坏账率应单调上升，否则说明标签或特征有问题。"""
    g = bad_rate_by(df, "grade")
    order = list("ABCDEFG")
    g = g.set_index("grade").reindex([c for c in order if c in g["grade"].values])
    g = g.reset_index()
    overall = df["bad_loan"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_bad_rate(g, "grade", ax, "坏账率随评级（grade）的变化", overall=overall)
    ax.set_xlabel("Lending Club 内部评级（A 最优 → G 最差）")
    fig.suptitle("图 2｜评级是最强的风控变量：A→G 坏账率单调上升，"
                 "说明评级体系本身有效", fontsize=11)
    _save(fig, "02_bad_rate_by_grade.png")


def fig03_fico_intrate(df: pd.DataFrame) -> None:
    """FICO 与利率：两个放款前就能拿到的核心变量。"""
    overall = df["bad_loan"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fico = df["fico_range_low"]
    fb = bad_rate_by(df, "fico_range_low",
                     bins=[580, 640, 680, 700, 720, 740, 780, 850],
                     labels=["<640", "640-680", "680-700", "700-720",
                             "720-740", "740-780", "780+"])
    bar_bad_rate(fb, "fico_range_low", axes[0],
                 "坏账率随 FICO 分段（fico_range_low）", overall=overall)
    axes[0].set_xlabel("FICO 分段")

    rate = df["int_rate"]
    rb = bad_rate_by(df, "int_rate",
                     bins=[0, 8, 11, 14, 17, 20, 25, 35],
                     labels=["<8%", "8-11%", "11-14%", "14-17%",
                             "17-20%", "20-25%", "25%+"])
    bar_bad_rate(rb, "int_rate", axes[1],
                 "坏账率随利率分段（int_rate）", overall=overall)
    axes[1].set_xlabel("利率分段")

    fig.suptitle("图 3｜FICO 与利率：两个放款前可得的核心风险变量，"
                 "均呈单调关系", fontsize=12)
    _save(fig, "03_bad_rate_by_fico_intrate.png")


def fig04_term_purpose(df: pd.DataFrame) -> None:
    """期限与借款用途 —— 用途是最好的「业务解释性」维度。"""
    overall = df["bad_loan"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    t = bad_rate_by(df, "term")
    bar_bad_rate(t, "term", axes[0], "坏账率随期限", overall=overall)
    axes[0].set_xlabel("贷款期限")

    p = bad_rate_by(df, "purpose", top=12)
    bar_bad_rate(p, "purpose", axes[1], "坏账率随借款用途（Top12）",
                 overall=overall, rotate=40)
    axes[1].set_xlabel("借款用途")

    fig.suptitle("图 4｜期限与用途：60 个月坏账率显著更高；"
                 "小企业贷 / 创业贷是高风险用途", fontsize=12)
    _save(fig, "04_bad_rate_by_term_purpose.png")


def fig05_region_year(df: pd.DataFrame) -> None:
    """地区与放款年份：时间维度是风控里最容易被忽略的。"""
    overall = df["bad_loan"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    r = bad_rate_by(df, "region")
    bar_bad_rate(r, "region", axes[0], "坏账率随地区", overall=overall)

    if "issue_year" in df.columns:
        y = bad_rate_by(df, "issue_year")
        y = y.sort_values("issue_year")
        bar_bad_rate(y, "issue_year", axes[1], "坏账率随放款年份",
                     overall=overall)
        axes[1].set_xlabel("放款年份")

    fig.suptitle("图 5｜地区与时间：年份维度提示「用随机划分做验证会高估效果」，"
                 "真实上线应按时间外推", fontsize=11)
    _save(fig, "05_bad_rate_by_region_year.png")


def fig06_dti_income(df: pd.DataFrame) -> None:
    """DTI 与收入：负债收入比是审批的核心闸口。"""
    overall = df["bad_loan"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    d = df["dti"].clip(upper=50)
    db = bad_rate_by(df.assign(dti=d), "dti",
                     bins=[0, 5, 10, 15, 20, 25, 30, 50],
                     labels=["<5", "5-10", "10-15", "15-20",
                             "20-25", "25-30", "30+"])
    bar_bad_rate(db, "dti", axes[0], "坏账率随 DTI 分段（负债收入比）",
                 overall=overall)
    axes[0].set_xlabel("DTI 分段")

    inc = df["annual_inc"]
    ib = bad_rate_by(df, "annual_inc",
                     bins=[0, 40000, 60000, 80000, 110000, 160000, 1e7],
                     labels=["<4万", "4-6万", "6-8万", "8-11万",
                             "11-16万", "16万+"])
    bar_bad_rate(ib, "annual_inc", axes[1], "坏账率随年收入分段", overall=overall)
    axes[1].set_xlabel("年收入分段")

    fig.suptitle("图 6｜DTI 与收入：DTI 越高坏账率越高；"
                 "收入关系弱于 DTI，符合风控常识", fontsize=12)
    _save(fig, "06_bad_rate_by_dti_income.png")


def fig07_corr(df_leaky: pd.DataFrame) -> None:
    """
    ★ 最关键的一张图：与标签相关性 Top20。

    在【保留泄漏字段】的口径下算相关性，其中来自 POST_ORIGINATION 的字段标红。
    一看就知道：排最前面的几个全是「放款后」才产生的字段 —— 这就是时间泄漏的实证。
    """
    y = df_leaky["bad_loan"]
    num = df_leaky.select_dtypes(include=[np.number]).drop(
        columns=["bad_loan"], errors="ignore")

    corr = {}
    for col in num.columns:
        v = num[col]
        if v.nunique() < 2:          # 常量列相关性无意义
            continue
        corr[col] = abs(np.corrcoef(v.values, y.values)[0, 1])
    s = pd.Series(corr).dropna().sort_values(ascending=False).head(20)

    leaky_cols = set(P.POST_ORIGINATION) | set(P.LEAK_COLS)
    colors = [C_BAD if c in leaky_cols else "#2980b9" for c in s.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(s))[::-1], s.values, color=colors, alpha=0.9)
    ax.set_yticks(range(len(s))[::-1])
    ax.set_yticklabels(s.index, fontsize=9)
    ax.set_xlabel("|与标签的相关系数|")
    for i, v in zip(range(len(s))[::-1], s.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_BAD, label="放款后才产生的字段（时间泄漏，必须删）"),
        Patch(color="#2980b9", label="放款前可得（可用）"),
    ], loc="lower right", fontsize=9)

    ax.set_title("图 7｜与标签相关性 Top20（含泄漏字段口径）\n"
                 "红色字段在申请时点根本拿不到 —— 模型用了它们，"
                 "离线 AUC 才会虚高到 0.99", fontsize=12)
    _save(fig, "07_corr_with_target.png")


def fig08_missing(df_raw: pd.DataFrame) -> None:
    """缺失率 Top20：解释「为什么不能无脑 fillna(0)」。"""
    miss = df_raw.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(20)

    colors = [C_BAD if c in P.NEVER_HAPPENED else C_NEUTRAL for c in miss.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(miss))[::-1], miss.values * 100, color=colors, alpha=0.9)
    ax.set_yticks(range(len(miss))[::-1])
    ax.set_yticklabels(miss.index, fontsize=8)
    ax.set_xlabel("缺失率 (%)")
    for i, v in zip(range(len(miss))[::-1], miss.values):
        ax.text(v * 100, i, f" {v:.1%}", va="center", fontsize=8)
    ax.axvline(80, color=C_BAD, ls="--", lw=1.2)
    ax.text(80, len(miss) - 0.5, " 80% 剔除线", color=C_BAD, fontsize=8)
    ax.grid(axis="x", alpha=0.25)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_BAD, label='缺失 = "从未发生"（整个字段只有异常值才有值）'),
        Patch(color=C_NEUTRAL, label="其他缺失"),
    ], loc="lower right", fontsize=9)

    ax.set_title("图 8｜缺失率 Top20\n"
                 "红色字段的高缺失不是数据脏，而是「从未逾期的人根本没这个值」——"
                 "填 0 等于说反话", fontsize=12)
    _save(fig, "08_missing_top.png")


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="数据集路径，不填则自动查找")
    ap.add_argument("--sample", type=int, default=P.SAMPLE_N, help="抽样行数")
    ap.add_argument("--no-cache", action="store_true", help="忽略 parquet 缓存，重读原始数据")
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("EDA 可视化 —— 与 pipeline.py 同口径")
    print("=" * 60)
    FIG_DIR.mkdir(exist_ok=True)

    raw = get_raw_sample(args.data, args.sample, use_cache=not args.no_cache)

    print("\n[清洗] 正在按 pipeline 口径清洗（无泄漏版）...")
    clean = P.clean(raw)

    print("\n[清洗] 再跑一遍泄漏版口径（只用于图 7）...")
    leaky = P.clean(raw, leaky=True)

    print(f"\n[绘图] 输出到 {FIG_DIR}")
    fig01_target(raw)
    fig02_grade(clean)
    fig03_fico_intrate(clean)
    fig04_term_purpose(clean)
    fig05_region_year(clean)
    fig06_dti_income(clean)
    fig07_corr(leaky)
    fig08_missing(raw)

    print(f"\n完成，共 8 张图。坏账率基准 = {clean['bad_loan'].mean():.2%}")


if __name__ == "__main__":
    main()
